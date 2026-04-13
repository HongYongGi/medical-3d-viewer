//! ONNX Runtime sliding window inference with optional TTA (mirroring).

use anyhow::{Context, Result};
use ndarray::{Array3, Array4, s};
use std::path::Path;

use crate::progress::ProgressReporter;

/// Load ONNX model session.
pub fn load_model(model_path: &Path, use_cuda: bool) -> Result<ort::session::Session> {
    let builder = ort::session::Session::builder()
        .map_err(|e| anyhow::anyhow!("Session builder error: {e}"))?;

    let mut builder = if use_cuda {
        builder
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .unwrap_or_else(|_| {
                eprintln!("CUDA not available, falling back to CPU");
                ort::session::Session::builder().unwrap()
            })
    } else {
        builder
    };

    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load ONNX {}: {e}", model_path.display()))?;

    Ok(session)
}

/// Run sliding window inference on a preprocessed volume.
pub fn sliding_window_inference(
    session: &mut ort::session::Session,
    volume: &Array3<f32>,
    patch_size: [usize; 3],
    num_classes: usize,
    tile_step_size: f64,
    use_mirroring: bool,
    mirror_axes: &[usize],
    progress: &mut ProgressReporter,
) -> Result<Array3<u16>> {
    let shape = volume.shape();
    let [d, h, w] = [shape[0], shape[1], shape[2]];

    let stride = [
        (patch_size[0] as f64 * tile_step_size).max(1.0) as usize,
        (patch_size[1] as f64 * tile_step_size).max(1.0) as usize,
        (patch_size[2] as f64 * tile_step_size).max(1.0) as usize,
    ];

    let positions = compute_patch_positions([d, h, w], patch_size, stride);
    let total_patches = positions.len();

    let mut aggregated = Array4::<f32>::zeros([num_classes, d, h, w]);
    let mut weight_map = Array3::<f32>::zeros([d, h, w]);
    let gaussian = generate_gaussian_importance(patch_size);

    progress.report(0.3, &format!("추론 시작 ({total_patches} patches)"));

    for (idx, &(pz, py, px)) in positions.iter().enumerate() {
        let pz_end = (pz + patch_size[0]).min(d);
        let py_end = (py + patch_size[1]).min(h);
        let px_end = (px + patch_size[2]).min(w);

        let patch = volume.slice(s![pz..pz_end, py..py_end, px..px_end]).to_owned();
        let padded = pad_patch(&patch, patch_size);

        let prediction = if use_mirroring {
            infer_with_mirroring(session, &padded, mirror_axes)?
        } else {
            infer_patch(session, &padded)?
        };

        let actual_d = pz_end - pz;
        let actual_h = py_end - py;
        let actual_w = px_end - px;

        for c in 0..num_classes {
            for z in 0..actual_d {
                for y in 0..actual_h {
                    for x in 0..actual_w {
                        let g = gaussian[[z, y, x]];
                        aggregated[[c, pz + z, py + y, px + x]] += prediction[[c, z, y, x]] * g;
                    }
                }
            }
        }
        for z in 0..actual_d {
            for y in 0..actual_h {
                for x in 0..actual_w {
                    weight_map[[pz + z, py + y, px + x]] += gaussian[[z, y, x]];
                }
            }
        }

        let frac = 0.3 + 0.6 * (idx + 1) as f64 / total_patches as f64;
        progress.report(frac, &format!("추론 중... (patch {}/{total_patches})", idx + 1));
    }

    // Normalize by weights
    for c in 0..num_classes {
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let w_val = weight_map[[z, y, x]];
                    if w_val > 0.0 {
                        aggregated[[c, z, y, x]] /= w_val;
                    }
                }
            }
        }
    }

    // Argmax
    progress.report(0.9, "후처리 중 (argmax)...");
    let mut segmentation = Array3::<u16>::zeros([d, h, w]);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_cls = 0u16;
                for c in 0..num_classes {
                    let v = aggregated[[c, z, y, x]];
                    if v > max_val {
                        max_val = v;
                        max_cls = c as u16;
                    }
                }
                segmentation[[z, y, x]] = max_cls;
            }
        }
    }

    Ok(segmentation)
}

fn infer_patch(session: &mut ort::session::Session, patch: &Array3<f32>) -> Result<Array4<f32>> {
    let [d, h, w] = [patch.shape()[0], patch.shape()[1], patch.shape()[2]];

    // Build [1, 1, D, H, W] input
    let raw: Vec<f32> = patch.iter().copied().collect();
    let input = ndarray::Array5::<f32>::from_shape_vec([1, 1, d, h, w], raw)
        .context("Failed to reshape input")?;

    let input_tensor = ort::value::Tensor::from_array(input)
        .map_err(|e| anyhow::anyhow!("Tensor creation error: {e}"))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("Inference error: {e}"))?;

    let (output_shape_ref, output_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Output extraction error: {e}"))?;

    let nc = output_shape_ref[1] as usize;
    let od = output_shape_ref[2] as usize;
    let oh = output_shape_ref[3] as usize;
    let ow = output_shape_ref[4] as usize;

    // Copy to owned Array4 [nc, od, oh, ow]
    let raw_out: Vec<f32> = output_data.iter().copied().collect();
    let out_5d = ndarray::Array5::<f32>::from_shape_vec([1, nc, od, oh, ow], raw_out)
        .context("Reshape output")?;
    let result = out_5d.index_axis(ndarray::Axis(0), 0).to_owned();

    // Softmax
    let mut softmaxed = result.clone();
    for z in 0..od {
        for y in 0..oh {
            for x in 0..ow {
                let mut max_val = f32::NEG_INFINITY;
                for c in 0..nc {
                    max_val = max_val.max(result[[c, z, y, x]]);
                }
                let mut sum_exp = 0.0f32;
                for c in 0..nc {
                    let e = (result[[c, z, y, x]] - max_val).exp();
                    softmaxed[[c, z, y, x]] = e;
                    sum_exp += e;
                }
                for c in 0..nc {
                    softmaxed[[c, z, y, x]] /= sum_exp;
                }
            }
        }
    }

    Ok(softmaxed)
}

fn infer_with_mirroring(
    session: &mut ort::session::Session,
    patch: &Array3<f32>,
    mirror_axes: &[usize],
) -> Result<Array4<f32>> {
    let base = infer_patch(session, patch)?;
    let num_mirrors = 1usize << mirror_axes.len();
    let mut accumulated = base.clone();

    for mirror_idx in 1..num_mirrors {
        let mut mirrored = patch.clone();
        for (bit, &axis) in mirror_axes.iter().enumerate() {
            if mirror_idx & (1 << bit) != 0 {
                mirrored.invert_axis(ndarray::Axis(axis));
            }
        }

        let mut pred = infer_patch(session, &mirrored)?;

        // Flip prediction back (axis + 1 for class dimension)
        for (bit, &axis) in mirror_axes.iter().enumerate() {
            if mirror_idx & (1 << bit) != 0 {
                pred.invert_axis(ndarray::Axis(axis + 1));
            }
        }

        accumulated += &pred;
    }

    accumulated /= num_mirrors as f32;
    Ok(accumulated)
}

fn compute_patch_positions(
    volume_shape: [usize; 3],
    patch_size: [usize; 3],
    stride: [usize; 3],
) -> Vec<(usize, usize, usize)> {
    let mut positions = Vec::new();

    let steps = |dim: usize, patch: usize, step: usize| -> Vec<usize> {
        let mut v = Vec::new();
        let mut pos = 0;
        while pos + patch <= dim {
            v.push(pos);
            pos += step;
        }
        if v.is_empty() || *v.last().unwrap() + patch < dim {
            v.push(dim.saturating_sub(patch));
        }
        v.sort();
        v.dedup();
        v
    };

    for z in steps(volume_shape[0], patch_size[0], stride[0]) {
        for y in steps(volume_shape[1], patch_size[1], stride[1]) {
            for x in steps(volume_shape[2], patch_size[2], stride[2]) {
                positions.push((z, y, x));
            }
        }
    }
    positions
}

fn pad_patch(patch: &Array3<f32>, target_size: [usize; 3]) -> Array3<f32> {
    let shape = patch.shape();
    if shape[0] >= target_size[0] && shape[1] >= target_size[1] && shape[2] >= target_size[2] {
        return patch.clone();
    }
    let mut padded = Array3::<f32>::zeros(target_size);
    let d = shape[0].min(target_size[0]);
    let h = shape[1].min(target_size[1]);
    let w = shape[2].min(target_size[2]);
    padded.slice_mut(s![..d, ..h, ..w]).assign(&patch.slice(s![..d, ..h, ..w]));
    padded
}

fn generate_gaussian_importance(patch_size: [usize; 3]) -> Array3<f32> {
    let sigma = 0.125;
    let mut importance = Array3::<f32>::zeros(patch_size);
    for z in 0..patch_size[0] {
        let gz = gaussian_1d(z, patch_size[0], sigma);
        for y in 0..patch_size[1] {
            let gy = gaussian_1d(y, patch_size[1], sigma);
            for x in 0..patch_size[2] {
                let gx = gaussian_1d(x, patch_size[2], sigma);
                importance[[z, y, x]] = gz * gy * gx;
            }
        }
    }
    let max_val = importance.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_val > 0.0 {
        importance /= max_val;
    }
    importance
}

fn gaussian_1d(pos: usize, length: usize, sigma_scale: f64) -> f32 {
    let center = (length - 1) as f64 / 2.0;
    let sigma = length as f64 * sigma_scale;
    let diff = pos as f64 - center;
    (-(diff * diff) / (2.0 * sigma * sigma)).exp() as f32
}
