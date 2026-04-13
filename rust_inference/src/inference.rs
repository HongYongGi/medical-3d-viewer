//! ONNX Runtime sliding window inference with optional TTA (mirroring).

use anyhow::{Context, Result};
use ndarray::{Array3, Array5, s};
use ort::session::Session;
use std::path::Path;

use crate::progress::ProgressReporter;

/// Load ONNX model session.
pub fn load_model(model_path: &Path, use_cuda: bool) -> Result<Session> {
    let mut builder = Session::builder()?;

    if use_cuda {
        // Try CUDA, fall back to CPU
        match builder.clone().with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default().build(),
        ]) {
            Ok(b) => builder = b,
            Err(_) => eprintln!("CUDA not available, falling back to CPU"),
        }
    }

    let session = builder
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX: {}", model_path.display()))?;

    Ok(session)
}

/// Run sliding window inference on a preprocessed volume.
pub fn sliding_window_inference(
    session: &Session,
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

    // Compute stride
    let stride = [
        (patch_size[0] as f64 * tile_step_size).max(1.0) as usize,
        (patch_size[1] as f64 * tile_step_size).max(1.0) as usize,
        (patch_size[2] as f64 * tile_step_size).max(1.0) as usize,
    ];

    // Compute patch positions
    let positions = compute_patch_positions([d, h, w], patch_size, stride);
    let total_patches = positions.len();
    let mirror_count = if use_mirroring { 1 << mirror_axes.len() } else { 1 };

    // Accumulation buffers
    let mut aggregated = ndarray::Array4::<f32>::zeros([num_classes, d, h, w]);
    let mut weight_map = Array3::<f32>::zeros([d, h, w]);

    // Gaussian importance map for overlap blending
    let gaussian = generate_gaussian_importance(patch_size);

    progress.report(0.3, &format!("추론 시작 ({total_patches} patches, mirror={mirror_count}x)"));

    for (idx, &(pz, py, px)) in positions.iter().enumerate() {
        let pz_end = (pz + patch_size[0]).min(d);
        let py_end = (py + patch_size[1]).min(h);
        let px_end = (px + patch_size[2]).min(w);

        // Extract patch
        let patch = volume.slice(s![pz..pz_end, py..py_end, px..px_end]).to_owned();

        // Pad if patch is smaller than patch_size
        let padded = pad_patch(&patch, patch_size);

        // Run inference (with optional mirroring)
        let prediction = if use_mirroring {
            infer_with_mirroring(session, &padded, mirror_axes)?
        } else {
            infer_patch(session, &padded)?
        };

        // Accumulate results
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

fn infer_patch(session: &Session, patch: &Array3<f32>) -> Result<ndarray::Array4<f32>> {
    let shape = patch.shape();
    // Reshape to [1, 1, D, H, W] (batch=1, channel=1)
    let input = patch
        .clone()
        .into_shape_with_order([1, 1, shape[0], shape[1], shape[2]])?;

    let input_values = ort::value::Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![input_values]?)?;

    let output = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;

    // Output: [1, num_classes, D, H, W] → [num_classes, D, H, W]
    let output_shape = output.shape();
    let result = output
        .to_owned()
        .into_shape_with_order([output_shape[1], output_shape[2], output_shape[3], output_shape[4]])?;

    // Apply softmax
    let mut softmaxed = result.clone();
    for z in 0..output_shape[2] {
        for y in 0..output_shape[3] {
            for x in 0..output_shape[4] {
                let mut max_val = f32::NEG_INFINITY;
                for c in 0..output_shape[1] {
                    max_val = max_val.max(result[[c, z, y, x]]);
                }
                let mut sum_exp = 0.0f32;
                for c in 0..output_shape[1] {
                    let e = (result[[c, z, y, x]] - max_val).exp();
                    softmaxed[[c, z, y, x]] = e;
                    sum_exp += e;
                }
                for c in 0..output_shape[1] {
                    softmaxed[[c, z, y, x]] /= sum_exp;
                }
            }
        }
    }

    Ok(softmaxed)
}

fn infer_with_mirroring(
    session: &Session,
    patch: &Array3<f32>,
    mirror_axes: &[usize],
) -> Result<ndarray::Array4<f32>> {
    let base = infer_patch(session, patch)?;
    let num_mirrors = 1 << mirror_axes.len();
    let mut accumulated = base.clone();

    for mirror_idx in 1..num_mirrors {
        // Create mirrored input
        let mut mirrored = patch.clone();
        for (bit, &axis) in mirror_axes.iter().enumerate() {
            if mirror_idx & (1 << bit) != 0 {
                flip_axis_inplace(&mut mirrored, axis);
            }
        }

        let mut pred = infer_patch(session, &mirrored)?;

        // Flip prediction back
        for (bit, &axis) in mirror_axes.iter().enumerate() {
            if mirror_idx & (1 << bit) != 0 {
                flip_axis_4d_inplace(&mut pred, axis + 1); // +1 because pred has class dim
            }
        }

        accumulated += &pred;
    }

    accumulated /= num_mirrors as f32;
    Ok(accumulated)
}

fn flip_axis_inplace(arr: &mut Array3<f32>, axis: usize) {
    let len = arr.shape()[axis];
    for i in 0..len / 2 {
        let j = len - 1 - i;
        // Swap slices along axis
        let shape = arr.shape().to_vec();
        for idx in ndarray::indices(&shape[..]) {
            let mut idx_i = idx.clone().into_pattern();
            let mut idx_j = idx.clone().into_pattern();
            // Only swap when the axis index matches
            match axis {
                0 if idx.as_array_view()[0] == i => {
                    let vi = arr[idx.as_array_view().as_slice().unwrap()];
                    // This is simplified; real implementation would use raw slicing
                    let _ = vi;
                }
                _ => {}
            }
        }
        // Simplified: use slice assignment
        break; // Placeholder
    }
    // Use ndarray's built-in slice reversal
    arr.invert_axis(ndarray::Axis(axis));
}

fn flip_axis_4d_inplace(arr: &mut ndarray::Array4<f32>, axis: usize) {
    arr.invert_axis(ndarray::Axis(axis));
}

fn compute_patch_positions(
    volume_shape: [usize; 3],
    patch_size: [usize; 3],
    stride: [usize; 3],
) -> Vec<(usize, usize, usize)> {
    let mut positions = Vec::new();
    let mut z = 0;
    while z + patch_size[0] <= volume_shape[0] || z == 0 {
        let mut y = 0;
        while y + patch_size[1] <= volume_shape[1] || y == 0 {
            let mut x = 0;
            while x + patch_size[2] <= volume_shape[2] || x == 0 {
                positions.push((z, y, x));
                x += stride[2];
                if x + patch_size[2] > volume_shape[2] && x < volume_shape[2] {
                    x = volume_shape[2].saturating_sub(patch_size[2]);
                    positions.push((z, y, x));
                    break;
                }
            }
            y += stride[1];
            if y + patch_size[1] > volume_shape[1] && y < volume_shape[1] {
                y = volume_shape[1].saturating_sub(patch_size[1]);
            } else if y + patch_size[1] > volume_shape[1] {
                break;
            }
        }
        z += stride[0];
        if z + patch_size[0] > volume_shape[0] && z < volume_shape[0] {
            z = volume_shape[0].saturating_sub(patch_size[0]);
        } else if z + patch_size[0] > volume_shape[0] {
            break;
        }
    }

    positions.sort();
    positions.dedup();
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
    // Normalize so max = 1
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
