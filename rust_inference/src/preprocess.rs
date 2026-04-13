//! Volume preprocessing: resampling, normalization, padding.

use crate::config::PreprocessConfig;
use ndarray::Array3;

/// Resample volume to target spacing using trilinear interpolation.
pub fn resample_to_spacing(
    volume: &Array3<f32>,
    source_spacing: [f64; 3],
    target_spacing: [f64; 3],
) -> Array3<f32> {
    let src_shape = volume.shape();
    let scale = [
        source_spacing[0] / target_spacing[0],
        source_spacing[1] / target_spacing[1],
        source_spacing[2] / target_spacing[2],
    ];
    let new_shape = [
        (src_shape[0] as f64 * scale[0]).round() as usize,
        (src_shape[1] as f64 * scale[1]).round() as usize,
        (src_shape[2] as f64 * scale[2]).round() as usize,
    ];

    if new_shape == [src_shape[0], src_shape[1], src_shape[2]] {
        return volume.clone();
    }

    trilinear_resample(volume, new_shape)
}

/// Resample segmentation to original spacing using nearest-neighbor.
pub fn resample_segmentation(
    seg: &Array3<u16>,
    target_shape: [usize; 3],
) -> Array3<u16> {
    let src_shape = seg.shape();
    if [src_shape[0], src_shape[1], src_shape[2]] == target_shape {
        return seg.clone();
    }

    let mut output = Array3::<u16>::zeros(target_shape);
    for z in 0..target_shape[0] {
        for y in 0..target_shape[1] {
            for x in 0..target_shape[2] {
                let sz = (z as f64 * src_shape[0] as f64 / target_shape[0] as f64).min((src_shape[0] - 1) as f64) as usize;
                let sy = (y as f64 * src_shape[1] as f64 / target_shape[1] as f64).min((src_shape[1] - 1) as f64) as usize;
                let sx = (x as f64 * src_shape[2] as f64 / target_shape[2] as f64).min((src_shape[2] - 1) as f64) as usize;
                output[[z, y, x]] = seg[[sz, sy, sx]];
            }
        }
    }
    output
}

/// Normalize volume according to nnUNet scheme.
pub fn normalize(volume: &mut Array3<f32>, config: &PreprocessConfig) {
    if config.channel_normalization.is_empty() {
        return;
    }
    let norm = &config.channel_normalization[0];

    match norm.scheme.as_str() {
        "CTNormalization" => {
            // Clip to percentile range, then z-score
            let lower = norm.percentile_00_5 as f32;
            let upper = norm.percentile_99_5 as f32;
            let mean = norm.mean as f32;
            let std = norm.std as f32;
            volume.mapv_inplace(|v| {
                let clipped = v.clamp(lower, upper);
                (clipped - mean) / std.max(1e-8)
            });
        }
        _ => {
            // ZScoreNormalization (default)
            let mean = norm.mean as f32;
            let std = norm.std as f32;
            volume.mapv_inplace(|v| (v - mean) / std.max(1e-8));
        }
    }
}

/// Pad volume to be divisible by patch size.
pub fn pad_to_patch_size(volume: &Array3<f32>, patch_size: [usize; 3]) -> (Array3<f32>, [usize; 3]) {
    let shape = volume.shape();
    let padded_shape = [
        next_multiple(shape[0], patch_size[0]),
        next_multiple(shape[1], patch_size[1]),
        next_multiple(shape[2], patch_size[2]),
    ];

    if padded_shape == [shape[0], shape[1], shape[2]] {
        return (volume.clone(), [0, 0, 0]);
    }

    let pad = [
        padded_shape[0] - shape[0],
        padded_shape[1] - shape[1],
        padded_shape[2] - shape[2],
    ];

    let mut padded = Array3::<f32>::zeros(padded_shape);
    padded
        .slice_mut(ndarray::s![..shape[0], ..shape[1], ..shape[2]])
        .assign(volume);

    (padded, pad)
}

fn next_multiple(value: usize, divisor: usize) -> usize {
    if value % divisor == 0 {
        value
    } else {
        value + divisor - (value % divisor)
    }
}

fn trilinear_resample(volume: &Array3<f32>, new_shape: [usize; 3]) -> Array3<f32> {
    let src = volume.shape();
    let mut output = Array3::<f32>::zeros(new_shape);

    for z in 0..new_shape[0] {
        let fz = z as f64 * (src[0] - 1) as f64 / (new_shape[0] - 1).max(1) as f64;
        let z0 = (fz.floor() as usize).min(src[0] - 1);
        let z1 = (z0 + 1).min(src[0] - 1);
        let dz = (fz - z0 as f64) as f32;

        for y in 0..new_shape[1] {
            let fy = y as f64 * (src[1] - 1) as f64 / (new_shape[1] - 1).max(1) as f64;
            let y0 = (fy.floor() as usize).min(src[1] - 1);
            let y1 = (y0 + 1).min(src[1] - 1);
            let dy = (fy - y0 as f64) as f32;

            for x in 0..new_shape[2] {
                let fx = x as f64 * (src[2] - 1) as f64 / (new_shape[2] - 1).max(1) as f64;
                let x0 = (fx.floor() as usize).min(src[2] - 1);
                let x1 = (x0 + 1).min(src[2] - 1);
                let dx = (fx - x0 as f64) as f32;

                // Trilinear interpolation
                let c000 = volume[[z0, y0, x0]];
                let c001 = volume[[z0, y0, x1]];
                let c010 = volume[[z0, y1, x0]];
                let c011 = volume[[z0, y1, x1]];
                let c100 = volume[[z1, y0, x0]];
                let c101 = volume[[z1, y0, x1]];
                let c110 = volume[[z1, y1, x0]];
                let c111 = volume[[z1, y1, x1]];

                let c00 = c000 * (1.0 - dx) + c001 * dx;
                let c01 = c010 * (1.0 - dx) + c011 * dx;
                let c10 = c100 * (1.0 - dx) + c101 * dx;
                let c11 = c110 * (1.0 - dx) + c111 * dx;

                let c0 = c00 * (1.0 - dy) + c01 * dy;
                let c1 = c10 * (1.0 - dy) + c11 * dy;

                output[[z, y, x]] = c0 * (1.0 - dz) + c1 * dz;
            }
        }
    }
    output
}
