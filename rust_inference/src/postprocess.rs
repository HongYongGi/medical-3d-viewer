//! Post-processing: reverse resampling, unpadding.

use crate::preprocess;
use ndarray::Array3;

/// Remove padding added during preprocessing.
pub fn unpad(seg: &Array3<u16>, original_shape: [usize; 3]) -> Array3<u16> {
    let shape = seg.shape();
    if [shape[0], shape[1], shape[2]] == original_shape {
        return seg.clone();
    }

    seg.slice(ndarray::s![
        ..original_shape[0],
        ..original_shape[1],
        ..original_shape[2]
    ])
    .to_owned()
}

/// Resample segmentation back to original spacing/shape.
pub fn resample_to_original(
    seg: &Array3<u16>,
    original_shape: [usize; 3],
) -> Array3<u16> {
    preprocess::resample_segmentation(seg, original_shape)
}
