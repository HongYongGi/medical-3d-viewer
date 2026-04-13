//! Configuration parsing for nnUNet preprocessing parameters.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct ChannelNormalization {
    pub scheme: String,
    pub mean: f64,
    pub std: f64,
    pub percentile_00_5: f64,
    pub percentile_99_5: f64,
}

#[derive(Debug, Deserialize)]
pub struct PreprocessConfig {
    pub configuration: String,
    pub patch_size: Vec<usize>,
    pub spacing: Vec<f64>,
    pub num_input_channels: usize,
    pub num_classes: usize,
    pub channel_normalization: Vec<ChannelNormalization>,
    pub transpose_forward: Vec<usize>,
    pub transpose_backward: Vec<usize>,
    pub use_mirroring: bool,
    pub mirror_axes: Vec<usize>,
    pub tile_step_size: f64,
}

impl PreprocessConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: PreprocessConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn patch_size_3d(&self) -> [usize; 3] {
        [
            self.patch_size.get(0).copied().unwrap_or(128),
            self.patch_size.get(1).copied().unwrap_or(128),
            self.patch_size.get(2).copied().unwrap_or(128),
        ]
    }

    pub fn target_spacing(&self) -> [f64; 3] {
        [
            self.spacing.get(0).copied().unwrap_or(1.0),
            self.spacing.get(1).copied().unwrap_or(1.0),
            self.spacing.get(2).copied().unwrap_or(1.0),
        ]
    }
}
