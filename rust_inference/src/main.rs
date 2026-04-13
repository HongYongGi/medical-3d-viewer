//! nnUNet ONNX inference CLI.
//!
//! Fast nnUNet-compatible inference using ONNX Runtime.
//! Designed to be called from Python via subprocess.

mod config;
mod inference;
mod nifti_io;
mod postprocess;
mod preprocess;
mod progress;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use crate::config::PreprocessConfig;
use crate::progress::ProgressReporter;

#[derive(Parser)]
#[command(name = "nnunet-infer", about = "Fast nnUNet inference via ONNX Runtime")]
struct Cli {
    /// Path to ONNX model file
    #[arg(long)]
    model: PathBuf,

    /// Path to preprocess_config.json
    #[arg(long)]
    config: PathBuf,

    /// Input NIfTI file (.nii.gz)
    #[arg(long)]
    input: PathBuf,

    /// Output segmentation NIfTI file
    #[arg(long)]
    output: PathBuf,

    /// Device: "cuda" or "cpu"
    #[arg(long, default_value = "cuda")]
    device: String,

    /// Enable test-time augmentation (mirroring)
    #[arg(long, default_value = "true")]
    mirror: bool,

    /// Tile step size for sliding window (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    tile_step_size: f64,

    /// Output progress as JSON to stdout
    #[arg(long, default_value = "true")]
    progress: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut progress = ProgressReporter::new(cli.progress);

    // 1. Load configuration
    progress.report(0.0, "설정 로딩 중...");
    let config = PreprocessConfig::load(&cli.config)?;

    // 2. Load NIfTI input
    progress.report(0.05, "NIfTI 파일 로딩 중...");
    let volume = nifti_io::load_nifti(&cli.input)?;
    let original_shape = [
        volume.data.shape()[0],
        volume.data.shape()[1],
        volume.data.shape()[2],
    ];

    // 3. Preprocess
    progress.report(0.1, "전처리 중 (리샘플링)...");
    let mut resampled = preprocess::resample_to_spacing(
        &volume.data,
        volume.spacing,
        config.target_spacing(),
    );

    progress.report(0.15, "전처리 중 (정규화)...");
    preprocess::normalize(&mut resampled, &config);

    progress.report(0.2, "전처리 중 (패딩)...");
    let resampled_shape = [
        resampled.shape()[0],
        resampled.shape()[1],
        resampled.shape()[2],
    ];
    let (padded, _pad_amount) = preprocess::pad_to_patch_size(&resampled, config.patch_size_3d());

    // 4. Load ONNX model
    progress.report(0.25, "ONNX 모델 로딩 중...");
    let use_cuda = cli.device == "cuda";
    let session = inference::load_model(&cli.model, use_cuda)?;

    // 5. Sliding window inference
    let segmentation = inference::sliding_window_inference(
        &session,
        &padded,
        config.patch_size_3d(),
        config.num_classes,
        cli.tile_step_size,
        cli.mirror,
        &config.mirror_axes,
        &mut progress,
    )?;

    // 6. Post-process
    progress.report(0.92, "후처리 중 (언패딩)...");
    let unpadded = postprocess::unpad(&segmentation, resampled_shape);

    progress.report(0.95, "후처리 중 (리샘플링)...");
    let final_seg = postprocess::resample_to_original(&unpadded, original_shape);

    // 7. Save output
    progress.report(0.98, "NIfTI 저장 중...");
    nifti_io::save_nifti_segmentation(&final_seg, &volume, &cli.output)?;

    progress.report(1.0, "완료!");
    Ok(())
}
