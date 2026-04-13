//! NIfTI file reading and writing.

use anyhow::{Context, Result};
use ndarray::Array3;
use std::path::Path;

/// Volume data loaded from a NIfTI file.
pub struct NiftiVolume {
    pub data: Array3<f32>,
    pub affine: [[f64; 4]; 4],
    pub spacing: [f64; 3],
    pub header_bytes: Vec<u8>,
}

/// Load a NIfTI file and return volume data as f32 Array3.
pub fn load_nifti(path: &Path) -> Result<NiftiVolume> {
    use nifti::{NiftiObject, ReaderOptions};

    let obj = ReaderOptions::new()
        .read_file(path)
        .with_context(|| format!("Failed to read NIfTI: {}", path.display()))?;

    let header = obj.header();
    let pixdim = header.pixdim;
    let spacing = [pixdim[1] as f64, pixdim[2] as f64, pixdim[3] as f64];
    let affine = extract_affine(header);

    let dim = &header.dim;
    let shape = [dim[1] as usize, dim[2] as usize, dim[3] as usize];

    // Get raw volume data and convert to our ndarray version
    let volume = obj.into_volume();
    // Use nifti's ndarray (0.15) and convert to raw data, then rebuild with our ndarray (0.17)
    use nifti::IntoNdArray;
    let nifti_arr = volume
        .into_ndarray::<f32>()
        .context("Failed to convert NIfTI to f32")?;

    // Copy raw data into our ndarray 0.17
    let raw_data: Vec<f32> = nifti_arr.iter().copied().collect();
    let data = Array3::from_shape_vec(shape, raw_data)
        .context("Failed to reshape volume to 3D")?;

    // Serialize header for later use when saving
    let header_bytes = Vec::new(); // Placeholder

    Ok(NiftiVolume {
        data,
        affine,
        spacing,
        header_bytes,
    })
}

/// Save a segmentation volume as NIfTI.
pub fn save_nifti_segmentation(
    data: &Array3<u16>,
    reference: &NiftiVolume,
    output_path: &Path,
) -> Result<()> {
    use std::io::Write;

    let shape = data.shape();
    let raw: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();

    // Build minimal NIfTI-1 header (348 bytes)
    let mut header = [0u8; 348];

    // sizeof_hdr
    header[0..4].copy_from_slice(&348i32.to_le_bytes());
    // dim
    header[40..42].copy_from_slice(&3i16.to_le_bytes()); // ndim
    header[42..44].copy_from_slice(&(shape[0] as i16).to_le_bytes());
    header[44..46].copy_from_slice(&(shape[1] as i16).to_le_bytes());
    header[46..48].copy_from_slice(&(shape[2] as i16).to_le_bytes());
    // datatype = uint16 (512)
    header[70..72].copy_from_slice(&512i16.to_le_bytes());
    // bitpix = 16
    header[72..74].copy_from_slice(&16i16.to_le_bytes());
    // pixdim
    header[76..80].copy_from_slice(&1.0f32.to_le_bytes()); // pixdim[0]
    header[80..84].copy_from_slice(&(reference.spacing[0] as f32).to_le_bytes());
    header[84..88].copy_from_slice(&(reference.spacing[1] as f32).to_le_bytes());
    header[88..92].copy_from_slice(&(reference.spacing[2] as f32).to_le_bytes());
    // vox_offset
    header[108..112].copy_from_slice(&352.0f32.to_le_bytes());
    // sform_code = 1
    header[254..256].copy_from_slice(&1i16.to_le_bytes());
    // srow_x, srow_y, srow_z (from affine)
    for i in 0..4 {
        let offset = 280 + i * 4;
        header[offset..offset + 4].copy_from_slice(&(reference.affine[0][i] as f32).to_le_bytes());
    }
    for i in 0..4 {
        let offset = 296 + i * 4;
        header[offset..offset + 4].copy_from_slice(&(reference.affine[1][i] as f32).to_le_bytes());
    }
    for i in 0..4 {
        let offset = 312 + i * 4;
        header[offset..offset + 4].copy_from_slice(&(reference.affine[2][i] as f32).to_le_bytes());
    }
    // magic = "n+1\0"
    header[344..348].copy_from_slice(b"n+1\0");

    // Write as .nii.gz
    let file = std::fs::File::create(output_path)
        .with_context(|| format!("Failed to create: {}", output_path.display()))?;
    let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());

    // Header (348 bytes) + 4 bytes padding = 352 vox_offset
    gz.write_all(&header)?;
    gz.write_all(&[0u8; 4])?; // padding to 352
    gz.write_all(&raw)?;
    gz.finish()?;

    Ok(())
}

fn extract_affine(header: &nifti::NiftiHeader) -> [[f64; 4]; 4] {
    let mut affine = [[0.0f64; 4]; 4];
    if header.sform_code > 0 {
        affine[0] = [header.srow_x[0] as f64, header.srow_x[1] as f64,
                     header.srow_x[2] as f64, header.srow_x[3] as f64];
        affine[1] = [header.srow_y[0] as f64, header.srow_y[1] as f64,
                     header.srow_y[2] as f64, header.srow_y[3] as f64];
        affine[2] = [header.srow_z[0] as f64, header.srow_z[1] as f64,
                     header.srow_z[2] as f64, header.srow_z[3] as f64];
    } else {
        affine[0][0] = header.pixdim[1] as f64;
        affine[1][1] = header.pixdim[2] as f64;
        affine[2][2] = header.pixdim[3] as f64;
    }
    affine[3][3] = 1.0;
    affine
}
