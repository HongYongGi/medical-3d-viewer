from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from pathlib import Path


class MPRSlicer:
    """Multi-Planar Reconstruction slicer for NIfTI volumes."""

    def __init__(self, nifti_path: str | Path):
        img = nib.load(str(nifti_path))
        # Load volume into memory once; callers cache this instance via @st.cache_resource.
        self._img = img
        self.volume = img.get_fdata(dtype=np.float32)
        self.affine = img.affine
        self.inv_affine = np.linalg.inv(img.affine)
        self.shape = self.volume.shape
        self.voxel_spacing = np.array(img.header.get_zooms()[:3])

    @property
    def num_axial(self) -> int:
        return self.shape[2]

    @property
    def num_sagittal(self) -> int:
        return self.shape[0]

    @property
    def num_coronal(self) -> int:
        return self.shape[1]

    def get_axial(self, index: int) -> np.ndarray:
        index = np.clip(index, 0, self.shape[2] - 1)
        return self.volume[:, :, index].T

    def get_sagittal(self, index: int) -> np.ndarray:
        index = np.clip(index, 0, self.shape[0] - 1)
        return self.volume[index, :, :].T

    def get_coronal(self, index: int) -> np.ndarray:
        index = np.clip(index, 0, self.shape[1] - 1)
        return self.volume[:, index, :].T

    def world_to_voxel(self, world_coords: np.ndarray) -> np.ndarray:
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)
        ones = np.ones((world_coords.shape[0], 1))
        world_h = np.hstack([world_coords, ones])
        voxel_h = (self.inv_affine @ world_h.T).T
        return voxel_h[:, :3]

    def voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        if voxel_coords.ndim == 1:
            voxel_coords = voxel_coords.reshape(1, -1)
        ones = np.ones((voxel_coords.shape[0], 1))
        voxel_h = np.hstack([voxel_coords, ones])
        world_h = (self.affine @ voxel_h.T).T
        return world_h[:, :3]

    def get_oblique(
        self,
        center_world: np.ndarray,
        normal: np.ndarray,
        up: np.ndarray | None = None,
        size: int = 256,
        spacing: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct an oblique slice.

        Args:
            center_world: Center point in world (RAS) coordinates [3]
            normal: Normal vector of the slice plane [3]
            up: Up direction vector [3]. If None, auto-computed.
            size: Output image size in pixels
            spacing: Pixel spacing in mm. If None, uses minimum voxel spacing.

        Returns:
            (slice_image, transform_3x3)
        """
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            raise ValueError("Normal 벡터의 크기가 0에 가깝습니다.")
        normal = normal / norm_len

        coords = self._compute_oblique_coords(center_world, normal, up, size, spacing)
        values = map_coordinates(self.volume, coords, order=1, mode='constant', cval=-1024)
        slice_img = values.reshape(size, size)

        right, up_vec, sp = self._oblique_basis(normal, up, spacing)
        transform = np.column_stack([right * sp, up_vec * sp, center_world])
        return slice_img, transform

    def get_oblique_seg(
        self,
        seg_volume: np.ndarray,
        center_world: np.ndarray,
        normal: np.ndarray,
        up: np.ndarray | None = None,
        size: int = 256,
        spacing: float | None = None,
    ) -> np.ndarray:
        """Same as get_oblique but for segmentation (nearest-neighbor)."""
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            raise ValueError("Normal 벡터의 크기가 0에 가깝습니다.")
        normal = normal / norm_len

        coords = self._compute_oblique_coords(center_world, normal, up, size, spacing)
        values = map_coordinates(seg_volume, coords, order=0, mode='constant', cval=0)
        return values.reshape(size, size).astype(np.int32)

    def _oblique_basis(self, normal, up, spacing):
        """Compute right/up basis vectors and effective spacing."""
        if up is None:
            if abs(np.dot(normal, [0, 0, 1])) < 0.9:
                up = np.array([0.0, 0.0, 1.0])
            else:
                up = np.array([0.0, 1.0, 0.0])
        right = np.cross(normal, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, normal)
        up /= np.linalg.norm(up)
        if spacing is None:
            spacing = float(np.min(self.voxel_spacing))
        return right, up, spacing

    def _compute_oblique_coords(self, center_world, normal, up, size, spacing):
        """Compute voxel coordinates for an oblique slice (shared by CT and seg)."""
        right, up_vec, sp = self._oblique_basis(normal, up, spacing)

        half = size / 2.0
        u = np.linspace(-half * sp, half * sp, size)
        v = np.linspace(-half * sp, half * sp, size)
        uu, vv = np.meshgrid(u, v, indexing='xy')

        world_points = (
            center_world[np.newaxis, np.newaxis, :]
            + uu[:, :, np.newaxis] * right[np.newaxis, np.newaxis, :]
            + vv[:, :, np.newaxis] * up_vec[np.newaxis, np.newaxis, :]
        )

        flat_world = world_points.reshape(-1, 3)
        flat_voxel = self.world_to_voxel(flat_world)
        return flat_voxel.T
