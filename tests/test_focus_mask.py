"""Tests for real-space focus mask functionality in train_vae."""

import numpy as np
import pytest
import torch
import os

from cryodrgn import fft
from cryodrgn.mrcfile import write_mrc, parse_mrc
from cryodrgn.commands.train_vae import (
    load_focus_mask,
    unsymmetrize_ht,
    hartley_to_real,
)


class TestFocusMaskLoading:
    """Test focus mask loading and validation."""

    @pytest.fixture
    def circular_mask(self, tmp_path):
        """Create a simple circular focus mask."""
        ny = 64
        y, x = np.ogrid[-ny // 2 : ny // 2, -ny // 2 : ny // 2]
        mask = (x**2 + y**2 <= (ny // 4) ** 2).astype(np.float32)
        mask_path = tmp_path / "circular_mask.mrc"
        write_mrc(str(mask_path), mask[np.newaxis], is_vol=False)
        return str(mask_path), mask

    @pytest.fixture
    def soft_mask(self, tmp_path):
        """Create a soft (non-binary) focus mask with cosine edges."""
        ny = 64
        y, x = np.ogrid[-ny // 2 : ny // 2, -ny // 2 : ny // 2]
        dist = np.sqrt(x**2 + y**2)
        inner_rad = ny // 6
        outer_rad = ny // 4
        mask = np.ones((ny, ny), dtype=np.float32)
        transition = (dist > inner_rad) & (dist <= outer_rad)
        mask[transition] = 0.5 * (
            1 + np.cos(np.pi * (dist[transition] - inner_rad) / (outer_rad - inner_rad))
        )
        mask[dist > outer_rad] = 0.0
        mask_path = tmp_path / "soft_mask.mrc"
        write_mrc(str(mask_path), mask[np.newaxis], is_vol=False)
        return str(mask_path), mask

    def test_load_circular_mask(self, circular_mask):
        """Test loading a binary circular mask."""
        mask_path, expected_mask = circular_mask
        ny = expected_mask.shape[0]
        device = torch.device("cpu")

        mask = load_focus_mask(mask_path, ny, device)

        assert mask.shape == (ny, ny)
        assert mask.min() >= 0
        assert mask.max() <= 1
        # Check it's properly normalized (was binary, should still be 0 and 1)
        assert torch.allclose(mask, torch.tensor(expected_mask))

    def test_load_soft_mask(self, soft_mask):
        """Test loading a soft (non-binary) mask."""
        mask_path, expected_mask = soft_mask
        ny = expected_mask.shape[0]
        device = torch.device("cpu")

        mask = load_focus_mask(mask_path, ny, device)

        assert mask.shape == (ny, ny)
        assert mask.min() >= 0
        assert mask.max() <= 1
        # Soft mask should have intermediate values
        assert (mask > 0).sum() > (mask == 1).sum()

    def test_load_mask_with_threshold(self, soft_mask):
        """Test loading a mask with binarization threshold."""
        mask_path, _ = soft_mask
        ny = 64
        device = torch.device("cpu")

        mask = load_focus_mask(mask_path, ny, device, threshold=0.5)

        # After thresholding, should be binary
        unique_vals = torch.unique(mask)
        assert len(unique_vals) <= 2
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_load_3d_mask_takes_central_slice(self, tmp_path):
        """Test that 3D masks use the central slice."""
        nz, ny, nx = 32, 64, 64
        vol = np.zeros((nz, ny, nx), dtype=np.float32)
        # Only the central slice has non-zero values
        vol[nz // 2, :, :] = 1.0
        mask_path = tmp_path / "3d_mask.mrc"
        write_mrc(str(mask_path), vol, is_vol=True)

        device = torch.device("cpu")
        mask = load_focus_mask(str(mask_path), ny, device)

        assert mask.shape == (ny, nx)
        # Should have loaded the central slice which is all ones
        assert mask.sum() == ny * nx

    def test_load_mask_size_mismatch_raises(self, tmp_path):
        """Test that size mismatch raises an error."""
        ny = 64
        wrong_ny = 128
        mask = np.ones((wrong_ny, wrong_ny), dtype=np.float32)
        mask_path = tmp_path / "wrong_size_mask.mrc"
        write_mrc(str(mask_path), mask[np.newaxis], is_vol=False)

        device = torch.device("cpu")
        with pytest.raises(AssertionError, match="does not match image size"):
            load_focus_mask(str(mask_path), ny, device)


class TestHartleyTransforms:
    """Test Hartley transform helper functions."""

    def test_hartley_roundtrip(self):
        """Test that Hartley transform is invertible."""
        ny = 64
        # Create random real-space image
        img_real = torch.randn(ny, ny)

        # Forward Hartley
        img_ht = fft.ht2_center(img_real)

        # Inverse Hartley
        img_recovered = fft.iht2_center(img_ht)

        # Should recover original (up to numerical precision)
        assert torch.allclose(img_real, img_recovered, atol=1e-5)

    def test_hartley_roundtrip_batched(self):
        """Test batched Hartley transform roundtrip."""
        B, ny = 4, 64
        img_real = torch.randn(B, ny, ny)

        img_ht = fft.ht2_center(img_real)
        img_recovered = fft.iht2_center(img_ht)

        assert torch.allclose(img_real, img_recovered, atol=1e-5)

    def test_unsymmetrize_ht(self):
        """Test that unsymmetrize correctly removes the extra row/column."""
        ny = 64
        D = ny + 1  # Symmetrized size

        # Create symmetrized Hartley (D x D)
        ht_sym = torch.randn(D, D)

        # Unsymmetrize
        ht_unsym = unsymmetrize_ht(ht_sym)

        assert ht_unsym.shape == (ny, ny)
        # Should be the [:-1, :-1] slice
        assert torch.allclose(ht_unsym, ht_sym[:-1, :-1])

    def test_unsymmetrize_ht_batched(self):
        """Test batched unsymmetrization."""
        B, ny = 4, 64
        D = ny + 1

        ht_sym = torch.randn(B, D, D)
        ht_unsym = unsymmetrize_ht(ht_sym)

        assert ht_unsym.shape == (B, ny, ny)
        assert torch.allclose(ht_unsym, ht_sym[:, :-1, :-1])

    def test_symmetrize_unsymmetrize_roundtrip(self):
        """Test that symmetrize -> unsymmetrize recovers original."""
        ny = 64
        img_ht = torch.randn(ny, ny)

        # Symmetrize
        img_sym = fft.symmetrize_ht(img_ht)
        assert img_sym.shape == (ny + 1, ny + 1)

        # Unsymmetrize
        img_unsym = unsymmetrize_ht(img_sym)
        assert img_unsym.shape == (ny, ny)

        # Should recover original
        assert torch.allclose(img_ht, img_unsym)


class TestHartleyToReal:
    """Test the hartley_to_real conversion function."""

    def test_hartley_to_real_basic(self):
        """Test basic hartley_to_real conversion."""
        B, ny = 4, 64
        D = ny + 1
        norm = (0.0, 1.0)  # No normalization effect

        # Create normalized symmetrized Hartley
        ht_sym_norm = torch.randn(B, D, D)

        # Convert to real
        real = hartley_to_real(ht_sym_norm, norm)

        assert real.shape == (B, ny, ny)
        assert real.dtype == torch.float32

    def test_hartley_to_real_with_normalization(self):
        """Test that normalization is correctly undone."""
        B, ny = 2, 32
        D = ny + 1
        mean, std = 5.0, 2.0
        norm = (mean, std)

        # Create a simple image, apply normalization, symmetrize
        img_real_orig = torch.randn(B, ny, ny)
        img_ht = fft.ht2_center(img_real_orig)
        img_ht_norm = (img_ht - mean) / std
        img_ht_sym_norm = fft.symmetrize_ht(img_ht_norm)

        # Convert back
        img_real_recovered = hartley_to_real(img_ht_sym_norm, norm)

        # Should approximately recover original
        assert torch.allclose(img_real_orig, img_real_recovered, atol=1e-4)


class TestRealSpaceLossComputation:
    """Test the real-space loss computation logic."""

    def test_masked_loss_ignores_outside_region(self):
        """Test that loss only considers the masked region."""
        B, ny = 4, 64

        # Create two images that are identical inside the mask
        # but different outside
        y1_real = torch.randn(B, ny, ny)
        y2_real = y1_real.clone()

        # Create mask (central region only)
        mask = torch.zeros(ny, ny)
        mask[ny // 4 : 3 * ny // 4, ny // 4 : 3 * ny // 4] = 1.0

        # Modify y2 only outside the mask
        outside_mask = 1 - mask
        y2_real = y2_real * mask + torch.randn(B, ny, ny) * outside_mask

        # Apply mask and compute loss
        y1_masked = y1_real * mask
        y2_masked = y2_real * mask

        # Loss should be (nearly) zero since masked regions are identical
        loss = torch.nn.functional.mse_loss(y1_masked, y2_masked)
        assert loss < 1e-10

    def test_masked_loss_detects_differences_inside(self):
        """Test that loss detects differences inside the masked region."""
        B, ny = 4, 64

        y1_real = torch.randn(B, ny, ny)
        y2_real = torch.randn(B, ny, ny)  # Different image

        # Create mask
        mask = torch.zeros(ny, ny)
        mask[ny // 4 : 3 * ny // 4, ny // 4 : 3 * ny // 4] = 1.0

        # Apply mask
        y1_masked = y1_real * mask
        y2_masked = y2_real * mask

        # Loss should be non-zero
        loss = torch.nn.functional.mse_loss(y1_masked, y2_masked)
        assert loss > 0.1  # Should have significant difference

    def test_soft_mask_weighting(self):
        """Test that soft mask properly weights contributions."""
        B, ny = 2, 32

        # Create images with uniform difference
        y1_real = torch.zeros(B, ny, ny)
        y2_real = torch.ones(B, ny, ny)

        # Create soft mask with known values
        mask = torch.zeros(ny, ny)
        mask[:ny // 2, :] = 1.0  # Top half = 1
        mask[ny // 2 :, :] = 0.5  # Bottom half = 0.5

        # Compute masked difference
        diff = (y1_real - y2_real) ** 2  # = 1 everywhere
        weighted_diff = diff * mask

        # Expected: top half contributes 1.0, bottom half contributes 0.5
        expected_sum = (ny // 2 * ny * 1.0 + ny // 2 * ny * 0.5) * B
        actual_sum = weighted_diff.sum()

        assert torch.isclose(actual_sum, torch.tensor(expected_sum))
