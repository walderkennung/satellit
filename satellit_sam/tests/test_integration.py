"""Integration tests for end-to-end image processing workflows."""

import os
from pathlib import Path

import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim

from satellit_sam.core import Image, TilesDir, tile_image


@pytest.mark.integration
class TestEndToEndTileReconstruct:
    """End-to-end tests for tiling and reconstruction."""

    def test_tile_and_reconstruct_no_overlap(self, medium_test_image, temp_dir):
        """Test complete pipeline: tile → reconstruct with no overlap."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        # Step 1: Tile the image
        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=0,
            output_path=str(output_path),
        )
        tiles_dir.save_to_dir()

        # Step 2: Reconstruct the image
        reconstructed = tiles_dir.reconstruct_image()

        # Step 3: Compare with original
        assert reconstructed.size == medium_test_image.size
        assert reconstructed.channels == 4  # Reconstruction adds alpha channel

    def test_tile_and_reconstruct_with_overlap(self, medium_test_image, temp_dir):
        """Test complete pipeline: tile → reconstruct with overlap."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        # Tile with overlap
        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=64,
            output_path=str(output_path),
        )
        tiles_dir.save_to_dir()

        # Reconstruct
        reconstructed = tiles_dir.reconstruct_image()

        # Verify dimensions
        assert reconstructed.size == medium_test_image.size
        assert reconstructed.data.shape[0] == medium_test_image.data.shape[0]
        assert reconstructed.data.shape[1] == medium_test_image.data.shape[1]

    def test_reconstruction_accuracy_no_overlap(self, medium_test_image, temp_dir):
        """Test reconstruction accuracy without overlap (should be perfect)."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        # Tile and reconstruct
        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=0,
            output_path=str(output_path),
        )
        reconstructed = tiles_dir.reconstruct_image()

        # Compare RGB channels (ignore alpha from reconstruction)
        original_rgb = medium_test_image.data.astype(np.float32)
        reconstructed_rgb = reconstructed.data[:, :, :3].astype(np.float32)

        # Calculate SSIM (structural similarity)
        similarity = ssim(
            original_rgb,
            reconstructed_rgb,
            channel_axis=2,
            data_range=255.0,
        )

        # With no overlap, reconstruction should be near-perfect
        assert similarity > 0.95

    def test_reconstruction_accuracy_with_overlap(self, medium_test_image, temp_dir):
        """Test reconstruction accuracy with overlap."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        # Tile with overlap
        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=64,
            output_path=str(output_path),
        )
        reconstructed = tiles_dir.reconstruct_image()

        # Compare RGB channels
        original_rgb = medium_test_image.data.astype(np.float32)
        reconstructed_rgb = reconstructed.data[:, :, :3].astype(np.float32)

        # Calculate SSIM
        similarity = ssim(
            original_rgb,
            reconstructed_rgb,
            channel_axis=2,
            data_range=255.0,
        )

        # With overlap and blending, should still be very similar
        assert similarity > 0.90

    def test_forest_image_pipeline(self, forest_image, temp_dir):
        """Test complete pipeline with real forest image."""
        output_path = temp_dir / "forest_tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        # Tile the forest image
        tiles_dir = tile_image(
            forest_image,
            tile_size=512,
            overlap=128,
            output_path=str(output_path),
        )
        tiles_dir.save_to_dir()

        # Verify tiles were created
        tile_files = list(Path(tiles_rgb_path).glob("tile_*.png"))
        assert len(tile_files) > 0

        # Reconstruct
        reconstructed = tiles_dir.reconstruct_image()

        # Verify dimensions match
        assert reconstructed.size == forest_image.size

    def test_save_and_reload_reconstruction(self, medium_test_image, temp_dir):
        """Test saving, loading metadata, and reconstructing."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        # Step 1: Tile and save
        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=64,
            output_path=str(output_path),
        )
        tiles_dir.save_to_dir()

        # Step 2: Load from directory (simulating a fresh session)
        loaded_tiles_dir = TilesDir.load_from_dir(str(output_path))

        # Step 3: Reconstruct from loaded metadata
        reconstructed = loaded_tiles_dir.reconstruct_image()

        # Verify reconstruction
        assert reconstructed.size == medium_test_image.size

    def test_multiple_tile_sizes(self, medium_test_image, temp_dir):
        """Test pipeline with different tile sizes."""
        tile_sizes = [128, 256, 384]

        for tile_size in tile_sizes:
            output_path = temp_dir / f"tiles_{tile_size}"
            tiles_rgb_path = output_path / "tiles_rgb"
            os.makedirs(tiles_rgb_path, exist_ok=True)

            tiles_dir = tile_image(
                medium_test_image,
                tile_size=tile_size,
                overlap=tile_size // 4,
                output_path=str(output_path),
            )

            reconstructed = tiles_dir.reconstruct_image()
            assert reconstructed.size == medium_test_image.size


@pytest.mark.integration
class TestTileIteration:
    """Tests for iterating over tiles."""

    def test_iterate_over_tiles(self, small_test_image, temp_dir):
        """Test iterating through all tiles in a TilesDir."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            small_test_image,
            tile_size=50,
            overlap=10,
            output_path=str(output_path),
        )

        # Count tiles through iteration
        tile_count = 0
        for tile in tiles_dir:
            tile_count += 1
            assert tile.image is not None
            assert isinstance(tile.image, Image)

        assert tile_count > 0

    def test_tile_iterator_image_properties(self, medium_test_image, temp_dir):
        """Test that iterated tiles have correct properties."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=64,
            output_path=str(output_path),
        )

        for tile in tiles_dir:
            # Each tile should have valid dimensions
            assert tile.image.size[0] <= 256
            assert tile.image.size[1] <= 256
            assert tile.image.channels == 3


@pytest.mark.integration
class TestTilePositions:
    """Tests for tile position tracking."""

    def test_get_tile_positions(self, small_test_image, temp_dir):
        """Test retrieving tile positions."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            small_test_image,
            tile_size=50,
            overlap=10,
            output_path=str(output_path),
        )

        positions = tiles_dir.get_tile_positions()

        assert isinstance(positions, set)
        assert len(positions) > 0

        # Each position should be a tuple of two integers
        for pos in positions:
            assert isinstance(pos, tuple)
            assert len(pos) == 2
            assert isinstance(pos[0], int)
            assert isinstance(pos[1], int)

    def test_tile_positions_match_files(self, medium_test_image, temp_dir):
        """Test that tile positions match actual tile files."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=64,
            output_path=str(output_path),
        )

        positions = tiles_dir.get_tile_positions()
        tile_files = list(Path(tiles_rgb_path).glob("tile_*.png"))

        # Number of positions should match number of files
        assert len(positions) == len(tile_files)


@pytest.mark.integration
class TestEdgeCasesIntegration:
    """Integration tests for edge cases."""

    def test_small_overlap(self, medium_test_image, temp_dir):
        """Test with very small overlap."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=1,
            output_path=str(output_path),
        )

        reconstructed = tiles_dir.reconstruct_image()
        assert reconstructed.size == medium_test_image.size

    def test_large_overlap(self, medium_test_image, temp_dir):
        """Test with large overlap (half tile size)."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=128,
            output_path=str(output_path),
        )

        reconstructed = tiles_dir.reconstruct_image()
        assert reconstructed.size == medium_test_image.size

    def test_non_square_image(self, temp_dir):
        """Test with non-square image dimensions."""
        # Create a rectangular image
        data = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
        rect_image = Image(size=(600, 300), channels=3, data=data)

        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            rect_image,
            tile_size=200,
            overlap=50,
            output_path=str(output_path),
        )

        reconstructed = tiles_dir.reconstruct_image()
        assert reconstructed.size == rect_image.size


@pytest.mark.integration
class TestPixelAccuracy:
    """Tests for pixel-level accuracy of reconstruction."""

    def test_pixel_perfect_no_overlap(self, temp_dir):
        """Test pixel-perfect reconstruction with no overlap."""
        # Create image with distinct regions for verification
        data = np.zeros((256, 256, 3), dtype=np.uint8)
        data[:128, :128] = [255, 0, 0]  # Red top-left
        data[:128, 128:] = [0, 255, 0]  # Green top-right
        data[128:, :128] = [0, 0, 255]  # Blue bottom-left
        data[128:, 128:] = [255, 255, 0]  # Yellow bottom-right

        test_image = Image(size=(256, 256), channels=3, data=data)

        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            test_image,
            tile_size=128,
            overlap=0,
            output_path=str(output_path),
        )

        reconstructed = tiles_dir.reconstruct_image()

        # With no overlap, RGB channels should be identical
        np.testing.assert_array_equal(
            test_image.data,
            reconstructed.data[:, :, :3],
        )

    def test_mean_squared_error(self, medium_test_image, temp_dir):
        """Test MSE between original and reconstructed image."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=32,
            output_path=str(output_path),
        )

        reconstructed = tiles_dir.reconstruct_image()

        # Calculate MSE
        original = medium_test_image.data.astype(np.float32)
        reconstructed_rgb = reconstructed.data[:, :, :3].astype(np.float32)

        mse = np.mean((original - reconstructed_rgb) ** 2)

        # MSE should be low (blending with overlap introduces small errors)
        assert mse < 50.0  # Allow small error due to blending
