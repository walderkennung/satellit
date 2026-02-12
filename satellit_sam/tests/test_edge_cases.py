"""Edge case tests to achieve 100% coverage."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from satellit_sam.image_processing import Image, TilesDir, tile_image


@pytest.mark.integration
class TestMissingCoverage:
    """Tests for edge cases to achieve 100% coverage."""

    def test_reconstruct_with_missing_tile(self, medium_test_image, temp_dir, capsys):
        """Test reconstruction when a tile file is missing."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=0,
            output_path=str(tiles_rgb_path),
        )
        tiles_dir.output_path = str(output_path)

        # Create a fake tile as a directory (not a file)
        # This will cause cv2.imread to fail and Image.load to raise FileNotFoundError
        fake_tile_path = tiles_rgb_path / "tile_x9999_y9999.png"
        os.makedirs(fake_tile_path, exist_ok=True)

        # Should handle missing file gracefully
        reconstructed = tiles_dir.reconstruct_image()

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning: Could not read" in captured.out

        # Should still reconstruct (with missing region)
        assert reconstructed is not None
        assert reconstructed.size == medium_test_image.size

    def test_rgba_tiles(self, temp_dir):
        """Test reconstruction with RGBA tiles (4 channels)."""
        # Create an RGB test image first (since cv2.imwrite strips alpha)
        data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        rgb_image = Image(size=(256, 256), channels=3, data=data)

        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        # Tile the RGB image
        tiles_dir = tile_image(
            rgb_image,
            tile_size=128,
            overlap=0,
            output_path=str(tiles_rgb_path),
        )
        tiles_dir.output_path = str(output_path)

        # Manually save one tile as RGBA to test the else branch (line 196)
        tile_files = list(Path(tiles_rgb_path).glob("tile_*.png"))
        if tile_files:
            # Load the tile and add alpha channel
            tile_img = Image.load(str(tile_files[0]))
            rgba_data = np.dstack(
                [tile_img.data, np.full(tile_img.data.shape[:2], 255, dtype=np.uint8)]
            )
            # Save with cv2.imwrite using IMWRITE_PNG_COMPRESSION and alpha channel
            import cv2

            cv2.imwrite(
                str(tile_files[0]),
                cv2.cvtColor(rgba_data, cv2.COLOR_RGBA2BGRA),
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )

        # Reconstruct - should handle the RGBA tile
        reconstructed = tiles_dir.reconstruct_image()

        # Should handle mixed input (some RGBA tiles)
        assert reconstructed.channels == 4
        assert reconstructed.size == rgb_image.size

    def test_tile_resize_during_reconstruction(self, medium_test_image, temp_dir):
        """Test reconstruction when tiles need resizing."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            medium_test_image,
            tile_size=256,
            overlap=0,
            output_path=str(tiles_rgb_path),
        )
        tiles_dir.output_path = str(output_path)

        # Manually resize one tile to trigger the resize path
        tile_files = list(Path(tiles_rgb_path).glob("tile_*.png"))
        if tile_files:
            tile_path = tile_files[0]
            tile_img = Image.load(str(tile_path))
            # Resize to a different dimension
            import cv2

            resized_data = cv2.resize(tile_img.data, (200, 200))
            resized_tile = Image(
                size=(200, 200), channels=tile_img.channels, data=resized_data
            )
            resized_tile.save(str(tile_path))

        # Reconstruct should handle resizing
        reconstructed = tiles_dir.reconstruct_image()
        assert reconstructed.size == medium_test_image.size

    def test_get_tile_positions_empty_directory(self, temp_dir):
        """Test get_tile_positions when directory doesn't exist."""
        output_path = temp_dir / "nonexistent"

        tiles_dir = TilesDir(
            output_path=str(output_path),
            tile_size=(256, 256),
            overlap=(0, 0),
            original_shape=(512, 512, 3),
        )

        positions = tiles_dir.get_tile_positions()

        # Should return empty set
        assert isinstance(positions, set)
        assert len(positions) == 0

    def test_iterator_stop_iteration(self, small_test_image, temp_dir):
        """Test iterator properly raises StopIteration."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = tile_image(
            small_test_image,
            tile_size=50,
            overlap=0,
            output_path=str(tiles_rgb_path),
        )
        tiles_dir.output_path = str(output_path)

        # Use the iterator in a way that explicitly calls __next__ until StopIteration
        iterator = iter(tiles_dir)

        # Consume all tiles manually
        tile_count = 0
        try:
            while True:
                next(iterator)
                tile_count += 1
        except StopIteration:
            pass

        # Should have consumed some tiles
        assert tile_count > 0

        # Call __next__ explicitly one more time to ensure line 277 is covered
        try:
            iterator.__next__()
            assert False, "Should have raised StopIteration"
        except StopIteration:
            pass  # Expected

    def test_grayscale_image_load(self, temp_dir):
        """Test loading a grayscale image."""
        # Create a grayscale image
        gray_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        path = temp_dir / "gray.png"

        import cv2

        cv2.imwrite(str(path), gray_data)

        # Load with Image.load
        loaded = Image.load(str(path))

        assert loaded.channels == 1
        assert loaded.size == (100, 100)
        np.testing.assert_array_equal(loaded.data, gray_data)

    def test_grayscale_image_save(self, temp_dir):
        """Test saving a grayscale image."""
        # Create a grayscale image
        gray_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray_image = Image(size=(100, 100), channels=1, data=gray_data)

        path = temp_dir / "gray_save.png"
        gray_image.save(str(path))

        # Load it back
        loaded = Image.load(str(path))
        assert loaded.channels == 1
        assert loaded.size == (100, 100)
        np.testing.assert_array_equal(loaded.data, gray_data)

    def test_rgba_save_load_roundtrip(self, temp_dir):
        """Test RGBA image save and load round trip."""
        # Create an RGBA image
        rgba_data = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        rgba_image = Image(size=(100, 100), channels=4, data=rgba_data)

        path = temp_dir / "rgba.png"
        rgba_image.save(str(path))

        # Load it back
        loaded = Image.load(str(path))
        assert loaded.channels == 4
        assert loaded.size == (100, 100)
        np.testing.assert_array_equal(loaded.data, rgba_data)

    def test_iterator_empty(self, temp_dir):
        """Test iterator with no tiles."""
        output_path = temp_dir / "tiles"
        tiles_rgb_path = output_path / "tiles_rgb"
        os.makedirs(tiles_rgb_path, exist_ok=True)

        tiles_dir = TilesDir(
            output_path=str(output_path),
            tile_size=(256, 256),
            overlap=(0, 0),
            original_shape=(512, 512, 3),
        )

        # Iterate over empty directory
        tile_count = 0
        for tile in tiles_dir:
            tile_count += 1

        # Should complete without error
        assert tile_count == 0

        # Explicitly call next on empty iterator to trigger line 277
        iterator = iter(tiles_dir)
        try:
            next(iterator)
            assert False, "Should raise StopIteration"
        except StopIteration:
            pass  # This should execute line 277
