"""Tests for the Image class."""

import os

import numpy as np
import pytest

from satellit_sam.image_processing import Image


@pytest.mark.unit
class TestImageCreation:
    """Tests for Image instantiation."""

    def test_create_image_with_valid_parameters(self):
        """Test creating an Image with valid parameters."""
        data = np.zeros((100, 200, 3), dtype=np.uint8)
        image = Image(size=(200, 100), channels=3, data=data)

        assert image.size == (200, 100)
        assert image.channels == 3
        assert image.data.shape == (100, 200, 3)

    def test_image_size_order(self):
        """Test that size is (width, height) while data is (height, width, channels)."""
        height, width = 50, 100
        data = np.zeros((height, width, 3), dtype=np.uint8)
        image = Image(size=(width, height), channels=3, data=data)

        assert image.size == (width, height)
        assert image.data.shape == (height, width, 3)


@pytest.mark.unit
class TestImageLoad:
    """Tests for Image.load() method."""

    def test_load_existing_image(self, forest_image_path):
        """Test loading an existing image file."""
        image = Image.load(str(forest_image_path))

        assert image is not None
        assert image.channels == 3
        assert image.size[0] > 0
        assert image.size[1] > 0
        assert image.data is not None

    def test_load_image_dimensions(self, forest_image_path):
        """Test that loaded image has correct dimensions."""
        image = Image.load(str(forest_image_path))

        height, width, channels = image.data.shape
        assert image.size == (width, height)
        assert image.channels == channels

    def test_load_nonexistent_image(self):
        """Test loading a non-existent image raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found at path"):
            Image.load("/nonexistent/path/to/image.jpg")

    def test_load_invalid_path(self):
        """Test loading from an invalid path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Image.load("")


@pytest.mark.unit
class TestImageSave:
    """Tests for Image.save() method."""

    def test_save_image(self, small_test_image, temp_dir):
        """Test saving an image to a file."""
        output_path = temp_dir / "test_output.png"
        small_test_image.save(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_and_load_roundtrip(self, small_test_image, temp_dir):
        """Test that an image can be saved and loaded back."""
        output_path = temp_dir / "roundtrip.png"
        small_test_image.save(str(output_path))

        loaded_image = Image.load(str(output_path))

        assert loaded_image.size == small_test_image.size
        assert loaded_image.channels == small_test_image.channels
        assert loaded_image.data.shape == small_test_image.data.shape

    def test_save_to_different_formats(self, small_test_image, temp_dir):
        """Test saving images to different formats."""
        formats = ["png", "jpg", "jpeg"]

        for fmt in formats:
            output_path = temp_dir / f"test_output.{fmt}"
            small_test_image.save(str(output_path))
            assert output_path.exists()

    def test_save_creates_parent_directories(self, small_test_image, temp_dir):
        """Test that saving creates parent directories if needed."""
        nested_path = temp_dir / "subdir" / "nested" / "image.png"
        os.makedirs(nested_path.parent, exist_ok=True)
        small_test_image.save(str(nested_path))

        assert nested_path.exists()


@pytest.mark.unit
class TestImageCrop:
    """Tests for Image.crop() method."""

    def test_crop_center_region(self, medium_test_image):
        """Test cropping a region from the center of an image."""
        # Original is 512x512, crop 100x100 from center
        start_x, start_y = 206, 206
        end_x, end_y = 306, 306

        cropped = medium_test_image.crop(start_x, start_y, end_x, end_y)

        assert cropped.size == (100, 100)
        assert cropped.channels == medium_test_image.channels
        assert cropped.data.shape == (100, 100, 3)

    def test_crop_top_left_corner(self, medium_test_image):
        """Test cropping from the top-left corner."""
        cropped = medium_test_image.crop(0, 0, 50, 50)

        assert cropped.size == (50, 50)
        assert cropped.data.shape == (50, 50, 3)

    def test_crop_bottom_right_corner(self, medium_test_image):
        """Test cropping from the bottom-right corner."""
        # 512x512 image, crop last 50x50 pixels
        cropped = medium_test_image.crop(462, 462, 512, 512)

        assert cropped.size == (50, 50)
        assert cropped.data.shape == (50, 50, 3)

    def test_crop_full_image(self, small_test_image):
        """Test cropping the entire image returns same dimensions."""
        cropped = small_test_image.crop(0, 0, 100, 100)

        assert cropped.size == small_test_image.size
        assert cropped.channels == small_test_image.channels

    def test_crop_preserves_data(self):
        """Test that cropping preserves the correct pixel data."""
        # Create image with known pattern
        data = np.arange(100 * 100 * 3, dtype=np.uint8).reshape((100, 100, 3))
        image = Image(size=(100, 100), channels=3, data=data)

        cropped = image.crop(10, 20, 30, 40)

        expected_data = data[20:40, 10:30]
        np.testing.assert_array_equal(cropped.data, expected_data)

    def test_crop_different_sizes(self, medium_test_image):
        """Test cropping with various sizes."""
        crop_configs = [
            (0, 0, 100, 50),  # Wide rectangle
            (0, 0, 50, 100),  # Tall rectangle
            (100, 100, 200, 200),  # Square
            (0, 0, 512, 256),  # Half height
        ]

        for start_x, start_y, end_x, end_y in crop_configs:
            cropped = medium_test_image.crop(start_x, start_y, end_x, end_y)
            expected_width = end_x - start_x
            expected_height = end_y - start_y

            assert cropped.size == (expected_width, expected_height)
            assert cropped.data.shape == (expected_height, expected_width, 3)

    def test_crop_rgba_image(self, rgba_test_image):
        """Test cropping an RGBA image preserves alpha channel."""
        cropped = rgba_test_image.crop(10, 10, 50, 50)

        assert cropped.channels == 4
        assert cropped.data.shape == (40, 40, 4)


@pytest.mark.unit
class TestImageEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_crop_at_boundaries(self, small_test_image):
        """Test cropping at exact image boundaries."""
        width, height = small_test_image.size

        # Crop to exact boundaries
        cropped = small_test_image.crop(0, 0, width, height)
        assert cropped.size == small_test_image.size

    def test_image_with_single_pixel(self):
        """Test operations on a 1x1 image."""
        data = np.array([[[255, 0, 0]]], dtype=np.uint8)
        image = Image(size=(1, 1), channels=3, data=data)

        assert image.size == (1, 1)
        assert image.channels == 3

        cropped = image.crop(0, 0, 1, 1)
        assert cropped.size == (1, 1)

    def test_save_and_load_preserves_dimensions(self, forest_image, temp_dir):
        """Test that save/load preserves dimensions of real image."""
        output_path = temp_dir / "forest_copy.png"
        forest_image.save(str(output_path))

        loaded = Image.load(str(output_path))

        assert loaded.size == forest_image.size
        assert loaded.channels == forest_image.channels
