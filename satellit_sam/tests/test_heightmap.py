"""Tests for heightmap generation from LiDAR point cloud data."""

import numpy as np
import pytest

from satellit_sam.core import HeightMap, LiDARData


class TestLiDARData:
    """Tests for LiDARData class."""

    def test_init(self):
        """Test LiDARData initialization."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        z = np.array([0, 1, 2])
        lidar = LiDARData(x, y, z)

        assert lidar.num_points == 3
        assert lidar.x_range == (0.0, 2.0)
        assert lidar.y_range == (0.0, 2.0)
        assert lidar.z_range == (0.0, 2.0)

    def test_from_las_file_not_found(self):
        """Test that from_las raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            LiDARData.from_las("nonexistent.las")


class TestHeightMap:
    """Tests for HeightMap class."""

    def test_init(self):
        """Test HeightMap initialization."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        hm = HeightMap(data, resolution=0.5, origin=(0.0, 0.0))

        assert hm.shape == (2, 2)
        assert hm.z_range == (1.0, 4.0)
        assert hm.resolution == 0.5
        assert hm.origin == (0.0, 0.0)

    def test_to_grayscale_default(self):
        """Test conversion to grayscale with default parameters."""
        data = np.array([[0, 5], [10, 15]], dtype=np.float32)
        hm = HeightMap(data, resolution=1.0, origin=(0.0, 0.0))

        grayscale = hm.to_grayscale()

        assert grayscale.dtype == np.uint8
        assert grayscale.min() == 0
        assert grayscale.max() == 255
        assert grayscale[0, 0] == 0  # Minimum value
        assert grayscale[1, 1] == 255  # Maximum value

    def test_to_grayscale_custom_range(self):
        """Test conversion to grayscale with custom z range."""
        data = np.array([[5, 10], [15, 20]], dtype=np.float32)
        hm = HeightMap(data, resolution=1.0, origin=(0.0, 0.0))

        # Set custom range 0-30 (data will be scaled within this range)
        grayscale = hm.to_grayscale(z_min=0, z_max=30)

        assert grayscale.dtype == np.uint8
        # 5/30 * 255 ≈ 42
        assert 40 <= grayscale[0, 0] <= 45
        # 20/30 * 255 ≈ 170
        assert 168 <= grayscale[1, 1] <= 172

    def test_to_grayscale_constant_values(self):
        """Test conversion when all heights are the same."""
        data = np.full((3, 3), 10.0, dtype=np.float32)
        hm = HeightMap(data, resolution=1.0, origin=(0.0, 0.0))

        grayscale = hm.to_grayscale()

        # All values should be 0 when min == max
        assert np.all(grayscale == 0)

    def test_to_image(self):
        """Test conversion to Image object."""
        data = np.array([[0, 5], [10, 15]], dtype=np.float32)
        hm = HeightMap(data, resolution=1.0, origin=(0.0, 0.0))

        img = hm.to_image()

        assert img.channels == 1
        assert img.size == (2, 2)  # width, height
        # Check that image is flipped (north is up)
        # Bottom-right (15) becomes top-right after flip, which is 255 in grayscale
        assert img.data[0, 1] == 255  # Top-right should be max (was bottom-right)

    def test_from_lidar_max_method(self):
        """Test height map creation with max method."""
        # Create simple point cloud in a 2x2 grid
        # Points need to be spread across 2+ meters to create 2x2 grid with 1m resolution
        x = np.array([0.2, 0.2, 1.2, 1.2])
        y = np.array([0.2, 1.2, 0.2, 1.2])
        z = np.array([1.0, 2.0, 3.0, 4.0])
        lidar = LiDARData(x, y, z)

        hm = HeightMap.from_lidar(lidar, resolution=0.5, method="max")

        # Range is 1.0 meters (from 0.2 to 1.2), which creates ceil(1.0/0.5) = 2 cells
        assert hm.shape == (2, 2)
        # Each cell should contain max z value from that region
        assert hm.data[0, 0] == 1.0  # Point at (0.2, 0.2)
        assert hm.data[0, 1] == 3.0  # Point at (1.2, 0.2)
        assert hm.data[1, 0] == 2.0  # Point at (0.2, 1.2)
        assert hm.data[1, 1] == 4.0  # Point at (1.2, 1.2)

    def test_from_lidar_mean_method(self):
        """Test height map creation with mean method."""
        # Create points where multiple points fall in same cell
        x = np.array([0.2, 0.4, 0.6])
        y = np.array([0.2, 0.4, 0.6])
        z = np.array([1.0, 2.0, 3.0])
        lidar = LiDARData(x, y, z)

        hm = HeightMap.from_lidar(lidar, resolution=1.0, method="mean")

        # All points should be in same cell, mean = 2.0
        assert hm.data[0, 0] == pytest.approx(2.0)

    def test_from_lidar_median_method(self):
        """Test height map creation with median method."""
        # Create points where multiple points fall in same cell
        x = np.array([0.2, 0.4, 0.6])
        y = np.array([0.2, 0.4, 0.6])
        z = np.array([1.0, 2.0, 3.0])
        lidar = LiDARData(x, y, z)

        hm = HeightMap.from_lidar(lidar, resolution=1.0, method="median")

        # All points should be in same cell, median = 2.0
        assert hm.data[0, 0] == pytest.approx(2.0)

    def test_from_lidar_invalid_method(self):
        """Test that invalid method raises ValueError."""
        x = np.array([0.2])
        y = np.array([0.2])
        z = np.array([1.0])
        lidar = LiDARData(x, y, z)

        with pytest.raises(ValueError, match="Unknown method"):
            HeightMap.from_lidar(lidar, resolution=1.0, method="invalid")

    def test_from_lidar_with_resolution(self):
        """Test that different resolutions produce different grid sizes."""
        x = np.array([0, 10])
        y = np.array([0, 10])
        z = np.array([1, 2])
        lidar = LiDARData(x, y, z)

        hm_fine = HeightMap.from_lidar(lidar, resolution=1.0)
        hm_coarse = HeightMap.from_lidar(lidar, resolution=5.0)

        # Finer resolution should produce larger grid
        assert hm_fine.shape[0] > hm_coarse.shape[0]
        assert hm_fine.shape[1] > hm_coarse.shape[1]

    def test_origin_preservation(self):
        """Test that origin coordinates are preserved."""
        x = np.array([100, 101])
        y = np.array([200, 201])
        z = np.array([1, 2])
        lidar = LiDARData(x, y, z)

        hm = HeightMap.from_lidar(lidar, resolution=1.0)

        assert hm.origin == (100.0, 200.0)

    def test_from_lidar_with_width(self):
        """Test height map creation with fixed width."""
        x = np.array([0, 10])
        y = np.array([0, 20])
        z = np.array([1, 2])
        lidar = LiDARData(x, y, z)

        hm = HeightMap.from_lidar(lidar, width=100)

        assert hm.shape[1] == 100  # width
        # Height should be calculated to maintain aspect ratio
        assert hm.shape[0] > 0

    def test_from_lidar_with_height(self):
        """Test height map creation with fixed height."""
        x = np.array([0, 10])
        y = np.array([0, 20])
        z = np.array([1, 2])
        lidar = LiDARData(x, y, z)

        hm = HeightMap.from_lidar(lidar, height=100)

        assert hm.shape[0] == 100  # height
        # Width should be calculated to maintain aspect ratio
        assert hm.shape[1] > 0

    def test_from_lidar_with_width_and_height(self):
        """Test height map creation with fixed width and height."""
        x = np.array([0, 10])
        y = np.array([0, 20])
        z = np.array([1, 2])
        lidar = LiDARData(x, y, z)

        hm = HeightMap.from_lidar(lidar, width=200, height=100)

        assert hm.shape == (100, 200)  # (height, width)

    def test_resolution_and_dimensions_conflict(self):
        """Test that specifying both resolution and dimensions raises error."""
        x = np.array([0, 10])
        y = np.array([0, 20])
        z = np.array([1, 2])
        lidar = LiDARData(x, y, z)

        with pytest.raises(ValueError, match="Cannot specify both resolution"):
            HeightMap.from_lidar(lidar, resolution=1.0, width=100)

        with pytest.raises(ValueError, match="Cannot specify both resolution"):
            HeightMap.from_lidar(lidar, resolution=1.0, height=100)

        with pytest.raises(ValueError, match="Cannot specify both resolution"):
            HeightMap.from_lidar(lidar, resolution=1.0, width=100, height=100)

    def test_default_resolution_when_none_specified(self):
        """Test that default 0.5m resolution is used when nothing specified."""
        x = np.array([0, 10])
        y = np.array([0, 10])
        z = np.array([1, 2])
        lidar = LiDARData(x, y, z)

        hm = HeightMap.from_lidar(lidar)

        # With 10m range and 0.5m resolution, expect ceil(10/0.5) = 20 cells
        assert hm.shape == (20, 20)
        assert hm.resolution == 0.5
