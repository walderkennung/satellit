"""
Heightmap generation from LiDAR point cloud data.

This module provides functionality to read LiDAR .las files and generate
grayscale height maps that can be used for visualization and further processing.
"""

from pathlib import Path
from typing import Literal

import laspy
import numpy as np

from .image import Image


class LiDARData:
    """Container for LiDAR point cloud data."""

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """
        Initialize LiDAR data.

        Args:
            x: X coordinates of points
            y: Y coordinates of points
            z: Z coordinates (height) of points
        """
        self.x = x
        self.y = y
        self.z = z

    @property
    def num_points(self) -> int:
        """Get total number of points."""
        return len(self.x)

    @property
    def x_range(self) -> tuple[float, float]:
        """Get min and max X coordinates."""
        return float(self.x.min()), float(self.x.max())

    @property
    def y_range(self) -> tuple[float, float]:
        """Get min and max Y coordinates."""
        return float(self.y.min()), float(self.y.max())

    @property
    def z_range(self) -> tuple[float, float]:
        """Get min and max Z coordinates."""
        return float(self.z.min()), float(self.z.max())

    @staticmethod
    def from_las(las_path: str | Path) -> "LiDARData":
        """
        Read LiDAR data from .las file.

        Args:
            las_path: Path to .las file

        Returns:
            LiDARData object containing point cloud coordinates

        Raises:
            FileNotFoundError: If the .las file doesn't exist
        """
        las_path = Path(las_path)
        if not las_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {las_path}")

        las = laspy.read(str(las_path))

        # Extract coordinates
        x = np.array(las.x, dtype=np.float64)
        y = np.array(las.y, dtype=np.float64)
        z = np.array(las.z, dtype=np.float64)

        return LiDARData(x, y, z)


class HeightMap:
    """2D height map representation."""

    def __init__(
        self, data: np.ndarray, resolution: float, origin: tuple[float, float]
    ):
        """
        Initialize height map.

        Args:
            data: 2D array of height values
            resolution: Grid cell size in meters
            origin: (x_min, y_min) origin coordinates in the original coordinate system
        """
        self.data = data
        self.resolution = resolution
        self.origin = origin

    @property
    def shape(self) -> tuple[int, int]:
        """Get height map dimensions (height, width)."""
        return self.data.shape

    @property
    def z_range(self) -> tuple[float, float]:
        """Get min and max height values."""
        return float(self.data.min()), float(self.data.max())

    def to_grayscale(
        self, z_min: float | None = None, z_max: float | None = None
    ) -> np.ndarray:
        """
        Convert height map to grayscale image (0-255).

        Args:
            z_min: Minimum height value for normalization. If None, uses data minimum.
            z_max: Maximum height value for normalization. If None, uses data maximum.

        Returns:
            2D numpy array with uint8 values (0-255)
        """
        if z_min is None:
            z_min = float(self.data.min())
        if z_max is None:
            z_max = float(self.data.max())

        # Normalize to 0-255
        if z_max > z_min:
            normalized = (self.data - z_min) / (z_max - z_min) * 255
        else:
            normalized = np.zeros_like(self.data)

        return normalized.astype(np.uint8)

    def to_rgb(
        self, z_min: float | None = None, z_max: float | None = None
    ) -> np.ndarray:
        """
        Convert height map to RGB image (grayscale).

        Args:
            z_min: Minimum height value for normalization. If None, uses data minimum.
            z_max: Maximum height value for normalization. If None, uses data maximum.

        Returns:
            3D numpy array with uint8 values (height, width, 3)
        """
        grayscale = self.to_grayscale(z_min, z_max)
        return np.stack((grayscale,) * 3, axis=-1)

    def to_image(self, z_min: float | None = None, z_max: float | None = None) -> Image:
        """
        Convert height map to Image object.

        Args:
            z_min: Minimum height value for normalization. If None, uses data minimum.
            z_max: Maximum height value for normalization. If None, uses data maximum.

        Returns:
            Image object with grayscale height map (flipped vertically so north is up)
        """
        grayscale = self.to_grayscale(z_min, z_max)
        # Flip vertically so north is up
        flipped = np.flipud(grayscale)
        height, width = flipped.shape
        return Image(size=(width, height), channels=1, data=flipped)

    def save(
        self,
        path: str | Path,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> None:
        """
        Save height map as grayscale image file.

        Args:
            path: Output file path
            z_min: Minimum height value for normalization. If None, uses data minimum.
            z_max: Maximum height value for normalization. If None, uses data maximum.
        """
        self.to_image(z_min, z_max).save(str(path))

    @staticmethod
    def from_lidar(
        lidar: LiDARData,
        resolution: float | None = None,
        method: Literal["max", "mean", "median"] = "max",
        width: int | None = None,
        height: int | None = None,
    ) -> "HeightMap":
        """
        Create height map from LiDAR point cloud data.

        Args:
            lidar: LiDARData object containing point cloud
            resolution: Grid cell size in meters. If None, calculated from width/height.
                Cannot be used together with width/height parameters.
            method: Method to aggregate heights in each cell:
                - 'max': Use maximum height (good for canopy height)
                - 'mean': Use average height
                - 'median': Use median height
            width: Target width in pixels. If provided with height, resolution is calculated.
                Cannot be used together with resolution parameter.
            height: Target height in pixels. If provided with width, resolution is calculated.
                Cannot be used together with resolution parameter.

        Returns:
            HeightMap object

        Raises:
            ValueError: If neither resolution nor width/height are provided, or if both are provided
        """
        x, y, z = lidar.x, lidar.y, lidar.z

        # Calculate grid dimensions
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        # Determine grid dimensions based on input parameters
        if resolution is not None and (width is not None or height is not None):
            raise ValueError(
                "Cannot specify both resolution and width/height. Use either resolution OR width/height."
            )

        if resolution is None and width is None and height is None:
            # Default to 0.5m resolution
            resolution = 0.5

        if resolution is not None:
            # Use resolution to calculate dimensions
            grid_width = int(np.ceil(x_range / resolution))
            grid_height = int(np.ceil(y_range / resolution))
        else:
            # Use width/height to calculate resolution
            if width is not None and height is not None:
                grid_width = width
                grid_height = height
                # Calculate resolution based on dimensions
                resolution = max(x_range / width, y_range / height)
            elif width is not None:
                grid_width = width
                resolution = x_range / width
                grid_height = int(np.ceil(y_range / resolution))
            elif height is not None:
                grid_height = height
                resolution = y_range / height
                grid_width = int(np.ceil(x_range / resolution))
            else:
                raise ValueError("Must specify either resolution or width/height")

        # Initialize grid
        height_data = np.zeros((grid_height, grid_width), dtype=np.float32)
        count_map = np.zeros((grid_height, grid_width), dtype=np.int32)

        # For max method, initialize with very small values
        if method == "max":
            height_data.fill(-np.inf)

        # Map points to grid cells
        x_indices = ((x - x_min) / resolution).astype(np.int32)
        y_indices = ((y - y_min) / resolution).astype(np.int32)

        # Clip indices to grid bounds
        x_indices = np.clip(x_indices, 0, grid_width - 1)
        y_indices = np.clip(y_indices, 0, grid_height - 1)

        # Fill grid based on method
        if method == "max":
            for i in range(len(x)):
                xi, yi = x_indices[i], y_indices[i]
                if z[i] > height_data[yi, xi]:
                    height_data[yi, xi] = z[i]
            # Replace -inf with 0 for empty cells
            height_data[np.isinf(height_data)] = 0

        elif method == "mean":
            for i in range(len(x)):
                xi, yi = x_indices[i], y_indices[i]
                height_data[yi, xi] += z[i]
                count_map[yi, xi] += 1
            # Calculate mean for cells with points
            mask = count_map > 0
            height_data[mask] /= count_map[mask]

        elif method == "median":
            # For median, we need to store lists of values
            grid_dict: dict[tuple[int, int], list[float]] = {}
            for i in range(len(x)):
                xi, yi = x_indices[i], y_indices[i]
                key = (yi, xi)
                if key not in grid_dict:
                    grid_dict[key] = []
                grid_dict[key].append(float(z[i]))

            # Calculate median for each cell
            for (yi, xi), values in grid_dict.items():
                height_data[yi, xi] = np.median(values)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'max', 'mean', or 'median'"
            )

        return HeightMap(
            data=height_data, resolution=resolution, origin=(float(x_min), float(y_min))
        )


def create_heightmap_from_las(
    las_path: str | Path,
    resolution: float | None = None,
    method: Literal["max", "mean", "median"] = "max",
    width: int | None = None,
    height: int | None = None,
) -> HeightMap:
    """
    Create a height map from a LiDAR .las file.

    This is a convenience function that combines loading LiDAR data and
    creating a height map in one step.

    Args:
        las_path: Path to .las file
        resolution: Grid cell size in meters. If None, calculated from width/height or defaults to 0.5.
            Cannot be used together with width/height parameters.
        method: Method to aggregate heights in each cell ('max', 'mean', 'median')
        width: Target width in pixels. If provided with height, resolution is calculated.
            Cannot be used together with resolution parameter.
        height: Target height in pixels. If provided with width, resolution is calculated.
            Cannot be used together with resolution parameter.

    Returns:
        HeightMap object

    Examples:
        >>> # Using resolution
        >>> heightmap = create_heightmap_from_las("data.las", resolution=1.0)
        >>> heightmap.save("heightmap.png")

        >>> # Using fixed dimensions
        >>> heightmap = create_heightmap_from_las("data.las", width=1024, height=1024)
        >>> heightmap.save("heightmap.png")
    """
    lidar = LiDARData.from_las(las_path)
    return HeightMap.from_lidar(
        lidar, resolution=resolution, method=method, width=width, height=height
    )
