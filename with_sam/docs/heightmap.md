# LiDAR Height Map Generation

This module provides functionality to generate grayscale height maps from LiDAR point cloud data (.las files). Height maps are useful for visualizing terrain, canopy heights, and other elevation-based features.

## Features

- **Read LiDAR data** from .las files
- **Multiple aggregation methods**: max, mean, median
- **Configurable resolution**: set grid cell size in meters
- **Grayscale normalization**: automatic or custom height range mapping
- **Integration with Image class**: seamlessly convert to Image objects for further processing
- **Memory efficient**: processes large point clouds

## Installation

The heightmap module requires the `laspy` package:

```bash
pixi run pip install laspy
```

## Quick Start

### Basic Usage

```python
from satellit_sam.image_processing import create_heightmap_from_las

# Create with resolution (meters per pixel)
heightmap = create_heightmap_from_las("data.las", resolution=0.5, method="max")
heightmap.save("heightmap.png")

# Or create with fixed dimensions
heightmap = create_heightmap_from_las("data.las", width=1024, height=768)
heightmap.save("heightmap.png")
```

### Using the Command Line

```bash
# Basic usage (uses 0.5m resolution by default)
pixi run python scripts/generate_heightmap.py data/input.las

# With custom resolution
pixi run python scripts/generate_heightmap.py data/input.las --resolution 1.0

# With fixed dimensions
pixi run python scripts/generate_heightmap.py data/input.las -W 1024 -H 768

# Combine with other options
pixi run python scripts/generate_heightmap.py data/input.las \
    -W 2048 -H 1024 \
    --method mean \
    --output output/heightmap.png \
    --verbose
```

## API Reference

### Classes

#### `LiDARData`

Container for LiDAR point cloud data.

**Methods:**

- `from_las(las_path)`: Load LiDAR data from .las file
- `num_points`: Get total number of points
- `x_range`, `y_range`, `z_range`: Get coordinate ranges

**Example:**

```python
from satellit_sam.image_processing import LiDARData

lidar = LiDARData.from_las("data.las")
print(f"Total points: {lidar.num_points:,}")
print(f"Height range: {lidar.z_range}")
```

#### `HeightMap`

2D height map representation with methods for conversion and saving.

**Properties:**

- `data`: 2D numpy array of height values
- `resolution`: Grid cell size in meters
- `origin`: (x_min, y_min) coordinates in original system
- `shape`: Height map dimensions (height, width)
- `z_range`: Min and max height values

**Methods:**

- `from_lidar(lidar, resolution, method, width, height)`: Create from LiDARData
    - Use either `resolution` OR `width`/`height`, not both
- `to_grayscale(z_min, z_max)`: Convert to grayscale array (0-255)
- `to_image(z_min, z_max)`: Convert to Image object
- `save(path, z_min, z_max)`: Save as grayscale PNG

**Example:**

```python
from satellit_sam.image_processing import LiDARData, HeightMap

# Load data
lidar = LiDARData.from_las("data.las")

# Create height map with resolution
heightmap = HeightMap.from_lidar(lidar, resolution=0.5, method="max")

# Or create with fixed dimensions
heightmap = HeightMap.from_lidar(lidar, width=1024, height=768, method="max")

# Save with custom normalization
heightmap.save("output.png", z_min=0, z_max=50)
```

### Functions

#### `create_heightmap_from_las(las_path, resolution, method, width, height)`

Convenience function that loads LiDAR data and creates a height map in one step.

**Parameters:**

- `las_path` (str | Path): Path to .las file
- `resolution` (float | None): Grid cell size in meters. Cannot be used with width/height.
- `method` (str): Aggregation method: 'max', 'mean', or 'median' (default: 'max')
- `width` (int | None): Target width in pixels. Cannot be used with resolution.
- `height` (int | None): Target height in pixels. Cannot be used with resolution.

**Returns:**

- `HeightMap`: Height map object

**Examples:**

```python
from satellit_sam.image_processing import create_heightmap_from_las

# Using resolution
heightmap = create_heightmap_from_las("data.las", resolution=1.0, method="max")
heightmap.save("heightmap.png")

# Using fixed dimensions
heightmap = create_heightmap_from_las("data.las", width=1024, height=768)
heightmap.save("heightmap.png")

# Using only width (height calculated automatically)
heightmap = create_heightmap_from_las("data.las", width=2048)
heightmap.save("heightmap.png")
```

## Aggregation Methods

### Max (Default)

Uses the maximum height value in each grid cell. Best for canopy height models where you want to capture the tallest vegetation.

```python
heightmap = HeightMap.from_lidar(lidar, method="max")
```

### Mean

Uses the average height value in each grid cell. Good for general terrain elevation.

```python
heightmap = HeightMap.from_lidar(lidar, method="mean")
```

### Median

Uses the median height value in each grid cell. More robust to outliers than mean.

```python
heightmap = HeightMap.from_lidar(lidar, method="median")
```

## Advanced Usage

### Custom Normalization

Control how heights are mapped to grayscale values:

```python
heightmap = create_heightmap_from_las("data.las")

# Normalize to specific height range (0-50 meters)
heightmap.save("output.png", z_min=0, z_max=50)

# Heights above 50m will be clipped to 255 (white)
# Heights below 0m will be clipped to 0 (black)
```

### Integration with Image Class

Convert height maps to Image objects for further processing:

```python
from satellit_sam.image_processing import create_heightmap_from_las

# Create height map
heightmap = create_heightmap_from_las("data.las")

# Convert to Image object
image = heightmap.to_image()

# Now you can use all Image methods
cropped = image.crop(0, 0, 500, 500)
cropped.save("cropped_heightmap.png")
```

### Multiple Resolutions and Dimensions

Create height maps at different resolutions or dimensions for multi-scale analysis:

```python
from satellit_sam.image_processing import LiDARData, HeightMap

lidar = LiDARData.from_las("data.las")

# Fine resolution for detail
fine = HeightMap.from_lidar(lidar, resolution=0.25)
fine.save("heightmap_fine.png")

# Coarse resolution for overview
coarse = HeightMap.from_lidar(lidar, resolution=2.0)
coarse.save("heightmap_coarse.png")

# Fixed dimensions for consistent output size
standard = HeightMap.from_lidar(lidar, width=1024, height=1024)
standard.save("heightmap_1024.png")

# Width only (height calculated to maintain aspect ratio)
wide = HeightMap.from_lidar(lidar, width=2048)
wide.save("heightmap_wide.png")
```

### Comparing Methods

Compare different aggregation methods:

```python
from satellit_sam.image_processing import LiDARData, HeightMap

lidar = LiDARData.from_las("data.las")

for method in ["max", "mean", "median"]:
    heightmap = HeightMap.from_lidar(lidar, resolution=1.0, method=method)
    heightmap.save(f"heightmap_{method}.png")
    print(f"{method}: {heightmap.z_range}")
```

## Command Line Interface

The `generate_heightmap.py` script provides a complete CLI for height map generation.

### Options

- `input`: Input .las file path (required)
- `-o, --output`: Output image path (default: `<input>_heightmap.png`)
- `-r, --resolution`: Grid resolution in meters (default: 0.5). Cannot be used with -W/-H.
- `-W, --width`: Output image width in pixels. Cannot be used with -r.
- `-H, --height`: Output image height in pixels. Cannot be used with -r.
- `-m, --method`: Aggregation method: max, mean, median (default: max)
- `-v, --verbose`: Print detailed information

### Examples

```bash
# Generate with defaults (0.5m resolution, max method)
pixi run python scripts/generate_heightmap.py data/forest.las

# High resolution height map
pixi run python scripts/generate_heightmap.py data/forest.las -r 0.25

# Fixed dimensions (1024x768 pixels)
pixi run python scripts/generate_heightmap.py data/forest.las -W 1024 -H 768

# Fixed width only (height calculated automatically)
pixi run python scripts/generate_heightmap.py data/forest.las -W 2048

# Use mean height instead of max
pixi run python scripts/generate_heightmap.py data/forest.las -m mean

# Specify output location
pixi run python scripts/generate_heightmap.py data/forest.las -o results/canopy.png

# Verbose output for debugging
pixi run python scripts/generate_heightmap.py data/forest.las -v

# Combine options with resolution
pixi run python scripts/generate_heightmap.py data/forest.las \
    -r 0.25 -m median -o results/detailed.png -v

# Combine options with dimensions
pixi run python scripts/generate_heightmap.py data/forest.las \
    -W 2048 -H 1024 -m mean -o results/standard_size.png -v
```

## Performance Considerations

### Resolution vs. Processing Time

- **Higher resolution** (e.g., 0.25m): More detail, larger images, slower processing
- **Lower resolution** (e.g., 2.0m): Less detail, smaller images, faster processing

For a point cloud with ~10 million points:

- 0.25m resolution: ~30-60 seconds
- 0.5m resolution: ~10-20 seconds
- 1.0m resolution: ~5-10 seconds
- 2.0m resolution: ~2-5 seconds

### Fixed Dimensions

Using fixed width/height parameters:

- Processing time depends on output size, not resolution
- Larger dimensions (e.g., 2048×2048): slower, more detail
- Smaller dimensions (e.g., 512×512): faster, less detail
- Good for standardizing output sizes across different datasets

### Memory Usage

Memory usage depends on grid size:

- Grid size = (x_range / resolution) × (y_range / resolution)
- Or: Grid size = width × height (when using fixed dimensions)
- Each cell requires ~4-8 bytes

For large point clouds, consider:

1. Using coarser resolution or smaller dimensions
2. Processing subsets of the data
3. Using the 'max' method (most memory efficient)

## Output Format

Height maps are saved as:

- **Format**: PNG (grayscale, 8-bit)
- **Orientation**: North is up (data is vertically flipped)
- **Color mapping**:
    - Black (0): Lowest height
    - White (255): Highest height
    - Gray values: Heights in between

## Testing

Run the test suite:

```bash
pixi run pytest tests/test_heightmap.py -v
```

Run example scripts:

```bash
cd with_sam
pixi run python examples/heightmap_example.py
```

## Troubleshooting

### ModuleNotFoundError: No module named 'laspy'

Install laspy:

```bash
pixi run pip install laspy
```

### FileNotFoundError

Ensure the .las file path is correct and the file exists:

```python
from pathlib import Path
las_file = Path("data.las")
print(f"File exists: {las_file.exists()}")
```

### Empty or black height map

Check if:

1. The LiDAR file contains data: `lidar.num_points > 0`
2. The height range is valid: `lidar.z_range`
3. The resolution is appropriate for your data

### Memory errors

For very large point clouds:

1. Increase resolution (e.g., from 0.25m to 1.0m)
2. Use 'max' method instead of 'median'
3. Process smaller regions separately

## See Also

- [Image Processing Module](image_processing.md)
- [Tiling Module](tiling.md)
- [Examples](../examples/heightmap_example.py)
