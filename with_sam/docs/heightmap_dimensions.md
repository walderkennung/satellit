# Configurable Height Map Dimensions

## Overview

The heightmap module now supports configurable width and height parameters, allowing you to generate height maps with specific pixel dimensions instead of only specifying resolution in meters per pixel.

## Feature Addition

### What Changed

Previously, you could only specify `resolution` (meters per pixel):

```python
# Old way (still supported)
heightmap = create_heightmap_from_las("data.las", resolution=0.5)
```

Now you can also specify exact output dimensions:

```python
# New way - fixed dimensions
heightmap = create_heightmap_from_las("data.las", width=1024, height=768)

# Or specify just width (height auto-calculated)
heightmap = create_heightmap_from_las("data.las", width=2048)

# Or specify just height (width auto-calculated)
heightmap = create_heightmap_from_las("data.las", height=1024)
```

### Why This Matters

**Use Cases:**
- **Standardized outputs**: Generate consistent dimensions across multiple datasets
- **Display requirements**: Match specific screen or tile sizes (e.g., 1024×1024)
- **Memory constraints**: Control output size without calculating resolution
- **Batch processing**: Ensure all outputs have same dimensions for training ML models
- **Web/mobile**: Generate thumbnails or preview images at specific sizes

## API Changes

### `HeightMap.from_lidar()`

**New Parameters:**
```python
def from_lidar(
    lidar: LiDARData,
    resolution: float | None = None,
    method: Literal["max", "mean", "median"] = "max",
    width: int | None = None,      # NEW
    height: int | None = None,     # NEW
) -> HeightMap:
```

**Parameter Rules:**
- Cannot use `resolution` together with `width` or `height`
- Can use `width` alone, `height` alone, or both together
- If neither `resolution` nor `width`/`height` specified, defaults to `resolution=0.5`
- When using one dimension, the other is calculated to maintain aspect ratio

### `create_heightmap_from_las()`

Same parameter additions as `from_lidar()`.

### Command Line Interface

**New Options:**
- `-W, --width`: Output image width in pixels
- `-H, --height`: Output image height in pixels

**Examples:**
```bash
# Fixed dimensions
pixi run python scripts/generate_heightmap.py data.las -W 1024 -H 768

# Width only (height auto-calculated)
pixi run python scripts/generate_heightmap.py data.las -W 2048

# Height only (width auto-calculated)
pixi run python scripts/generate_heightmap.py data.las -H 1024

# Error: cannot mix resolution with dimensions
pixi run python scripts/generate_heightmap.py data.las -r 0.5 -W 1024  # ❌ ERROR
```

## Usage Examples

### Example 1: Standard Sizes

Generate height maps in common sizes:

```python
from satellit_sam.image_processing import LiDARData, HeightMap

lidar = LiDARData.from_las("forest.las")

# HD resolution
hd = HeightMap.from_lidar(lidar, width=1920, height=1080)
hd.save("forest_hd.png")

# Square for ML training
square = HeightMap.from_lidar(lidar, width=512, height=512)
square.save("forest_512.png")

# 4K resolution
uhd = HeightMap.from_lidar(lidar, width=3840, height=2160)
uhd.save("forest_4k.png")
```

### Example 2: Maintain Aspect Ratio

Let the system calculate one dimension:

```python
from satellit_sam.image_processing import create_heightmap_from_las

# Specify width, height calculated automatically
wide = create_heightmap_from_las("data.las", width=2048)
print(f"Dimensions: {wide.shape}")  # e.g., (719, 2048)

# Specify height, width calculated automatically
tall = create_heightmap_from_las("data.las", height=1024)
print(f"Dimensions: {tall.shape}")  # e.g., (1024, 2917)
```

### Example 3: Batch Processing

Ensure consistent output for multiple files:

```python
from pathlib import Path
from satellit_sam.image_processing import create_heightmap_from_las

# Process multiple LiDAR files with same output size
las_files = Path("data").glob("*.las")
target_size = (1024, 1024)

for las_file in las_files:
    heightmap = create_heightmap_from_las(
        las_file, 
        width=target_size[0], 
        height=target_size[1],
        method="max"
    )
    output = f"output/{las_file.stem}_heightmap.png"
    heightmap.save(output)
    print(f"Generated {output}: {heightmap.shape}")
```

### Example 4: Comparing Methods at Fixed Size

Compare aggregation methods with identical dimensions:

```python
from satellit_sam.image_processing import LiDARData, HeightMap

lidar = LiDARData.from_las("data.las")

# All outputs will be exactly 1024x1024
for method in ["max", "mean", "median"]:
    heightmap = HeightMap.from_lidar(
        lidar, 
        width=1024, 
        height=1024, 
        method=method
    )
    heightmap.save(f"comparison_{method}_1024x1024.png")
```

## How It Works

### Resolution Calculation

When you specify dimensions, the resolution is calculated automatically:

```python
# Given:
# - X range: 1000 meters
# - Y range: 500 meters
# - Desired width: 2000 pixels
# - Desired height: 1000 pixels

# Resolution is calculated as:
resolution = max(x_range / width, y_range / height)
resolution = max(1000 / 2000, 500 / 1000)
resolution = max(0.5, 0.5) = 0.5 meters per pixel
```

### Aspect Ratio Handling

When specifying one dimension:

```python
# Specify width only:
# 1. Calculate resolution from width: resolution = x_range / width
# 2. Calculate height: height = ceil(y_range / resolution)

# Specify height only:
# 1. Calculate resolution from height: resolution = y_range / height
# 2. Calculate width: width = ceil(x_range / resolution)
```

### Both Dimensions Specified

When both are specified, the resolution is chosen to ensure the data fits:

```python
# Resolution is the larger of:
# - x_range / width
# - y_range / height
# This ensures all data fits within the specified dimensions
```

## Validation & Error Handling

### Conflict Detection

```python
# ❌ ERROR: Cannot use both resolution and dimensions
heightmap = create_heightmap_from_las(
    "data.las", 
    resolution=0.5,  # Conflict!
    width=1024
)
# Raises: ValueError: Cannot specify both resolution and width/height
```

### Valid Combinations

```python
# ✅ OK: Resolution only
create_heightmap_from_las("data.las", resolution=0.5)

# ✅ OK: Width only
create_heightmap_from_las("data.las", width=1024)

# ✅ OK: Height only
create_heightmap_from_las("data.las", height=768)

# ✅ OK: Width and height
create_heightmap_from_las("data.las", width=1024, height=768)

# ✅ OK: Neither (defaults to resolution=0.5)
create_heightmap_from_las("data.las")
```

## Performance Considerations

### Output Size vs Processing Time

Processing time is proportional to output grid size:

```python
# Small (512×512 = 262K cells): ~2-5 seconds
heightmap = create_heightmap_from_las("data.las", width=512, height=512)

# Medium (1024×1024 = 1M cells): ~5-10 seconds
heightmap = create_heightmap_from_las("data.las", width=1024, height=1024)

# Large (2048×2048 = 4M cells): ~15-30 seconds
heightmap = create_heightmap_from_las("data.las", width=2048, height=2048)

# Very Large (4096×4096 = 16M cells): ~60-120 seconds
heightmap = create_heightmap_from_las("data.las", width=4096, height=4096)
```

### Memory Usage

Memory scales with output size:

```python
# Approximate memory per method:
# - max method: width × height × 8 bytes
# - mean method: width × height × 12 bytes (includes count map)
# - median method: varies (stores lists per cell)

# Examples:
# 512×512:   ~2 MB (max), ~3 MB (mean)
# 1024×1024: ~8 MB (max), ~12 MB (mean)
# 2048×2048: ~32 MB (max), ~48 MB (mean)
# 4096×4096: ~128 MB (max), ~192 MB (mean)
```

## Testing

All functionality is covered by unit tests:

```bash
pixi run pytest tests/test_heightmap.py -v
```

**New Tests Added:**
- `test_from_lidar_with_width`: Width parameter only
- `test_from_lidar_with_height`: Height parameter only
- `test_from_lidar_with_width_and_height`: Both parameters
- `test_resolution_and_dimensions_conflict`: Error handling
- `test_default_resolution_when_none_specified`: Default behavior

**Test Results:** 18/18 passing ✓

## Migration Guide

### Existing Code

All existing code continues to work without changes:

```python
# This still works exactly as before
heightmap = create_heightmap_from_las("data.las", resolution=0.5)
heightmap = HeightMap.from_lidar(lidar, resolution=1.0, method="mean")
```

### New Code

Take advantage of new dimension parameters:

```python
# Generate standardized outputs
heightmap = create_heightmap_from_las("data.las", width=1024, height=1024)

# Maintain aspect ratio
heightmap = create_heightmap_from_las("data.las", width=2048)
```

## Summary

### Added Functionality

✅ `width` parameter for fixed width output
✅ `height` parameter for fixed height output  
✅ Automatic aspect ratio calculation when one dimension specified
✅ Resolution auto-calculation from dimensions
✅ CLI options: `-W/--width` and `-H/--height`
✅ Comprehensive tests (18 total, all passing)
✅ Updated documentation and examples

### Backward Compatibility

✅ All existing code works without modification
✅ Default behavior unchanged (0.5m resolution)
✅ No breaking changes to API

### Benefits

- Standardized output sizes across datasets
- Easier integration with fixed-size systems
- More intuitive for users thinking in pixels
- Better control over memory usage
- Simplified batch processing workflows
