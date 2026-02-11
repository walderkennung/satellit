# Satellit SAM

Satellite imagery segmentation using [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) from Meta AI.

## Features

- Process large satellite/orthophoto images by tiling
- Automatic mask generation using SAM
- Tile reconstruction with smooth blending for overlapping regions
- Weak-label generation from forest inventory data (CSV or SHP)
- GeoTIFF/UTM-aware inventory alignment for SHP inputs (EPSG:32633)
- Inventory QA visualizations (crowns + tile grid overlays)
- Support for CPU (macOS/Linux) and CUDA (Linux) acceleration

## Prerequisites

- [Pixi](https://pixi.sh) package manager

## Setup

### 1. Install dependencies

For CPU (macOS or Linux without CUDA):

```bash
pixi install
```

For CUDA (Linux with NVIDIA GPU):

```bash
pixi install -e cuda
```

### 2. Download SAM model checkpoints

```bash
pixi run sam-download
```

This downloads all three SAM model variants to `models/sam/`:

- `sam_vit_h_4b8939.pth` (ViT-H, largest, best quality)
- `sam_vit_l_0b3195.pth` (ViT-L, medium)
- `sam_vit_b_01ec64.pth` (ViT-B, smallest, fastest)

### 3. Prepare your data

Place your input image (e.g., orthophoto) in the `data/` directory at the parent level:

```
../data/orthophoto_wgs84_utm33n_agg200mm.tif
```

## Usage

### Test SAM setup

Verify that SAM loads correctly on your system:

```bash
pixi run sam-test
```

### Run mask prediction

Process your image and generate segmentation masks:

```bash
pixi run predict-masks
```

Run with one or more bounding-box prompts (image-space `x1,y1,x2,y2`):

```bash
pixi run predict-masks -- --bbox 500,600,900,1000 --bbox 1200,1400,1700,1900
```

This will:

1. Load the image and split it into tiles
2. Run SAM automatic mask generation on each tile
3. Save individual tile results to `output/tiles/`
4. Reconstruct the full image with blended overlaps
5. Save the final result to `output/reconstructed.png`

### Configuration

Edit `src/satellit_sam/main.py` to adjust processing parameters:

```python
process_tiles(
    image,
    sam,
    output_dir="output/tiles",
    initial_offset=[0, 0],  # Start offset in tiles
    max_tiles=32,           # Maximum number of tiles to process
    tile_size=1024,         # Tile size in pixels
    overlap=256,            # Overlap between tiles for blending
)
```

### Generate weak labels from inventory

Create per-tile weak labels (`x`, `y`, `crown_radius`) from inventory data.

Recommended input:
- `--inventory-shp` with UTM coordinates (for Traunstein: EPSG:32633)

Also supported:
- `--inventory-csv` with local coordinates (supports optional auto-alignment)

Run with SHP input:

```bash
pixi run python scripts/generate_inventory_weak_labels.py \
  --image-tif ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --inventory-shp ../data/Traunstein/inventory/processed/shifted_new_tree_positions_UTM33N.shp \
  --output-dir output/inventory_from_shp \
  --tile-size 1024 \
  --overlap 128 \
  --only-non-empty-tiles \
  --deduplicate-tree-id \
  --export-visualizations
```

Filter to trees with DBH >= 50 cm:

```bash
pixi run python scripts/generate_inventory_weak_labels.py \
  --image-tif ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --inventory-shp ../data/Traunstein/inventory/processed/shifted_new_tree_positions_UTM33N.shp \
  --output-dir output/inventory_from_shp_dbh50 \
  --min-dbh-cm 50 \
  --tile-size 1024 \
  --overlap 128 \
  --only-non-empty-tiles \
  --deduplicate-tree-id \
  --export-visualizations
```

Run with CSV input (local coordinates) and automatic alignment:

```bash
pixi run python scripts/generate_inventory_weak_labels.py \
  --image-tif ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --inventory-csv ../data/Traunstein/inventory/original/PlotDataReport10-04-2018_1323085911.csv \
  --output-dir output/inventory_from_csv_auto \
  --auto-align \
  --tile-size 1024 \
  --overlap 128 \
  --only-non-empty-tiles \
  --deduplicate-tree-id \
  --export-visualizations
```

Generated outputs include:
- `labels_tiles.yaml` and `labels_tiles.json` (tile-wise weak labels)
- `trees_projected.csv` (projected/global coordinates and crown radius)
- `summary.json` (run parameters and counts)
- `visualizations/labels_crowns.png` and `visualizations/labels_crowns_tiles.png` (QA)

## Project Structure

```
with_sam/
├── models/
│   └── sam/              # SAM model checkpoints (downloaded)
├── output/
│   ├── tiles/            # Individual tile predictions
│   └── reconstructed.png # Final reconstructed image
├── scripts/
│   ├── cuda_activation.sh
│   ├── download_sam.py   # Model download script
│   ├── generate_inventory_weak_labels.py  # Inventory -> weak labels
│   └── test_sam.py       # SAM test script
├── src/
│   └── satellit_sam/
│       ├── main.py       # Main processing script
│       └── pytorch.py    # PyTorch initialization utilities
├── pixi.toml             # Pixi configuration
├── pyproject.toml        # Python package configuration
└── README.md
```

## Testing

The project includes a comprehensive test suite for the `image_processing` module with **49 tests** covering all functionality.

### Run all tests

```bash
pixi run test
```

Or explicitly:

```bash
pixi run python -m pytest tests/ -v
```

### Run specific test categories

```bash
# Unit tests only
pixi run python -m pytest tests/ -m unit -v

# Integration tests only
pixi run python -m pytest tests/ -m integration -v
```

### Run specific test files

```bash
# Image class tests (20 tests)
pixi run python -m pytest tests/test_image.py -v

# Tiling functionality tests (13 tests)
pixi run python -m pytest tests/test_tiling.py -v

# End-to-end integration tests (16 tests)
pixi run python -m pytest tests/test_integration.py -v
```

### Test Coverage

The test suite covers:

- ✅ Image loading, saving, and cropping operations
- ✅ Tiling with various configurations (overlap, sizes)
- ✅ Metadata serialization and deserialization
- ✅ End-to-end tile → reconstruct pipelines
- ✅ Reconstruction accuracy validation (SSIM >95%)
- ✅ Real satellite imagery processing
- ✅ Edge cases and error handling

## Environments

| Environment | Platform     | Accelerator             |
| ----------- | ------------ | ----------------------- |
| `default`   | macOS, Linux | CPU                     |
| `cuda`      | Linux        | NVIDIA GPU (CUDA 12.0+) |

To run commands in a specific environment:

```bash
pixi run -e cuda predict-masks
```

## License

See the main project repository for license information.
