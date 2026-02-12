# Satellit SAM

Satellite imagery segmentation and weak-label generation using the Segment Anything workflow.

## Prerequisites

- [Pixi](https://pixi.sh)

## Setup

Install dependencies from `satellit_sam/`:

```bash
pixi install
```

For CUDA on Linux:

```bash
pixi install -e cuda
```

Download SAM checkpoints:

```bash
pixi run sam-download
```

Quick validation:

```bash
pixi run sam-test
```

## CLI

The current CLI is exposed through the `satellit` Pixi task:

```bash
pixi run satellit -- --help
```

Command tree:

- `image-processing`
- `label weak`

Global options:

- `--install-completion`
- `--show-completion`
- `--help`

### `image-processing`

Run image segmentation:

```bash
pixi run satellit -- image-processing \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --output-path output/predict \
  --tile-size 2048 \
  --overlap 64 \
  --text-prompt trees
```

With one or more image-space bounding boxes (`x1,y1,x2,y2`):

```bash
pixi run satellit -- image-processing \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --bbox 500,600,900,1000 \
  --bbox 1200,1400,1700,1900
```

Notes:

- If neither `--text-prompt` nor `--bbox` is provided, the prompt defaults to `trees`.
- If only `--bbox` is provided, segmentation runs from box prompts without a text prompt.

### `label weak`

Generate weak labels from inventory data:

```bash
pixi run satellit -- label weak \
  --image-tif ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --inventory-shp ../data/Traunstein/inventory/processed/shifted_new_tree_positions_UTM33N.shp \
  --output-dir output/inventory_from_shp \
  --tile-size 1024 \
  --overlap 128 \
  --only-non-empty-tiles \
  --deduplicate-tree-id \
  --export-visualizations
```

Rules:

- Provide exactly one of `--inventory-csv` or `--inventory-shp`.
- `--image-tif` and `--output-dir` are required.

Generated outputs include:

- `labels_tiles.yaml` and `labels_tiles.json`
- `trees_projected.csv`
- `summary.json`
- optional visualization PNGs under `visualizations/`

## Full Command Reference

Complete CLI docs (all commands/options/arguments) are in:

- `../docs/content/cli/application.md`

## Testing

Run tests from `satellit_sam/`:

```bash
pixi run test
```

## Environments

| Environment | Platform     | Accelerator             |
| ----------- | ------------ | ----------------------- |
| `default`   | macOS, Linux | CPU                     |
| `cuda`      | Linux        | NVIDIA GPU (CUDA 12.0+) |

Run in CUDA environment:

```bash
pixi run -e cuda satellit -- image-processing --help
```
