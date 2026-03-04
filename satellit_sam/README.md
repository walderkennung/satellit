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

- `label weak`
- `label by-bounding-boxes`
- `predict image-masks`

Global options:

- `--install-completion`
- `--show-completion`
- `--help`

### `label by-bounding-boxes`

Run label generation from bounding-box prompts:

```bash
pixi run satellit -- label by-bounding-boxes \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --output-path output/predict \
  --tile-size 2048 \
  --overlap 64
```

With one or more image-space bounding boxes (`x1,y1,x2,y2`):

```bash
pixi run satellit -- label by-bounding-boxes \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --bbox 500,600,900,1000 \
  --bbox 1200,1400,1700,1900
```

Notes:

- `--weak-labels-csv` can load per-tree tile-local bboxes from `label weak` output (`labels_tiles.csv`).

### `predict image-masks`

Run streamed tile mask prediction on one input image (default `--tile-size 640`, `--tile-overlap 64`):

```bash
pixi run satellit -- predict image-masks \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --output-path output/predict \
  --text "tree crowns"
```

With point and bounding-box prompts in image coordinates (default `sam3`):

```bash
pixi run satellit -- predict image-masks \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --point 1200,900 \
  --bbox 1100,800,1500,1200
```

Use SAM2 explicitly:

```bash
pixi run satellit -- predict image-masks \
  --model sam2 \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --point 1200,900
```

Use DINOv3 explicitly (text filtering only):

```bash
pixi run satellit -- predict image-masks \
  --model dinov3 \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --text "tree"
```

Note:

- `--model` accepts `sam3` (default), `sam2`, and `dinov3`.
- `--text` is supported for `sam3` and `dinov3` (not `sam2`).
- `dinov3` currently supports `--text` only (no `--bbox`, `--point`, or `--weak-labels-csv`).
- Prediction uses streamed image tiles by default; tune with `--tile-size` and `--tile-overlap`.
- Cross-tile detections are merged globally with IoU NMS (`--merge-iou-threshold`, default `0.5`).
- With the current `transformers` SAM3 processor API, point prompts are approximated as small box prompts.

Outputs are written under `--output-path`:

- `image_masks_visualization.png` (mask visualization)
- `masks/image_masks.npz` (predicted masks, boxes, scores, image_size)

### `label weak`

Generate weak labels from inventory data:

```bash
pixi run satellit -- label weak \
  --image-tif ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --inventory-shp ../data/Traunstein/inventory/processed/shifted_new_tree_positions_UTM33N.shp \
  --output-dir output/inventory_from_shp \
  --tile-size 1024 \
  --overlap 128 \
  --deduplicate-tree-id \
  --export-visualizations
```

Rules:

- Provide exactly one of `--inventory-csv` or `--inventory-shp`.
- `--image-tif` and `--output-dir` are required.

Generated outputs include:

- `labels_tiles.csv` (flat weak-label rows with `bbox_x1,bbox_y1,bbox_x2,bbox_y2`)
- `labels_tiles.shp` (WGS84 points with bbox attributes)
- optional visualization TIFFs under `visualizations/`

## Full Command Reference

Complete CLI docs (all commands/options/arguments) are in:

- `../docs/content/cli/index.md`

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
pixi run -e cuda satellit -- label by-bounding-boxes --help
```
