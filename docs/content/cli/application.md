---
title: "CLI Application"
description: "Reference for the satellit_sam command-line interface."
---

# CLI Application

The CLI entrypoint is `python -m satellit_sam.main`, exposed by Pixi as:

```bash
pixi run satellit -- [COMMAND] [OPTIONS]
```

No commands currently use positional arguments; all inputs are passed via options.

## Command Tree

```text
satellit
├── image-processing
└── label
    └── weak
```

## Root Command: `satellit`

Usage:

```bash
pixi run satellit -- --help
```

Global options:

| Option | Type | Description |
| --- | --- | --- |
| `--install-completion` | flag | Install shell completion for the current shell. |
| `--show-completion` | flag | Print completion script for the current shell. |
| `--help` | flag | Show help and exit. |

## Command: `image-processing`

Usage:

```bash
pixi run satellit -- image-processing --image <PATH> [OPTIONS]
```

Description: Process satellite imagery using SAM.

Options:

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--image` | file path | yes | - | Input image file (for example, GeoTIFF). |
| `--tile-size` | int | no | `2048` | Tile size in pixels. |
| `--overlap` | int | no | `64` | Overlap between tiles in pixels. |
| `--output-path` | path | no | `output/test_tiles` | Output directory for generated tiles/results. |
| `--text-prompt` | str | no | `None` | Optional text prompt for detection. |
| `--bbox` | str (repeatable) | no | `None` | Bounding box in `x1,y1,x2,y2` format. Repeat flag for multiple boxes. |
| `--help` | flag | no | - | Show help and exit. |

Behavior notes:

- If neither `--text-prompt` nor `--bbox` is provided, the text prompt defaults to `trees`.
- `--bbox` values must be numeric and satisfy `x2 > x1` and `y2 > y1`.

Examples:

```bash
pixi run satellit -- image-processing \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --tile-size 2048 \
  --overlap 64 \
  --output-path output/predict
```

```bash
pixi run satellit -- image-processing \
  --image ../data/Traunstein/orthophoto_wgs84_utm33n_agg200mm.tif \
  --bbox 500,600,900,1000 \
  --bbox 1200,1400,1700,1900
```

## Command Group: `label`

Usage:

```bash
pixi run satellit -- label --help
```

Subcommands:

- `weak`

## Command: `label weak`

Usage:

```bash
pixi run satellit -- label weak [OPTIONS]
```

Description: Generate weak labels from a forest inventory and image tiles.

Validation rules:

- Provide exactly one of `--inventory-csv` or `--inventory-shp`.
- `--image-tif` and `--output-dir` are required.

Options:

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--image-tif` | path | yes | - | Path to orthophoto GeoTIFF. |
| `--inventory-csv` | path | no | `None` | Inventory CSV path (semicolon-delimited). |
| `--inventory-shp` | path | no | `None` | Inventory ESRI Shapefile (`.shp`) path. |
| `--output-dir` | path | yes | - | Output directory for labels and optional tiles. |
| `--tile-size` | int | no | `1024` | Tile size in pixels. |
| `--overlap` | int | no | `128` | Tile overlap in pixels. |
| `--only-non-empty-tiles` | flag | no | `False` | Write only tiles that contain at least one tree. |
| `--x-field` | str | no | `PX` | Inventory x-coordinate field name. |
| `--y-field` | str | no | `PY` | Inventory y-coordinate field name. |
| `--tree-id-field` | str | no | `TreeID` | Inventory tree-id field name. |
| `--species-field` | str | no | `Latin` | Inventory species field name. |
| `--status-field` | str | no | `Status` | Inventory status field name. |
| `--status-filter` | str | no | `alive` | Keep rows with this status (case-insensitive). Empty string disables filter. |
| `--dbh-field` | str | no | `DBH` | Inventory DBH field name. |
| `--dbh-unit` | enum (`mm`,`cm`,`m`) | no | `mm` | Unit for DBH values. |
| `--default-crown-radius-m` | float | no | `2.5` | Fallback crown radius when DBH is unavailable. |
| `--min-dbh-cm` | float | no | `0.0` | Minimum DBH (cm) filter. |
| `--max-dbh-cm` | float | no | `0.0` | Maximum DBH (cm) filter; `0` disables upper bound. |
| `--deduplicate-tree-id` | flag | no | `False` | Keep only one row per tree id (highest DBH). |
| `--crown-model` | enum (`linear`,`power`) | no | `linear` | Crown radius model. |
| `--linear-factor-m-per-cm` | float | no | `0.08` | Linear model slope. |
| `--linear-intercept-m` | float | no | `0.0` | Linear model intercept. |
| `--power-a` | float | no | `0.15` | Power model coefficient `a`. |
| `--power-b` | float | no | `0.8` | Power model exponent `b`. |
| `--min-crown-radius-m` | float | no | `0.5` | Lower clamp for crown radius. |
| `--max-crown-radius-m` | float | no | `15.0` | Upper clamp for crown radius. |
| `--export-visualizations` | flag | no | `False` | Export labeling visualization PNG files. |
| `--help` | flag | no | - | Show help and exit. |

Example:

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
