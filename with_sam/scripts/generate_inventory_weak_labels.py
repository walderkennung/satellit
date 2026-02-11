#!/usr/bin/env python3
"""
Generate weak tree labels per image tile from inventory data (CSV or SHP).

The script creates labels of the form:
  tiles:
    - tile_id: "tile_x0_y0"
      trees:
        - { x: int, y: int, crown_radius: float }

Coordinates are mapped from inventory positions into image pixel space using either:
1) pixel mode: treat PX/PY as pixel coordinates
2) meter_from_image_ul mode: treat PX/PY as local metric offsets from image UL
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GeoTiffMeta:
    width: int
    height: int
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float
    crs_wkt: str | None
    nodata_band_1: float | None


@dataclass
class InventoryTree:
    tree_id: str
    species: str
    status: str
    local_x: float
    local_y: float
    dbh_raw: float
    dbh_cm: float


@dataclass
class ProjectedTree:
    tree_id: str
    species: str
    status: str
    dbh_cm: float
    crown_radius_m: float
    crown_radius_px: float
    local_x: float
    local_y: float
    utm_x: float | None
    utm_y: float | None
    pixel_x: float
    pixel_y: float


@dataclass
class LocalBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    span_x: float
    span_y: float


@dataclass
class AutoAlignResult:
    swap_local_xy: bool
    mirror_local_x: bool
    mirror_local_y: bool
    rotation_deg: float
    scale_m_per_unit: float
    offset_x_m: float
    offset_y_m: float
    score: float
    inside_fraction: float
    mean_greenness: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate weak tree labels from inventory (CSV or SHP) + orthophoto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image-tif",
        type=Path,
        required=True,
        help="Path to orthophoto GeoTIFF.",
    )
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        required=False,
        help="Path to inventory CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--inventory-shp",
        type=Path,
        required=False,
        help="Path to inventory ESRI Shapefile (.shp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for labels and optional tiles.",
    )

    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in px.")
    parser.add_argument("--overlap", type=int, default=128, help="Tile overlap in px.")
    parser.add_argument(
        "--only-non-empty-tiles",
        action="store_true",
        help="Write only tiles that contain at least one tree.",
    )

    parser.add_argument(
        "--x-field",
        type=str,
        default="PX",
        help="Inventory field containing x coordinate (CSV or SHP attribute).",
    )
    parser.add_argument(
        "--y-field",
        type=str,
        default="PY",
        help="Inventory field containing y coordinate (CSV or SHP attribute).",
    )
    parser.add_argument(
        "--tree-id-field",
        type=str,
        default="TreeID",
        help="Inventory field containing tree id.",
    )
    parser.add_argument(
        "--species-field",
        type=str,
        default="Latin",
        help="Inventory field containing species.",
    )
    parser.add_argument(
        "--status-field",
        type=str,
        default="Status",
        help="Inventory field containing status.",
    )
    parser.add_argument(
        "--status-filter",
        type=str,
        default="alive",
        help="Only keep rows with this status (case-insensitive). Empty disables filter.",
    )
    parser.add_argument(
        "--dbh-field",
        type=str,
        default="DBH",
        help="Inventory field containing diameter at breast height.",
    )
    parser.add_argument(
        "--inventory-coordinates",
        choices=("auto", "local", "utm"),
        default="auto",
        help=(
            "Coordinate mode of inventory positions. "
            "'auto' uses 'utm' for SHP input and 'local' for CSV input."
        ),
    )
    parser.add_argument(
        "--dbh-unit",
        choices=("mm", "cm", "m"),
        default="mm",
        help="Unit of DBH values in CSV.",
    )
    parser.add_argument(
        "--default-crown-radius-m",
        type=float,
        default=2.5,
        help="Fallback crown radius in meters when DBH is unavailable (common for SHP-only input).",
    )
    parser.add_argument(
        "--min-dbh-cm",
        type=float,
        default=0.0,
        help="Exclude trees with DBH below this threshold (in cm).",
    )
    parser.add_argument(
        "--max-dbh-cm",
        type=float,
        default=0.0,
        help="Exclude trees with DBH above this threshold (in cm). Use 0 to disable.",
    )
    parser.add_argument(
        "--deduplicate-tree-id",
        action="store_true",
        help="Keep only one row per tree id (highest DBH).",
    )

    parser.add_argument(
        "--crown-model",
        choices=("linear", "power"),
        default="linear",
        help="Crown radius model.",
    )
    parser.add_argument(
        "--linear-factor-m-per-cm",
        type=float,
        default=0.08,
        help="Linear model slope: crown_radius_m = intercept + factor * dbh_cm.",
    )
    parser.add_argument(
        "--linear-intercept-m",
        type=float,
        default=0.0,
        help="Linear model intercept in meters.",
    )
    parser.add_argument(
        "--power-a",
        type=float,
        default=0.15,
        help="Power model factor: crown_radius_m = a * (dbh_cm ^ b).",
    )
    parser.add_argument(
        "--power-b",
        type=float,
        default=0.8,
        help="Power model exponent.",
    )
    parser.add_argument(
        "--min-crown-radius-m",
        type=float,
        default=0.5,
        help="Lower clamp for crown radius.",
    )
    parser.add_argument(
        "--max-crown-radius-m",
        type=float,
        default=15.0,
        help="Upper clamp for crown radius.",
    )

    parser.add_argument(
        "--transform-mode",
        choices=("meter_from_image_ul", "pixel"),
        default="meter_from_image_ul",
        help="How to transform inventory coordinates into image pixels.",
    )
    parser.add_argument(
        "--scale-m-per-unit",
        type=float,
        default=1.0,
        help="Scale for local coordinate units (meter_from_image_ul mode).",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=0.0,
        help="Rotation applied to local coordinates before mapping.",
    )
    parser.add_argument(
        "--invert-local-y",
        action="store_true",
        help="Invert local Y before rotation.",
    )
    parser.add_argument(
        "--offset-x-m",
        type=float,
        default=0.0,
        help="Additional x offset in meters from image UL (meter_from_image_ul mode).",
    )
    parser.add_argument(
        "--offset-y-m",
        type=float,
        default=0.0,
        help="Additional y offset in meters from image UL (meter_from_image_ul mode).",
    )
    parser.add_argument(
        "--offset-x-px",
        type=float,
        default=0.0,
        help="Additional x offset in pixels (pixel mode).",
    )
    parser.add_argument(
        "--offset-y-px",
        type=float,
        default=0.0,
        help="Additional y offset in pixels (pixel mode).",
    )
    parser.add_argument(
        "--swap-local-xy",
        action="store_true",
        help="Swap local inventory axes before rotation/projection.",
    )
    parser.add_argument(
        "--mirror-local-x",
        action="store_true",
        help="Mirror local X across the local bounds.",
    )
    parser.add_argument(
        "--mirror-local-y",
        action="store_true",
        help="Mirror local Y across the local bounds.",
    )
    parser.add_argument(
        "--anchor-corner",
        choices=("ul", "ur", "ll", "lr"),
        default="ul",
        help=(
            "Anchor transformed local coordinates to an image corner "
            "(UL=upper-left, UR=upper-right, LL=lower-left, LR=lower-right)."
        ),
    )
    parser.add_argument(
        "--auto-align",
        action="store_true",
        help=(
            "Automatically estimate mirror/rotation/offset using GeoTIFF footprint "
            "and image greenness (recommended when inventory has local PX/PY)."
        ),
    )
    parser.add_argument(
        "--auto-align-sample-size",
        type=int,
        default=5000,
        help="Max number of trees sampled during auto-align scoring.",
    )
    parser.add_argument(
        "--auto-align-rotation-min",
        type=float,
        default=-40.0,
        help="Auto-align minimum rotation in degrees.",
    )
    parser.add_argument(
        "--auto-align-rotation-max",
        type=float,
        default=40.0,
        help="Auto-align maximum rotation in degrees.",
    )
    parser.add_argument(
        "--auto-align-rotation-step",
        type=float,
        default=0.5,
        help="Auto-align rotation step in degrees.",
    )
    parser.add_argument(
        "--auto-align-scale-min",
        type=float,
        default=1.0,
        help="Auto-align minimum local->meter scale.",
    )
    parser.add_argument(
        "--auto-align-scale-max",
        type=float,
        default=1.0,
        help="Auto-align maximum local->meter scale.",
    )
    parser.add_argument(
        "--auto-align-scale-step",
        type=float,
        default=0.01,
        help="Auto-align scale step.",
    )
    parser.add_argument(
        "--auto-align-search-swap",
        action="store_true",
        help="Include XY axis swapping in auto-align search.",
    )
    parser.add_argument(
        "--auto-align-disable-greenness",
        action="store_true",
        help="Ignore image greenness term in auto-align scoring.",
    )

    parser.add_argument(
        "--export-tile-images",
        action="store_true",
        help="Save tile images as PNG under output_dir/tiles.",
    )
    parser.add_argument(
        "--tile-image-format",
        choices=("png", "jpg"),
        default="png",
        help="Tile image format for exported tiles.",
    )
    parser.add_argument(
        "--export-visualizations",
        action="store_true",
        help="Generate labeling visualization PNGs.",
    )
    parser.add_argument(
        "--viz-max-dimension",
        type=int,
        default=2400,
        help="Maximum width/height of visualization image in pixels.",
    )
    parser.add_argument(
        "--viz-stroke-width",
        type=float,
        default=1.0,
        help="Stroke width for crown circles in visualization.",
    )
    parser.add_argument(
        "--viz-center-radius",
        type=float,
        default=1.5,
        help="Radius of center points in visualization.",
    )

    return parser.parse_args()


def get_geotiff_meta(image_tif: Path) -> GeoTiffMeta:
    result = subprocess.run(
        ["gdalinfo", "-json", str(image_tif)],
        check=True,
        capture_output=True,
        text=True,
    )
    doc = json.loads(result.stdout)
    width, height = doc["size"]
    gt = doc["geoTransform"]
    crs_wkt = None
    coordinate_system = doc.get("coordinateSystem")
    if isinstance(coordinate_system, dict):
        raw_wkt = coordinate_system.get("wkt")
        if isinstance(raw_wkt, str) and raw_wkt.strip():
            crs_wkt = raw_wkt

    nodata_band_1 = None
    bands = doc.get("bands")
    if isinstance(bands, list) and bands:
        nodata = bands[0].get("noDataValue")
        if isinstance(nodata, (int, float)):
            nodata_band_1 = float(nodata)

    return GeoTiffMeta(
        width=width,
        height=height,
        origin_x=float(gt[0]),
        pixel_size_x=float(gt[1]),
        origin_y=float(gt[3]),
        pixel_size_y=float(gt[5]),
        crs_wkt=crs_wkt,
        nodata_band_1=nodata_band_1,
    )


def to_dbh_cm(dbh_raw: float, dbh_unit: str) -> float:
    if dbh_unit == "mm":
        return dbh_raw / 10.0
    if dbh_unit == "cm":
        return dbh_raw
    return dbh_raw * 100.0


def compute_crown_radius_m(
    dbh_cm: float,
    crown_model: str,
    linear_factor_m_per_cm: float,
    linear_intercept_m: float,
    power_a: float,
    power_b: float,
    min_crown_radius_m: float,
    max_crown_radius_m: float,
) -> float:
    if crown_model == "linear":
        value = linear_intercept_m + linear_factor_m_per_cm * dbh_cm
    else:
        value = power_a * (dbh_cm**power_b)

    if value < min_crown_radius_m:
        return min_crown_radius_m
    if value > max_crown_radius_m:
        return max_crown_radius_m
    return value


def parse_inventory_csv(args: argparse.Namespace) -> list[InventoryTree]:
    if args.inventory_csv is None:
        raise ValueError("CSV parsing requested but --inventory-csv was not provided.")

    trees: list[InventoryTree] = []
    with args.inventory_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            status = (row.get(args.status_field) or "").strip().strip("\r")
            if args.status_filter and status.lower() != args.status_filter.lower():
                continue

            x_str = (row.get(args.x_field) or "").strip().replace(",", ".")
            y_str = (row.get(args.y_field) or "").strip().replace(",", ".")
            dbh_str = (row.get(args.dbh_field) or "").strip().replace(",", ".")
            if not x_str or not y_str or not dbh_str:
                continue

            try:
                local_x = float(x_str)
                local_y = float(y_str)
                dbh_raw = float(dbh_str)
            except ValueError:
                continue

            if dbh_raw <= 0:
                continue

            tree_id = (row.get(args.tree_id_field) or "").strip()
            species = (row.get(args.species_field) or "").strip()
            dbh_cm = to_dbh_cm(dbh_raw=dbh_raw, dbh_unit=args.dbh_unit)
            if dbh_cm < args.min_dbh_cm:
                continue
            if args.max_dbh_cm > 0 and dbh_cm > args.max_dbh_cm:
                continue
            trees.append(
                InventoryTree(
                    tree_id=tree_id,
                    species=species,
                    status=status,
                    local_x=local_x,
                    local_y=local_y,
                    dbh_raw=dbh_raw,
                    dbh_cm=dbh_cm,
                )
            )

    if not args.deduplicate_tree_id:
        return trees

    # Keep one row per TreeID with highest DBH. Empty TreeID rows are kept as-is.
    by_id: dict[str, InventoryTree] = {}
    no_id_rows: list[InventoryTree] = []
    for tree in trees:
        if not tree.tree_id:
            no_id_rows.append(tree)
            continue
        previous = by_id.get(tree.tree_id)
        if previous is None or tree.dbh_cm > previous.dbh_cm:
            by_id[tree.tree_id] = tree

    deduped = list(by_id.values()) + no_id_rows
    deduped.sort(key=lambda t: (t.tree_id, t.local_x, t.local_y))
    return deduped


def _row_value_case_insensitive(row: dict[str, str], key: str) -> str:
    if key in row and row[key] is not None:
        return row[key]
    lowered = key.lower()
    for candidate, value in row.items():
        if candidate.lower() == lowered and value is not None:
            return value
    return ""


def _get_first_present(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = _row_value_case_insensitive(row=row, key=key)
        if value != "":
            return value
    return ""


def parse_inventory_shp(args: argparse.Namespace, meta: GeoTiffMeta) -> list[InventoryTree]:
    if args.inventory_shp is None:
        raise ValueError("SHP parsing requested but --inventory-shp was not provided.")

    with tempfile.TemporaryDirectory(prefix="inv_shp_") as tmpdir:
        csv_path = Path(tmpdir) / "inventory_from_shp.csv"
        cmd = [
            "ogr2ogr",
            "-f",
            "CSV",
            str(csv_path),
            str(args.inventory_shp),
            "-lco",
            "GEOMETRY=AS_XY",
        ]
        # Reproject to raster CRS so UTM->pixel projection is reliable even if SHP CRS differs.
        if meta.crs_wkt:
            cmd.extend(["-t_srs", meta.crs_wkt])

        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "SHAPE_RESTORE_SHX": "YES"},
        )

        trees: list[InventoryTree] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                x_str = _get_first_present(
                    row,
                    [
                        args.x_field,
                        "X",
                        "x",
                        "X_UTM33N",
                        "X_UTM33",
                        "X__UTM3",
                    ],
                ).strip()
                y_str = _get_first_present(
                    row,
                    [
                        args.y_field,
                        "Y",
                        "y",
                        "Y_UTM33N",
                        "Y_UTM33",
                        "Y__UTM3",
                    ],
                ).strip()
                if not x_str or not y_str:
                    continue

                try:
                    x = float(x_str.replace(",", "."))
                    y = float(y_str.replace(",", "."))
                except ValueError:
                    continue

                status = _get_first_present(row, [args.status_field, "status"]).strip()
                if not status:
                    status = "alive"
                status = status.strip("\r")
                if args.status_filter and status.lower() != args.status_filter.lower():
                    continue

                tree_id = _get_first_present(
                    row, [args.tree_id_field, "treeid", "tag", "stemtag"]
                ).strip()
                if not tree_id:
                    tree_id = f"shp_{idx}"
                species = _get_first_present(
                    row, [args.species_field, "species", "latin"]
                ).strip()

                dbh_cm = -1.0
                dbh_raw = -1.0
                dbh_text = _get_first_present(row, [args.dbh_field, "dbh"]).strip()
                if dbh_text:
                    try:
                        dbh_raw = float(dbh_text.replace(",", "."))
                        if dbh_raw > 0:
                            dbh_cm = to_dbh_cm(dbh_raw=dbh_raw, dbh_unit=args.dbh_unit)
                    except ValueError:
                        pass

                # If DBH filtering is enabled, require valid DBH and apply thresholds.
                if args.min_dbh_cm > 0 or args.max_dbh_cm > 0:
                    if dbh_cm <= 0:
                        continue
                    if dbh_cm < args.min_dbh_cm:
                        continue
                    if args.max_dbh_cm > 0 and dbh_cm > args.max_dbh_cm:
                        continue

                trees.append(
                    InventoryTree(
                        tree_id=tree_id,
                        species=species,
                        status=status,
                        local_x=x,
                        local_y=y,
                        dbh_raw=dbh_raw,
                        dbh_cm=dbh_cm,
                    )
                )

    if not args.deduplicate_tree_id:
        return trees

    by_id: dict[str, InventoryTree] = {}
    no_id_rows: list[InventoryTree] = []
    for tree in trees:
        if not tree.tree_id:
            no_id_rows.append(tree)
            continue
        previous = by_id.get(tree.tree_id)
        if previous is None:
            by_id[tree.tree_id] = tree
            continue
        previous_key = previous.dbh_cm if previous.dbh_cm > 0 else -1.0
        current_key = tree.dbh_cm if tree.dbh_cm > 0 else -1.0
        if current_key > previous_key:
            by_id[tree.tree_id] = tree

    deduped = list(by_id.values()) + no_id_rows
    deduped.sort(key=lambda t: (t.tree_id, t.local_x, t.local_y))
    return deduped


def parse_inventory(args: argparse.Namespace, meta: GeoTiffMeta) -> list[InventoryTree]:
    if args.inventory_shp is not None:
        return parse_inventory_shp(args, meta=meta)
    return parse_inventory_csv(args)


def compute_local_bounds(trees: list[InventoryTree]) -> LocalBounds:
    if not trees:
        return LocalBounds(
            min_x=0.0,
            max_x=0.0,
            min_y=0.0,
            max_y=0.0,
            span_x=0.0,
            span_y=0.0,
        )
    min_x = min(t.local_x for t in trees)
    max_x = max(t.local_x for t in trees)
    min_y = min(t.local_y for t in trees)
    max_y = max(t.local_y for t in trees)
    return LocalBounds(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        span_x=max_x - min_x,
        span_y=max_y - min_y,
    )


def _run_gdal_translate(
    image_tif: Path,
    output_path: Path,
    output_format: str,
    extra_args: list[str] | None = None,
) -> None:
    cmd = ["gdal_translate", "-of", output_format]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([str(image_tif), str(output_path)])
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GDAL_PAM_ENABLED": "NO"},
    )


def _parse_pnm(path: Path) -> tuple[str, int, int, int, bytes]:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P5", b"P6"):
            raise ValueError(f"Unsupported PNM format in {path}: {magic!r}")

        def next_token() -> bytes:
            while True:
                line = f.readline()
                if not line:
                    raise EOFError(f"Unexpected EOF while parsing {path}")
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                return line

        dims = next_token().split()
        if len(dims) == 2:
            width, height = int(dims[0]), int(dims[1])
        else:
            width = int(dims[0])
            height = int(next_token())
        maxv = int(next_token())
        payload = f.read()

    return magic.decode("ascii"), width, height, maxv, payload


def _load_valid_mask_from_geotiff(
    image_tif: Path,
    width: int,
    height: int,
    nodata_band_1: float | None,
) -> tuple[bytes, float, float]:
    with tempfile.TemporaryDirectory(prefix="inv_align_") as tmpdir:
        pgm_path = Path(tmpdir) / "band1.pgm"
        _run_gdal_translate(
            image_tif=image_tif,
            output_path=pgm_path,
            output_format="PNM",
            extra_args=["-b", "1"],
        )
        magic, w, h, _, payload = _parse_pnm(pgm_path)

    if magic != "P5":
        raise ValueError(f"Expected grayscale PGM (P5), got {magic}")
    if (w, h) != (width, height):
        raise ValueError(
            f"Mask dimensions mismatch: expected {(width, height)}, got {(w, h)}"
        )

    # Use raster nodata when available. If unavailable, treat all pixels as valid.
    valid = bytearray(w * h)
    sx = 0.0
    sy = 0.0
    n = 0
    nodata_int = None if nodata_band_1 is None else int(round(nodata_band_1))
    for y in range(h):
        row = payload[y * w : (y + 1) * w]
        for x, v in enumerate(row):
            if nodata_int is not None and int(v) == nodata_int:
                continue
            idx = (y * w) + x
            valid[idx] = 1
            sx += x
            sy += y
            n += 1

    if n == 0:
        raise RuntimeError("No valid footprint pixels found for auto-align.")

    return bytes(valid), sx / float(n), sy / float(n)


def _load_greenness_map_from_geotiff(
    image_tif: Path,
    width: int,
    height: int,
) -> list[float]:
    with tempfile.TemporaryDirectory(prefix="inv_align_rgb_") as tmpdir:
        ppm_path = Path(tmpdir) / "rgb.ppm"
        _run_gdal_translate(
            image_tif=image_tif,
            output_path=ppm_path,
            output_format="PNM",
            extra_args=["-b", "1", "-b", "2", "-b", "3"],
        )
        magic, w, h, _, payload = _parse_pnm(ppm_path)

    if magic != "P6":
        raise ValueError(f"Expected RGB PPM (P6), got {magic}")
    if (w, h) != (width, height):
        raise ValueError(
            f"RGB dimensions mismatch: expected {(width, height)}, got {(w, h)}"
        )

    # Excess Green index proxy: 2G - R - B
    # Stored as float list for direct indexing.
    greenness = [0.0] * (w * h)
    for i in range(w * h):
        r = payload[(3 * i) + 0]
        g = payload[(3 * i) + 1]
        b = payload[(3 * i) + 2]
        greenness[i] = float((2 * g) - r - b)
    return greenness


def _linspace_step(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    values: list[float] = []
    if start <= end:
        v = start
        while v <= end + 1e-12:
            values.append(round(v, 10))
            v += step
        return values
    v = start
    while v >= end - 1e-12:
        values.append(round(v, 10))
        v -= step
    return values


def _sample_trees_for_auto_align(
    trees: list[InventoryTree],
    sample_size: int,
) -> list[InventoryTree]:
    if sample_size <= 0 or len(trees) <= sample_size:
        return trees
    step = max(1, len(trees) // sample_size)
    sampled = trees[::step]
    if len(sampled) > sample_size:
        sampled = sampled[:sample_size]
    return sampled


def _prepare_local_vectors(
    trees: list[InventoryTree],
    bounds: LocalBounds,
    swap_xy: bool,
    mirror_x: bool,
    mirror_y: bool,
) -> list[tuple[float, float]]:
    vectors: list[tuple[float, float]] = []
    for tree in trees:
        x, y = normalize_local_coordinates(
            local_x=tree.local_x,
            local_y=tree.local_y,
            bounds=bounds,
            swap_xy=swap_xy,
            mirror_x=mirror_x,
            mirror_y=mirror_y,
        )
        vectors.append((x, y))
    return vectors


def auto_align_transform(
    args: argparse.Namespace,
    trees: list[InventoryTree],
    bounds: LocalBounds,
    meta: GeoTiffMeta,
) -> AutoAlignResult:
    sampled = _sample_trees_for_auto_align(
        trees=trees,
        sample_size=args.auto_align_sample_size,
    )

    valid_mask, mask_cx, mask_cy = _load_valid_mask_from_geotiff(
        image_tif=args.image_tif,
        width=meta.width,
        height=meta.height,
        nodata_band_1=meta.nodata_band_1,
    )
    use_greenness = not args.auto_align_disable_greenness
    greenness = (
        _load_greenness_map_from_geotiff(
            image_tif=args.image_tif,
            width=meta.width,
            height=meta.height,
        )
        if use_greenness
        else []
    )

    px_to_m_x = abs(meta.pixel_size_x)
    px_to_m_y = abs(meta.pixel_size_y)
    target_cx_m = mask_cx * px_to_m_x
    target_cy_m = mask_cy * px_to_m_y

    swap_options = [False, True] if args.auto_align_search_swap else [False]
    mirror_options = [False, True]

    rotations = _linspace_step(
        start=args.auto_align_rotation_min,
        end=args.auto_align_rotation_max,
        step=args.auto_align_rotation_step,
    )
    scales = _linspace_step(
        start=args.auto_align_scale_min,
        end=args.auto_align_scale_max,
        step=args.auto_align_scale_step,
    )

    best: AutoAlignResult | None = None
    width = meta.width
    height = meta.height

    for swap_xy in swap_options:
        for mirror_x in mirror_options:
            for mirror_y in mirror_options:
                base = _prepare_local_vectors(
                    trees=sampled,
                    bounds=bounds,
                    swap_xy=swap_xy,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                )
                for rotation_deg in rotations:
                    theta = math.radians(rotation_deg)
                    c = math.cos(theta)
                    s = math.sin(theta)

                    rotated = []
                    sum_x = 0.0
                    sum_y = 0.0
                    for x, y in base:
                        xr = (c * x) - (s * y)
                        yr = (s * x) + (c * y)
                        rotated.append((xr, yr))
                        sum_x += xr
                        sum_y += yr

                    if not rotated:
                        continue

                    for scale in scales:
                        scaled = []
                        sx = 0.0
                        sy = 0.0
                        for xr, yr in rotated:
                            xs = xr * scale
                            ys = yr * scale
                            scaled.append((xs, ys))
                            sx += xs
                            sy += ys

                        center_x_m = sx / float(len(scaled))
                        center_y_m = sy / float(len(scaled))
                        offset_x_m = target_cx_m - center_x_m
                        offset_y_m = target_cy_m - center_y_m

                        inside = 0
                        in_bounds = 0
                        green_sum = 0.0

                        for xs, ys in scaled:
                            px = int(round((xs + offset_x_m) / px_to_m_x))
                            py = int(round((ys + offset_y_m) / px_to_m_y))
                            if px < 0 or py < 0 or px >= width or py >= height:
                                continue
                            in_bounds += 1
                            idx = (py * width) + px
                            if valid_mask[idx] == 0:
                                continue
                            inside += 1
                            if use_greenness:
                                green_sum += greenness[idx]

                        if in_bounds == 0:
                            continue

                        inside_fraction = inside / float(in_bounds)
                        mean_green = (green_sum / float(inside)) if inside > 0 else -999.0

                        # Prioritize footprint consistency, then image-consistent vegetation.
                        score = (inside_fraction * 1000.0) + (mean_green * 0.10)

                        candidate = AutoAlignResult(
                            swap_local_xy=swap_xy,
                            mirror_local_x=mirror_x,
                            mirror_local_y=mirror_y,
                            rotation_deg=rotation_deg,
                            scale_m_per_unit=scale,
                            offset_x_m=offset_x_m,
                            offset_y_m=offset_y_m,
                            score=score,
                            inside_fraction=inside_fraction,
                            mean_greenness=mean_green,
                        )
                        if best is None or candidate.score > best.score:
                            best = candidate

    if best is None:
        raise RuntimeError("Auto-align failed to find a valid transform candidate.")
    return best


def normalize_local_coordinates(
    local_x: float,
    local_y: float,
    bounds: LocalBounds,
    swap_xy: bool,
    mirror_x: bool,
    mirror_y: bool,
) -> tuple[float, float]:
    # Normalize to local bounds first so mirror operations are stable.
    x = local_x - bounds.min_x
    y = local_y - bounds.min_y

    span_x = bounds.span_x
    span_y = bounds.span_y
    if swap_xy:
        x, y = y, x
        span_x, span_y = span_y, span_x

    if mirror_x:
        x = span_x - x
    if mirror_y:
        y = span_y - y

    return x, y


def project_tree(
    tree: InventoryTree,
    args: argparse.Namespace,
    meta: GeoTiffMeta,
    bounds: LocalBounds,
    coordinate_mode: str,
) -> ProjectedTree:
    if coordinate_mode == "utm":
        utm_x = tree.local_x
        utm_y = tree.local_y
        pixel_x = (utm_x - meta.origin_x) / meta.pixel_size_x
        pixel_y = (utm_y - meta.origin_y) / meta.pixel_size_y
    else:
        local_x, local_y = normalize_local_coordinates(
            local_x=tree.local_x,
            local_y=tree.local_y,
            bounds=bounds,
            swap_xy=args.swap_local_xy,
            mirror_x=args.mirror_local_x,
            mirror_y=args.mirror_local_y,
        )

        theta = math.radians(args.rotation_deg)
        if args.invert_local_y:
            local_y = -local_y
        x_rot = (math.cos(theta) * local_x) - (math.sin(theta) * local_y)
        y_rot = (math.sin(theta) * local_x) + (math.cos(theta) * local_y)

        if args.transform_mode == "pixel":
            x_px = x_rot * args.scale_m_per_unit
            y_px = y_rot * args.scale_m_per_unit

            if args.anchor_corner in ("ur", "lr"):
                x_px = float(meta.width) - x_px
            if args.anchor_corner in ("ll", "lr"):
                y_px = float(meta.height) - y_px

            pixel_x = x_px + args.offset_x_px
            pixel_y = y_px + args.offset_y_px
            utm_x = None
            utm_y = None
        else:
            x_m = x_rot * args.scale_m_per_unit
            y_m = y_rot * args.scale_m_per_unit

            image_width_m = abs(meta.pixel_size_x) * float(meta.width)
            image_height_m = abs(meta.pixel_size_y) * float(meta.height)
            if args.anchor_corner in ("ur", "lr"):
                x_m = image_width_m - x_m
            if args.anchor_corner in ("ll", "lr"):
                y_m = image_height_m - y_m

            x_m += args.offset_x_m
            y_m += args.offset_y_m
            utm_x = meta.origin_x + x_m
            utm_y = meta.origin_y - y_m
            pixel_x = (utm_x - meta.origin_x) / meta.pixel_size_x
            pixel_y = (utm_y - meta.origin_y) / meta.pixel_size_y

    if tree.dbh_cm > 0:
        crown_radius_m = compute_crown_radius_m(
            dbh_cm=tree.dbh_cm,
            crown_model=args.crown_model,
            linear_factor_m_per_cm=args.linear_factor_m_per_cm,
            linear_intercept_m=args.linear_intercept_m,
            power_a=args.power_a,
            power_b=args.power_b,
            min_crown_radius_m=args.min_crown_radius_m,
            max_crown_radius_m=args.max_crown_radius_m,
        )
    else:
        crown_radius_m = max(args.min_crown_radius_m, args.default_crown_radius_m)
    # Use average absolute pixel size to be robust to sign convention.
    meter_to_px = 1.0 / ((abs(meta.pixel_size_x) + abs(meta.pixel_size_y)) / 2.0)
    crown_radius_px = crown_radius_m * meter_to_px

    return ProjectedTree(
        tree_id=tree.tree_id,
        species=tree.species,
        status=tree.status,
        dbh_cm=tree.dbh_cm,
        crown_radius_m=crown_radius_m,
        crown_radius_px=crown_radius_px,
        local_x=tree.local_x,
        local_y=tree.local_y,
        utm_x=utm_x,
        utm_y=utm_y,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
    )


def build_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[dict]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    step = tile_size - overlap
    tiles: list[dict] = []
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tiles.append(
                {
                    "tile_id": f"tile_x{x0}_y{y0}",
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "trees": [],
                }
            )
    return tiles


def circle_intersects_rect(
    center_x: float,
    center_y: float,
    radius: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> bool:
    nearest_x = min(max(center_x, x0), x1)
    nearest_y = min(max(center_y, y0), y1)
    dx = center_x - nearest_x
    dy = center_y - nearest_y
    return (dx * dx + dy * dy) <= (radius * radius)


def assign_trees_to_tiles(
    trees: list[ProjectedTree],
    tiles: list[dict],
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    in_image = 0
    outside_image = 0

    for tree in trees:
        if not (0 <= tree.pixel_x < image_width and 0 <= tree.pixel_y < image_height):
            outside_image += 1
            continue

        in_image += 1
        for tile in tiles:
            if not circle_intersects_rect(
                center_x=tree.pixel_x,
                center_y=tree.pixel_y,
                radius=tree.crown_radius_px,
                x0=tile["x0"],
                y0=tile["y0"],
                x1=tile["x1"],
                y1=tile["y1"],
            ):
                continue

            tile["trees"].append(
                {
                    "tree_id": tree.tree_id,
                    "species": tree.species,
                    "x": int(round(tree.pixel_x - tile["x0"])),
                    "y": int(round(tree.pixel_y - tile["y0"])),
                    "crown_radius": round(tree.crown_radius_px, 3),
                    "crown_radius_m": round(tree.crown_radius_m, 3),
                    "dbh_cm": round(tree.dbh_cm, 3),
                    "x_global": round(tree.pixel_x, 3),
                    "y_global": round(tree.pixel_y, 3),
                }
            )

    return in_image, outside_image


def export_tile_images(
    image_tif: Path,
    tiles: list[dict],
    output_dir: Path,
    image_format: str,
) -> None:
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    output_format = "PNG" if image_format == "png" else "JPEG"

    for tile in tiles:
        x0, y0 = tile["x0"], tile["y0"]
        width, height = tile["width"], tile["height"]
        file_path = tiles_dir / f"{tile['tile_id']}.{image_format}"
        subprocess.run(
            [
                "gdal_translate",
                "-of",
                output_format,
                "-srcwin",
                str(x0),
                str(y0),
                str(width),
                str(height),
                str(image_tif),
                str(file_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GDAL_PAM_ENABLED": "NO"},
        )


def write_yaml(tiles: list[dict], yaml_path: Path) -> None:
    # YAML 1.2 is a superset of JSON. Writing JSON here avoids manual string-escaping bugs.
    with yaml_path.open("w", encoding="utf-8") as f:
        json.dump({"tiles": tiles}, f, ensure_ascii=True, indent=2)


def write_projected_csv(projected_trees: list[ProjectedTree], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tree_id",
                "species",
                "status",
                "local_x",
                "local_y",
                "utm_x",
                "utm_y",
                "pixel_x",
                "pixel_y",
                "dbh_cm",
                "crown_radius_m",
                "crown_radius_px",
            ]
        )
        for tree in projected_trees:
            writer.writerow(
                [
                    tree.tree_id,
                    tree.species,
                    tree.status,
                    tree.local_x,
                    tree.local_y,
                    "" if tree.utm_x is None else round(tree.utm_x, 6),
                    "" if tree.utm_y is None else round(tree.utm_y, 6),
                    round(tree.pixel_x, 6),
                    round(tree.pixel_y, 6),
                    round(tree.dbh_cm, 6),
                    round(tree.crown_radius_m, 6),
                    round(tree.crown_radius_px, 6),
                ]
            )


def _ensure_magick_available() -> str:
    result = subprocess.run(
        ["which", "magick"],
        check=False,
        capture_output=True,
        text=True,
    )
    path = result.stdout.strip()
    if not path:
        raise RuntimeError(
            "ImageMagick ('magick') is required for --export-visualizations but was not found in PATH."
        )
    return path


def _projected_trees_inside_image(
    projected_trees: list[ProjectedTree],
    width: int,
    height: int,
) -> list[ProjectedTree]:
    return [
        tree
        for tree in projected_trees
        if 0 <= tree.pixel_x < width and 0 <= tree.pixel_y < height
    ]


def export_visualizations(
    image_tif: Path,
    projected_trees: list[ProjectedTree],
    tiles: list[dict],
    output_dir: Path,
    meta: GeoTiffMeta,
    args: argparse.Namespace,
) -> dict[str, str]:
    _ensure_magick_available()

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Preserve aspect ratio while keeping one side capped for fast QA previews.
    scale = min(1.0, float(args.viz_max_dimension) / float(max(meta.width, meta.height)))
    scaled_width = max(1, int(round(meta.width * scale)))
    scaled_height = max(1, int(round(meta.height * scale)))

    base_png = viz_dir / "base.png"
    crown_overlay_png = viz_dir / "labels_crowns.png"
    crowns_and_tiles_png = viz_dir / "labels_crowns_tiles.png"
    crown_draw_file = viz_dir / "draw_crowns.mvg"
    tile_draw_file = viz_dir / "draw_tiles.mvg"

    subprocess.run(
        [
            "gdal_translate",
            "-of",
            "PNG",
            "-outsize",
            str(scaled_width),
            str(scaled_height),
            str(image_tif),
            str(base_png),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GDAL_PAM_ENABLED": "NO"},
    )

    visible_trees = _projected_trees_inside_image(
        projected_trees=projected_trees,
        width=meta.width,
        height=meta.height,
    )

    with crown_draw_file.open("w", encoding="utf-8") as f:
        f.write("push graphic-context\n")
        for tree in visible_trees:
            x = tree.pixel_x * scale
            y = tree.pixel_y * scale
            radius = max(tree.crown_radius_px * scale, 0.5)
            center_radius = max(args.viz_center_radius, 0.5)
            f.write(
                "stroke 'rgba(255,0,0,0.58)' "
                f"stroke-width {args.viz_stroke_width:.3f} "
                "fill 'none' "
                f"circle {x:.3f},{y:.3f} {x + radius:.3f},{y:.3f}\n"
            )
            f.write(
                "stroke 'none' fill 'rgba(255,255,0,0.90)' "
                f"circle {x:.3f},{y:.3f} {x + center_radius:.3f},{y:.3f}\n"
            )
        f.write("pop graphic-context\n")

    subprocess.run(
        [
            "magick",
            str(base_png),
            "-draw",
            f"@{crown_draw_file}",
            str(crown_overlay_png),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with tile_draw_file.open("w", encoding="utf-8") as f:
        f.write("push graphic-context\n")
        for tile in tiles:
            x0 = tile["x0"] * scale
            y0 = tile["y0"] * scale
            x1 = tile["x1"] * scale
            y1 = tile["y1"] * scale
            f.write(
                "stroke 'rgba(0,255,255,0.85)' "
                "stroke-width 1 "
                "fill 'none' "
                f"rectangle {x0:.3f},{y0:.3f} {x1:.3f},{y1:.3f}\n"
            )
        f.write("pop graphic-context\n")

    subprocess.run(
        [
            "magick",
            str(crown_overlay_png),
            "-draw",
            f"@{tile_draw_file}",
            str(crowns_and_tiles_png),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return {
        "visualization_base": str(base_png),
        "visualization_crowns": str(crown_overlay_png),
        "visualization_crowns_tiles": str(crowns_and_tiles_png),
        "visualization_scale": f"{scale:.6f}",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.inventory_csv is None and args.inventory_shp is None:
        raise ValueError("Provide either --inventory-csv or --inventory-shp.")

    meta = get_geotiff_meta(args.image_tif)
    trees = parse_inventory(args, meta=meta)
    bounds = compute_local_bounds(trees=trees)

    if args.inventory_coordinates == "auto":
        coordinate_mode = "utm" if args.inventory_shp is not None else "local"
    else:
        coordinate_mode = args.inventory_coordinates

    auto_align_summary: dict[str, float | bool] = {}
    if args.auto_align and coordinate_mode == "local":
        fit = auto_align_transform(
            args=args,
            trees=trees,
            bounds=bounds,
            meta=meta,
        )
        args.swap_local_xy = fit.swap_local_xy
        args.mirror_local_x = fit.mirror_local_x
        args.mirror_local_y = fit.mirror_local_y
        args.rotation_deg = fit.rotation_deg
        args.scale_m_per_unit = fit.scale_m_per_unit
        args.offset_x_m = fit.offset_x_m
        args.offset_y_m = fit.offset_y_m
        args.transform_mode = "meter_from_image_ul"
        args.anchor_corner = "ul"
        auto_align_summary = {
            "enabled": True,
            "score": fit.score,
            "inside_fraction": fit.inside_fraction,
            "mean_greenness": fit.mean_greenness,
        }
        print("Auto-align selected transform:")
        print(f"  swap_local_xy={fit.swap_local_xy}")
        print(f"  mirror_local_x={fit.mirror_local_x}")
        print(f"  mirror_local_y={fit.mirror_local_y}")
        print(f"  rotation_deg={fit.rotation_deg}")
        print(f"  scale_m_per_unit={fit.scale_m_per_unit}")
        print(f"  offset_x_m={fit.offset_x_m:.4f}")
        print(f"  offset_y_m={fit.offset_y_m:.4f}")
    elif args.auto_align and coordinate_mode == "utm":
        print(
            "Auto-align skipped: inventory coordinates are interpreted as UTM "
            "and are mapped directly using GeoTIFF geotransform."
        )

    projected = [
        project_tree(
            tree=t,
            args=args,
            meta=meta,
            bounds=bounds,
            coordinate_mode=coordinate_mode,
        )
        for t in trees
    ]

    tiles = build_tiles(
        width=meta.width,
        height=meta.height,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    in_image, outside_image = assign_trees_to_tiles(
        trees=projected,
        tiles=tiles,
        image_width=meta.width,
        image_height=meta.height,
    )

    if args.only_non_empty_tiles:
        tiles = [t for t in tiles if t["trees"]]

    if args.export_tile_images:
        export_tile_images(
            image_tif=args.image_tif,
            tiles=tiles,
            output_dir=args.output_dir,
            image_format=args.tile_image_format,
        )

    json_path = args.output_dir / "labels_tiles.json"
    yaml_path = args.output_dir / "labels_tiles.yaml"
    projected_csv_path = args.output_dir / "trees_projected.csv"
    summary_path = args.output_dir / "summary.json"
    visualization_outputs: dict[str, str] = {}

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"tiles": tiles}, f, ensure_ascii=True, indent=2)

    write_yaml(tiles=tiles, yaml_path=yaml_path)
    write_projected_csv(projected_trees=projected, path=projected_csv_path)

    if args.export_visualizations:
        visualization_outputs = export_visualizations(
            image_tif=args.image_tif,
            projected_trees=projected,
            tiles=tiles,
            output_dir=args.output_dir,
            meta=meta,
            args=args,
        )

    total_assigned = sum(len(t["trees"]) for t in tiles)
    summary = {
        "image": str(args.image_tif),
        "inventory_csv": None if args.inventory_csv is None else str(args.inventory_csv),
        "inventory_shp": None if args.inventory_shp is None else str(args.inventory_shp),
        "inventory_coordinate_mode": coordinate_mode,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "min_dbh_cm": args.min_dbh_cm,
        "max_dbh_cm": args.max_dbh_cm,
        "tile_count": len(tiles),
        "total_inventory_rows_after_filtering": len(trees),
        "trees_inside_image": in_image,
        "trees_outside_image": outside_image,
        "assigned_tree_instances_across_tiles": total_assigned,
        "transform_mode": args.transform_mode,
        "anchor_corner": args.anchor_corner,
        "rotation_deg": args.rotation_deg,
        "swap_local_xy": args.swap_local_xy,
        "mirror_local_x": args.mirror_local_x,
        "mirror_local_y": args.mirror_local_y,
        "scale_m_per_unit": args.scale_m_per_unit,
        "offset_x_m": args.offset_x_m,
        "offset_y_m": args.offset_y_m,
        "offset_x_px": args.offset_x_px,
        "offset_y_px": args.offset_y_px,
        "auto_align": auto_align_summary,
        "local_bounds": {
            "min_x": bounds.min_x,
            "max_x": bounds.max_x,
            "min_y": bounds.min_y,
            "max_y": bounds.max_y,
            "span_x": bounds.span_x,
            "span_y": bounds.span_y,
        },
        "crown_model": args.crown_model,
        "outputs": {
            "json": str(json_path),
            "yaml": str(yaml_path),
            "projected_csv": str(projected_csv_path),
            **visualization_outputs,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print("Weak label generation complete.")
    print(f"- Tiles written: {len(tiles)}")
    print(f"- Trees inside image: {in_image}")
    print(f"- Trees outside image: {outside_image}")
    print(f"- Assigned instances across tiles: {total_assigned}")
    print(f"- Labels JSON: {json_path}")
    print(f"- Labels YAML: {yaml_path}")
    print(f"- Projected tree CSV: {projected_csv_path}")
    if visualization_outputs:
        print(f"- Visualization (crowns): {visualization_outputs['visualization_crowns']}")
        print(
            "- Visualization (crowns + tiles): "
            + visualization_outputs["visualization_crowns_tiles"]
        )
    print(f"- Summary: {summary_path}")


if __name__ == "__main__":
    main()
