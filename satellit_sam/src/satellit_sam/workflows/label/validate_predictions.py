"""Validate SAM3 strong labels against inventory stem positions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import osr

from satellit_sam.core.allometry import DbhUnit
from satellit_sam.core.geotiff import GeoTiffMeta
from satellit_sam.core.inventory import Inventory
from satellit_sam.core.tree import Tree


@dataclass
class _CandidateMatch:
    """Candidate tree-to-label relation used for one-to-one assignment."""

    tree_index: int
    label_id: int
    score: float
    mask_area: int


@dataclass
class _TileLabel:
    """One label extracted from one per-tile prediction NPZ."""

    label_id: int
    score: float
    mask_area: int
    tile_origin: tuple[int, int]
    tile_size: tuple[int, int]
    tile_mask: np.ndarray


@dataclass
class _LoadedPredictions:
    """Unified representation for merged and tile-based prediction inputs."""

    source_mode: str
    merged_masks: np.ndarray | None
    merged_scores: np.ndarray | None
    merged_mask_areas: np.ndarray | None
    tile_labels_by_tile: dict[tuple[int, int], list[_TileLabel]] | None
    tile_extents: list[tuple[int, int, int, int]] | None
    total_labels: int


def validate_sam3_predictions(
    image_tif: Path,
    predictions_npz: Path | None = None,
    predictions_tiles_dir: Path | None = None,
    output_csv: Path = Path("output/validation/label_validation.csv"),
    inventory_csv: Path | None = None,
    inventory_shp: Path | None = None,
    x_field: str = "PX",
    y_field: str = "PY",
    tree_id_field: str = "TreeID",
    stem_id_field: str = "StemTag",
    species_field: str = "Latin",
    status_field: str = "Status",
    status_filter: str = "alive",
    dbh_field: str = "DBH",
    dbh_unit: DbhUnit = "mm",
    min_dbh_cm: float = 0.0,
    max_dbh_cm: float = 0.0,
    deduplicate_tree_id: bool = False,
) -> pd.DataFrame:
    """Validate SAM3 strong-label masks against inventory stems.

    Args:
        image_tif: GeoTIFF used to map inventory trees into image pixel space.
        predictions_npz: Optional path to SAM3 ``image_masks.npz`` artifact.
        predictions_tiles_dir: Optional path to per-tile mask NPZ directory.
        output_csv: Output CSV path for validation rows.
        inventory_csv: Optional inventory CSV path.
        inventory_shp: Optional inventory SHP path.
        x_field: Inventory CSV x-offset field.
        y_field: Inventory CSV y-offset field.
        tree_id_field: Inventory tree-id field.
        stem_id_field: Inventory stem-id field.
        species_field: Inventory species field.
        status_field: Inventory status field.
        status_filter: Optional status filter (case-insensitive).
        dbh_field: Inventory DBH field.
        dbh_unit: Unit for DBH values.
        min_dbh_cm: Minimum DBH threshold in cm.
        max_dbh_cm: Maximum DBH threshold in cm (``<=0`` disables upper bound).
        deduplicate_tree_id: Keep one row per tree id with highest DBH.

    Returns:
        DataFrame with columns
        ``tree_id, stem_id, label_id, tree_pos_x, tree_pos_y, prediction_coverage``.

    Raises:
        ValueError: If inputs are invalid or prediction payload is malformed.
    """
    if inventory_csv is not None and inventory_shp is not None:
        raise ValueError("Provide only one of `inventory_csv` or `inventory_shp`.")
    if inventory_csv is None and inventory_shp is None:
        raise ValueError(
            "No inventory source provided. Provide either `inventory_csv` or `inventory_shp`."
        )

    meta = GeoTiffMeta.load_tif(image_tif)
    inventory = _load_inventory(
        meta=meta,
        inventory_csv=inventory_csv,
        inventory_shp=inventory_shp,
        x_field=x_field,
        y_field=y_field,
        tree_id_field=tree_id_field,
        stem_id_field=stem_id_field,
        species_field=species_field,
        status_field=status_field,
        status_filter=status_filter,
        dbh_field=dbh_field,
        dbh_unit=dbh_unit,
        min_dbh_cm=min_dbh_cm,
        max_dbh_cm=max_dbh_cm,
        deduplicate_tree_id=deduplicate_tree_id,
    )

    loaded_predictions = _select_prediction_source(
        image_width=meta.width,
        image_height=meta.height,
        predictions_npz=predictions_npz,
        predictions_tiles_dir=predictions_tiles_dir,
    )

    wgs84_to_image = _build_wgs84_to_image_crs_transform(meta=meta)
    utm_fallback = _infer_utm_fallback_from_image_name(image_tif=image_tif)
    if meta.crs_wkt is None and wgs84_to_image is None and utm_fallback is None:
        raise ValueError(
            "GeoTIFF CRS is missing. Could not infer fallback UTM zone from image "
            "filename. Add CRS metadata to the image (recommended) or include "
            "'utm<zone><n|s>' in the filename (e.g. '...utm33n...')."
        )

    rows: list[dict[str, object]] = []
    candidates: list[_CandidateMatch] = []

    for tree_index, tree in enumerate(inventory.trees):
        row = {
            "tree_id": tree.tree_id,
            "stem_id": _stem_id_for_tree(tree=tree),
            "label_id": None,
            "tree_pos_x": tree.x_wgs84,
            "tree_pos_y": tree.y_wgs84,
            "prediction_coverage": False,
        }

        pixel_x, pixel_y = _wgs84_tree_to_pixel(
            lon=tree.x_wgs84,
            lat=tree.y_wgs84,
            meta=meta,
            wgs84_to_image=wgs84_to_image,
            utm_fallback=utm_fallback,
        )

        stem_x = int(round(pixel_x))
        stem_y = int(round(pixel_y))

        if loaded_predictions.source_mode == "merged_npz":
            if stem_x < 0 or stem_y < 0 or stem_x >= meta.width or stem_y >= meta.height:
                rows.append(row)
                continue

            row["prediction_coverage"] = True
            masks = loaded_predictions.merged_masks
            scores = loaded_predictions.merged_scores
            mask_areas = loaded_predictions.merged_mask_areas
            assert masks is not None
            assert scores is not None
            assert mask_areas is not None

            label_ids = np.where(masks[:, stem_y, stem_x])[0]
            for label_id in label_ids.tolist():
                candidates.append(
                    _CandidateMatch(
                        tree_index=tree_index,
                        label_id=int(label_id),
                        score=float(scores[label_id]),
                        mask_area=int(mask_areas[label_id]),
                    )
                )
            rows.append(row)
            continue

        if stem_x < 0 or stem_y < 0 or stem_x >= meta.width or stem_y >= meta.height:
            rows.append(row)
            continue

        tile_extents = loaded_predictions.tile_extents or []
        if not _point_in_any_extent(stem_x=stem_x, stem_y=stem_y, extents=tile_extents):
            rows.append(row)
            continue

        row["prediction_coverage"] = True
        tile_labels_by_tile = loaded_predictions.tile_labels_by_tile or {}
        for tile_origin, tile_labels in tile_labels_by_tile.items():
            tile_x0, tile_y0 = tile_origin
            if not tile_labels:
                continue

            tile_width, tile_height = tile_labels[0].tile_size
            tile_x1 = tile_x0 + tile_width
            tile_y1 = tile_y0 + tile_height
            if stem_x < tile_x0 or stem_y < tile_y0 or stem_x >= tile_x1 or stem_y >= tile_y1:
                continue

            local_x = stem_x - tile_x0
            local_y = stem_y - tile_y0
            for label in tile_labels:
                if local_x >= label.tile_mask.shape[1] or local_y >= label.tile_mask.shape[0]:
                    continue
                if not bool(label.tile_mask[local_y, local_x]):
                    continue
                candidates.append(
                    _CandidateMatch(
                        tree_index=tree_index,
                        label_id=label.label_id,
                        score=label.score,
                        mask_area=label.mask_area,
                    )
                )

        rows.append(row)

    used_labels: set[int] = set()
    assigned_trees: set[int] = set()
    if candidates:
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.mask_area,
                candidate.label_id,
                candidate.tree_index,
            )
        )

        for candidate in candidates:
            if candidate.tree_index in assigned_trees:
                continue
            if candidate.label_id in used_labels:
                continue
            rows[candidate.tree_index]["label_id"] = candidate.label_id
            used_labels.add(candidate.label_id)
            assigned_trees.add(candidate.tree_index)

    result = pd.DataFrame(
        rows,
        columns=[
            "tree_id",
            "stem_id",
            "label_id",
            "tree_pos_x",
            "tree_pos_y",
            "prediction_coverage",
        ],
    )
    result["label_id"] = pd.Series(
        [row["label_id"] for row in rows],
        dtype="object",
    )
    result["prediction_coverage"] = pd.Series(
        [bool(row["prediction_coverage"]) for row in rows],
        dtype=bool,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    total = len(result)
    covered = int(result["prediction_coverage"].sum())
    uncovered = total - covered
    matched = int(len(used_labels))
    unmatched = total - matched
    unmatched_mask_labels = max(0, int(loaded_predictions.total_labels - matched))
    match_rate = 0.0 if total == 0 else (matched / total) * 100.0
    true_positives = matched
    false_positives = unmatched_mask_labels
    false_negatives = unmatched

    print(f"Validation results saved to: {output_csv}")
    print("Validation summary:")
    print(f"- source mode: {loaded_predictions.source_mode}")
    print(f"- total trees: {total}")
    print(f"- covered trees: {covered}")
    print(f"- uncovered trees: {uncovered}")
    print(f"- matched trees: {matched}")
    print(f"- unmatched trees: {unmatched}")
    print(f"- unmatched mask labels: {unmatched_mask_labels}")
    print(f"- match rate: {match_rate:.2f}%")
    print("Confusion matrix:")
    print(f"- true positives: {true_positives}")
    print(f"- false positives: {false_positives}")
    print(f"- false negatives: {false_negatives}")

    return result


def _select_prediction_source(
    image_width: int,
    image_height: int,
    predictions_npz: Path | None,
    predictions_tiles_dir: Path | None,
) -> _LoadedPredictions:
    """Load prediction artifacts with source-priority semantics."""
    if predictions_npz is not None:
        return _load_predictions_from_merged_npz(
            image_width=image_width,
            image_height=image_height,
            predictions_npz=predictions_npz,
        )
    if predictions_tiles_dir is not None:
        return _load_predictions_from_tiles_dir(
            image_width=image_width,
            image_height=image_height,
            predictions_tiles_dir=predictions_tiles_dir,
        )
    raise ValueError(
        "Provide at least one predictions source: `predictions_npz` or `predictions_tiles_dir`."
    )


def _load_predictions_from_merged_npz(
    image_width: int,
    image_height: int,
    predictions_npz: Path,
) -> _LoadedPredictions:
    """Load merged ``image_masks.npz`` predictions."""
    with np.load(predictions_npz, allow_pickle=False) as predictions:
        if "masks" not in predictions:
            raise ValueError("Predictions NPZ is missing required 'masks' array.")
        masks = np.asarray(predictions["masks"], dtype=bool)

        if masks.ndim != 3:
            raise ValueError(
                "Predictions NPZ 'masks' must have shape (N, H, W) for SAM3 labels."
            )

        label_count = int(masks.shape[0])
        if "scores" in predictions:
            scores = np.asarray(predictions["scores"], dtype=np.float32).reshape(-1)
        else:
            scores = np.ones((label_count,), dtype=np.float32)

    if scores.shape[0] != label_count:
        raise ValueError(
            "Predictions NPZ 'scores' length must match number of masks. "
            f"scores={scores.shape[0]} masks={label_count}."
        )

    mask_height = int(masks.shape[1]) if masks.ndim == 3 else 0
    mask_width = int(masks.shape[2]) if masks.ndim == 3 else 0
    if mask_width != image_width or mask_height != image_height:
        raise ValueError(
            "Prediction mask shape does not match image dimensions: "
            f"masks=({mask_width},{mask_height}) image=({image_width},{image_height})."
        )

    mask_areas = masks.reshape(label_count, -1).sum(axis=1).astype(np.int64)
    return _LoadedPredictions(
        source_mode="merged_npz",
        merged_masks=masks,
        merged_scores=scores,
        merged_mask_areas=mask_areas,
        tile_labels_by_tile=None,
        tile_extents=None,
        total_labels=label_count,
    )


def _load_predictions_from_tiles_dir(
    image_width: int,
    image_height: int,
    predictions_tiles_dir: Path,
) -> _LoadedPredictions:
    """Load per-tile predictions from ``masks/tiles`` directory."""
    tile_paths = sorted(predictions_tiles_dir.glob("tile_x*_y*.npz"))
    if not tile_paths:
        raise ValueError(
            "No per-tile prediction NPZ files found in predictions_tiles_dir. "
            "Expected files like tile_x0_y0.npz."
        )

    parsed_tiles: list[
        tuple[
            str,
            tuple[int, int],
            tuple[int, int],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    for tile_path in tile_paths:
        with np.load(tile_path, allow_pickle=False) as tile_npz:
            missing = [
                key
                for key in ("tile_origin", "tile_size", "masks", "boxes", "scores")
                if key not in tile_npz
            ]
            if missing:
                raise ValueError(
                    f"Tile NPZ {tile_path} is missing required fields: {', '.join(missing)}."
                )

            tile_origin_values = np.asarray(tile_npz["tile_origin"], dtype=np.int32).reshape(-1)
            tile_size_values = np.asarray(tile_npz["tile_size"], dtype=np.int32).reshape(-1)
            if tile_origin_values.shape[0] != 2 or tile_size_values.shape[0] != 2:
                raise ValueError(
                    f"Tile NPZ {tile_path} has invalid tile_origin/tile_size shapes."
                )
            tile_origin = (int(tile_origin_values[0]), int(tile_origin_values[1]))
            tile_size = (int(tile_size_values[0]), int(tile_size_values[1]))

            masks = np.asarray(tile_npz["masks"], dtype=bool)
            boxes = np.asarray(tile_npz["boxes"], dtype=np.float32)
            scores = np.asarray(tile_npz["scores"], dtype=np.float32).reshape(-1)

        if masks.ndim != 3:
            raise ValueError(
                f"Tile NPZ {tile_path} has invalid 'masks' shape; expected (N,H,W)."
            )

        expected_h = max(0, min(tile_size[1], image_height - tile_origin[1]))
        expected_w = max(0, min(tile_size[0], image_width - tile_origin[0]))
        if masks.shape[1] != expected_h or masks.shape[2] != expected_w:
            raise ValueError(
                f"Tile NPZ {tile_path} mask shape does not match tile extent."
            )

        count = int(masks.shape[0])
        if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != count:
            raise ValueError(
                f"Tile NPZ {tile_path} has invalid 'boxes' shape; expected (N,4)."
            )
        if scores.shape[0] != count:
            raise ValueError(
                f"Tile NPZ {tile_path} has invalid 'scores' length; expected {count}."
            )

        parsed_tiles.append((tile_path.stem, tile_origin, tile_size, masks, boxes, scores))

    parsed_tiles.sort(key=lambda item: (item[1][1], item[1][0], item[0]))

    tile_labels_by_tile: dict[tuple[int, int], list[_TileLabel]] = {}
    tile_extents: list[tuple[int, int, int, int]] = []
    next_label_id = 0

    for _, tile_origin, tile_size, masks, _, scores in parsed_tiles:
        tile_x0, tile_y0 = tile_origin
        tile_width, tile_height = tile_size
        tile_x1 = min(image_width, tile_x0 + tile_width)
        tile_y1 = min(image_height, tile_y0 + tile_height)
        if tile_x1 <= tile_x0 or tile_y1 <= tile_y0:
            continue

        tile_extents.append((tile_x0, tile_y0, tile_x1, tile_y1))
        labels: list[_TileLabel] = []
        for local_index in range(int(masks.shape[0])):
            tile_mask = np.asarray(masks[local_index], dtype=bool)
            labels.append(
                _TileLabel(
                    label_id=next_label_id,
                    score=float(scores[local_index]),
                    mask_area=int(tile_mask.sum()),
                    tile_origin=tile_origin,
                    tile_size=(tile_x1 - tile_x0, tile_y1 - tile_y0),
                    tile_mask=tile_mask,
                )
            )
            next_label_id += 1
        tile_labels_by_tile[tile_origin] = labels

    return _LoadedPredictions(
        source_mode="tile_npz_dir",
        merged_masks=None,
        merged_scores=None,
        merged_mask_areas=None,
        tile_labels_by_tile=tile_labels_by_tile,
        tile_extents=tile_extents,
        total_labels=next_label_id,
    )


def _point_in_any_extent(
    stem_x: int,
    stem_y: int,
    extents: list[tuple[int, int, int, int]],
) -> bool:
    """Return whether ``(x,y)`` is within any covered tile extent."""
    for x0, y0, x1, y1 in extents:
        if stem_x >= x0 and stem_x < x1 and stem_y >= y0 and stem_y < y1:
            return True
    return False


def _load_inventory(
    meta: GeoTiffMeta,
    inventory_csv: Path | None,
    inventory_shp: Path | None,
    x_field: str,
    y_field: str,
    tree_id_field: str,
    stem_id_field: str,
    species_field: str,
    status_field: str,
    status_filter: str,
    dbh_field: str,
    dbh_unit: DbhUnit,
    min_dbh_cm: float,
    max_dbh_cm: float,
    deduplicate_tree_id: bool,
) -> Inventory:
    """Load inventory rows from CSV or SHP using shared filtering semantics."""
    inventory = Inventory()

    if inventory_shp is not None:
        inventory.load_shp(
            shp_path=inventory_shp,
            status_field=status_field,
            status_filter=status_filter,
            dbh_field=dbh_field,
            dbh_unit=dbh_unit,
            min_dbh_cm=min_dbh_cm,
            max_dbh_cm=max_dbh_cm,
            tree_id_field=tree_id_field,
            stem_id_field=stem_id_field,
            species_field=species_field,
            deduplicate_tree_id=deduplicate_tree_id,
        )
        return inventory

    if inventory_csv is not None:
        csv_origin_lon, csv_origin_lat = _image_upper_left_wgs84(meta=meta)
        inventory.load_csv(
            csv_path=inventory_csv,
            x_origin=csv_origin_lon,
            y_origin=csv_origin_lat,
            status_field=status_field,
            status_filter=status_filter,
            x_field=x_field,
            y_field=y_field,
            dbh_field=dbh_field,
            dbh_unit=dbh_unit,
            min_dbh_cm=min_dbh_cm,
            max_dbh_cm=max_dbh_cm,
            tree_id_field=tree_id_field,
            stem_id_field=stem_id_field,
            species_field=species_field,
            deduplicate_tree_id=deduplicate_tree_id,
        )
        return inventory

    raise ValueError(
        "No inventory source provided. Provide either `inventory_csv` or `inventory_shp`."
    )


def _stem_id_for_tree(tree: Tree) -> str:
    """Resolve stem id with required fallback to tree id."""
    stem_id = (tree.stem_id or "").strip()
    if stem_id:
        return stem_id
    return tree.tree_id


def _wgs84_tree_to_pixel(
    lon: float,
    lat: float,
    meta: GeoTiffMeta,
    wgs84_to_image: osr.CoordinateTransformation | None,
    utm_fallback: tuple[int, bool] | None,
) -> tuple[float, float]:
    """Project WGS84 coordinates into full-image pixel coordinates."""
    if wgs84_to_image is not None:
        x_geo, y_geo, _ = wgs84_to_image.TransformPoint(lon, lat, 0.0)
    elif utm_fallback is not None:
        zone, northern_hemisphere = utm_fallback
        tree = Tree(
            tree_id="",
            species=None,
            status=None,
            x_wgs84=lon,
            y_wgs84=lat,
            dbh_cm=-1.0,
        )
        x_geo, y_geo = tree.pos_to_utm(
            utm_zone=zone, northern_hemisphere=northern_hemisphere
        )
    else:
        x_geo, y_geo = lon, lat

    pixel_x = (x_geo - meta.origin_x) / meta.pixel_size_x
    pixel_y = (y_geo - meta.origin_y) / meta.pixel_size_y
    return pixel_x, pixel_y


def _infer_utm_fallback_from_image_name(
    image_tif: Path,
) -> tuple[int, bool] | None:
    """Infer fallback UTM zone metadata from the image filename."""
    name = image_tif.stem.lower()
    match = re.search(r"utm(\d{1,2})([ns])", name)
    if match is None:
        return None

    zone = int(match.group(1))
    if zone < 1 or zone > 60:
        return None
    northern_hemisphere = match.group(2) == "n"
    return zone, northern_hemisphere


def _image_upper_left_wgs84(meta: GeoTiffMeta) -> tuple[float, float]:
    """Get image upper-left corner as WGS84 lon/lat."""
    if not meta.crs_wkt:
        return meta.origin_x, meta.origin_y

    image_crs = _spatial_reference_from_wkt(meta.crs_wkt)
    wgs84 = _wgs84_srs()
    if image_crs.IsSame(wgs84):
        return meta.origin_x, meta.origin_y

    to_wgs84 = osr.CoordinateTransformation(image_crs, wgs84)
    lon, lat, _ = to_wgs84.TransformPoint(meta.origin_x, meta.origin_y, 0.0)
    return lon, lat


def _build_wgs84_to_image_crs_transform(
    meta: GeoTiffMeta,
) -> osr.CoordinateTransformation | None:
    """Create transform from WGS84 into image CRS when available."""
    if not meta.crs_wkt:
        return None

    wgs84 = _wgs84_srs()
    image_crs = _spatial_reference_from_wkt(meta.crs_wkt)
    return osr.CoordinateTransformation(wgs84, image_crs)


def _wgs84_srs() -> osr.SpatialReference:
    """Create a WGS84 spatial reference in traditional GIS axis order."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _spatial_reference_from_wkt(wkt: str) -> osr.SpatialReference:
    """Build a spatial reference from WKT in traditional GIS axis order."""
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(wkt) != 0:
        raise ValueError("Could not parse GeoTIFF CRS WKT.")
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs
