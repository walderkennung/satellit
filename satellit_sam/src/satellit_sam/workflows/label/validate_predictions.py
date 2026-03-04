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


def validate_sam3_predictions(
    image_tif: Path,
    predictions_npz: Path,
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
        predictions_npz: Path to SAM3 ``image_masks.npz`` prediction artifact.
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
        ``tree_id, stem_id, label_id, tree_pos_x, tree_pos_y``.

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

    masks, scores, mask_areas = _load_predictions(predictions_npz=predictions_npz)
    mask_count, mask_height, mask_width = masks.shape

    if mask_width != meta.width or mask_height != meta.height:
        raise ValueError(
            "Prediction mask shape does not match image dimensions: "
            f"masks=({mask_width},{mask_height}) image=({meta.width},{meta.height})."
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
        rows.append(
            {
                "tree_id": tree.tree_id,
                "stem_id": _stem_id_for_tree(tree=tree),
                "label_id": None,
                "tree_pos_x": tree.x_wgs84,
                "tree_pos_y": tree.y_wgs84,
            }
        )

        if mask_count == 0:
            continue

        pixel_x, pixel_y = _wgs84_tree_to_pixel(
            lon=tree.x_wgs84,
            lat=tree.y_wgs84,
            meta=meta,
            wgs84_to_image=wgs84_to_image,
            utm_fallback=utm_fallback,
        )

        stem_x = int(round(pixel_x))
        stem_y = int(round(pixel_y))
        if stem_x < 0 or stem_y < 0 or stem_x >= mask_width or stem_y >= mask_height:
            continue

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

    if candidates:
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.mask_area,
                candidate.label_id,
                candidate.tree_index,
            )
        )

        used_labels: set[int] = set()
        assigned_trees: set[int] = set()
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
        columns=["tree_id", "stem_id", "label_id", "tree_pos_x", "tree_pos_y"],
    )
    result["label_id"] = pd.Series(
        [row["label_id"] for row in rows],
        dtype="object",
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    total = len(result)
    matched = int(result["label_id"].notna().sum())
    unmatched = total - matched
    match_rate = 0.0 if total == 0 else (matched / total) * 100.0

    print(f"Validation results saved to: {output_csv}")
    print("Validation summary:")
    print(f"- total trees: {total}")
    print(f"- matched trees: {matched}")
    print(f"- unmatched trees: {unmatched}")
    print(f"- match rate: {match_rate:.2f}%")

    return result


def _load_predictions(
    predictions_npz: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load masks and scores from SAM3 NPZ predictions.

    Args:
        predictions_npz: Path to ``image_masks.npz``.

    Returns:
        Tuple ``(masks, scores, mask_areas)``.

    Raises:
        ValueError: If required NPZ arrays are missing or malformed.
    """
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

    mask_areas = masks.reshape(label_count, -1).sum(axis=1).astype(np.int64)
    return masks, scores, mask_areas


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
