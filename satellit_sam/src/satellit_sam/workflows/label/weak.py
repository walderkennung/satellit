"""Generate weak tree labels per tile from forest inventory data.

The resulting CSV file has the form:
```csv
tile_id,tree_id,x_pixel,y_pixel,crown_radius,bbox_x1,bbox_y1,bbox_x2,bbox_y2,x_long_wgs84,y_lat_wgs84,dbh_cm
```
The same rows are also exported as a point shapefile (`labels_tiles.shp`).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from osgeo import gdal, ogr, osr

from satellit_sam.core.allometry import (
    CrownModel,
    DbhUnit,
    compute_crown_radius_m,
)
from satellit_sam.core.geotiff import GeoTiffMeta
from satellit_sam.core.inventory import Inventory
from satellit_sam.core.tree import Tree


def make_weak_labels(
    output_dir: Path,
    image_tif: Path | None = None,
    tile_size: int = 1024,
    tile_overlap: int = 128,
    min_dbh_cm: float = 0.0,
    max_dbh_cm: float = 0.0,
    crown_model: CrownModel = "linear",
    export_visualizations: bool = False,
    inventory_csv: Path | None = None,
    inventory_shp: Path | None = None,
    x_field: str = "PX",
    y_field: str = "PY",
    tree_id_field: str = "TreeID",
    species_field: str = "Latin",
    status_field: str = "Status",
    status_filter: str = "alive",
    dbh_field: str = "DBH",
    dbh_unit: DbhUnit = "mm",
    deduplicate_tree_id: bool = True,
    default_crown_radius_m: float = 2.5,
    linear_factor_m_per_cm: float = 0.08,
    linear_intercept_m: float = 0.0,
    power_a: float = 0.15,
    power_b: float = 0.8,
    min_crown_radius_m: float = 0.5,
    max_crown_radius_m: float = 15.0,
    bbox_padding_px: float = 4.0,
) -> None:
    """Generate tile-wise weak labels and write them to ``labels_tiles.csv``.

    The function is intended to be configured via its explicit keyword
    arguments, e.g. when called from a CLI wrapper or other Python code.
    """
    if image_tif is None:
        raise ValueError("`image_tif` must be provided.")
    if inventory_csv is not None and inventory_shp is not None:
        raise ValueError("Provide only one of `inventory_csv` or `inventory_shp`.")

    if inventory_csv is None and inventory_shp is None:
        raise ValueError(
            "No inventory source provided. -- Provide either `inventory_csv` or `inventory_shp`."
        )

    if tile_size <= 0:
        raise ValueError("`tile_size` must be > 0.")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("`tile_overlap` must be >= 0 and < `tile_size`.")
    if bbox_padding_px < 0.0:
        raise ValueError("`bbox_padding_px` must be >= 0.")

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = GeoTiffMeta.load_tif(image_tif)

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
            species_field=species_field,
            deduplicate_tree_id=deduplicate_tree_id,
        )
    elif inventory_csv is not None:
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
            species_field=species_field,
            deduplicate_tree_id=deduplicate_tree_id,
        )
    else:
        assert False, (
            "Unreachable code: inventory source check should have raised an error."
        )

    wgs84_to_image = _build_wgs84_to_image_crs_transform(meta=meta)
    utm_fallback = _infer_utm_fallback_from_image_name(image_tif=image_tif)
    if meta.crs_wkt is None and wgs84_to_image is None and utm_fallback is None:
        raise ValueError(
            "GeoTIFF CRS is missing. Could not infer fallback UTM zone from image "
            "filename. Add CRS metadata to the image (recommended) or include "
            "'utm<zone><n|s>' in the filename (e.g. '...utm33n...')."
        )
    tiles = _build_tiles(
        width=meta.width,
        height=meta.height,
        tile_size=tile_size,
        overlap=tile_overlap,
    )
    meter_to_px = _meter_to_pixel_scale(meta=meta)
    projected_trees: list[dict[str, float]] = []

    for tree in inventory.trees:
        image_x, image_y = _wgs84_tree_to_pixel(
            lon=tree.x_wgs84,
            lat=tree.y_wgs84,
            meta=meta,
            wgs84_to_image=wgs84_to_image,
            utm_fallback=utm_fallback,
        )

        if image_x < 0.0 or image_y < 0.0:
            continue
        if image_x >= meta.width or image_y >= meta.height:
            continue

        if tree.dbh_cm > 0.0:
            crown_radius_m = compute_crown_radius_m(
                dbh_cm=tree.dbh_cm,
                crown_model=crown_model,
                linear_factor_m_per_cm=linear_factor_m_per_cm,
                linear_intercept_m=linear_intercept_m,
                power_a=power_a,
                power_b=power_b,
                min_crown_radius_m=min_crown_radius_m,
                max_crown_radius_m=max_crown_radius_m,
            )
        else:
            crown_radius_m = max(default_crown_radius_m, min_crown_radius_m)
        crown_radius_px = crown_radius_m * meter_to_px
        projected_trees.append(
            {
                "x_global": image_x,
                "y_global": image_y,
                "crown_radius_px": crown_radius_px,
            }
        )

        for tile in tiles:
            if not _circle_intersects_rect(
                center_x=image_x,
                center_y=image_y,
                radius=crown_radius_px,
                x0=tile["x0"],
                y0=tile["y0"],
                x1=tile["x1"],
                y1=tile["y1"],
            ):
                continue

            tile_center_x = image_x - tile["x0"]
            tile_center_y = image_y - tile["y0"]
            tile_width = tile["x1"] - tile["x0"]
            tile_height = tile["y1"] - tile["y0"]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _crown_bbox_in_tile(
                center_x=tile_center_x,
                center_y=tile_center_y,
                crown_radius_px=crown_radius_px,
                tile_width=tile_width,
                tile_height=tile_height,
                padding_px=bbox_padding_px,
            )

            tile["trees"].append(
                {
                    "tree_id": tree.tree_id,
                    "x_pixel": int(round(tile_center_x)),
                    "y_pixel": int(round(tile_center_y)),
                    "crown_radius": round(crown_radius_px, 3),
                    "bbox_x1": round(bbox_x1, 3),
                    "bbox_y1": round(bbox_y1, 3),
                    "bbox_x2": round(bbox_x2, 3),
                    "bbox_y2": round(bbox_y2, 3),
                    "x_long_wgs84": round(tree.x_wgs84, 9),
                    "y_lat_wgs84": round(tree.y_wgs84, 9),
                    "dbh_cm": round(tree.dbh_cm, 3),
                }
            )

    tiles_payload: list[dict[str, Any]] = []
    for tile in tiles:
        if not tile["trees"]:
            continue
        tiles_payload.append(
            {
                "tile_id": tile["tile_id"],
                "trees": tile["trees"],
            }
        )

    csv_path = output_dir / "labels_tiles.csv"
    write_csv(tiles=tiles_payload, csv_path=csv_path)
    shp_path = output_dir / "labels_tiles.shp"
    write_shapefile(tiles=tiles_payload, shp_path=shp_path)
    legacy_yaml_path = output_dir / "labels_tiles.yaml"
    if legacy_yaml_path.exists():
        legacy_yaml_path.unlink()

    visualization_outputs: dict[str, str] = {}
    if export_visualizations:
        visualization_tiles = [tile for tile in tiles if (tile["trees"])]
        if not only_non_empty_tiles:
            visualization_tiles = tiles
        visualization_outputs = export_visualizations_opencv(
            image_tif=image_tif,
            output_dir=output_dir,
            projected_trees=projected_trees,
            tiles=visualization_tiles,
        )
    print(f"Weak labels written: {csv_path}")
    print(f"Weak labels written: {shp_path}")
    if visualization_outputs:
        print(
            f"Weak labeling visualization (crowns): {visualization_outputs['visualization_crowns']}"
        )
        print(
            "Weak labeling visualization (crowns+tiles): "
            + visualization_outputs["visualization_crowns_tiles"]
        )


def _build_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[dict]:
    step = tile_size - overlap
    tiles: list[dict[str, Any]] = []
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
                    "trees": [],
                }
            )
    return tiles


def _circle_intersects_rect(
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


def _crown_bbox_in_tile(
    center_x: float,
    center_y: float,
    crown_radius_px: float,
    tile_width: int,
    tile_height: int,
    padding_px: float,
) -> tuple[float, float, float, float]:
    padded_radius = max(0.0, crown_radius_px + padding_px)
    x1 = max(0.0, center_x - padded_radius)
    y1 = max(0.0, center_y - padded_radius)
    x2 = min(float(tile_width), center_x + padded_radius)
    y2 = min(float(tile_height), center_y + padded_radius)

    # Keep SAM prompts valid even for edge-touching labels.
    if x2 <= x1:
        x_center = min(max(center_x, 0.0), float(tile_width))
        x1 = max(0.0, x_center - 0.5)
        x2 = min(float(tile_width), x_center + 0.5)
    if y2 <= y1:
        y_center = min(max(center_y, 0.0), float(tile_height))
        y1 = max(0.0, y_center - 0.5)
        y2 = min(float(tile_height), y_center + 0.5)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Could not derive a valid bbox for weak label.")
    return x1, y1, x2, y2


def _meter_to_pixel_scale(meta: GeoTiffMeta) -> float:
    px_size_x = abs(meta.pixel_size_x)
    px_size_y = abs(meta.pixel_size_y)
    mean_pixel_size = (px_size_x + px_size_y) / 2.0
    if mean_pixel_size <= 0.0:
        raise ValueError("GeoTIFF has invalid pixel size for crown radius conversion.")
    return 1.0 / mean_pixel_size


def _wgs84_tree_to_pixel(
    lon: float,
    lat: float,
    meta: GeoTiffMeta,
    wgs84_to_image: osr.CoordinateTransformation | None,
    utm_fallback: tuple[int, bool] | None,
) -> tuple[float, float]:
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
    if not meta.crs_wkt:
        return None

    wgs84 = _wgs84_srs()
    image_crs = _spatial_reference_from_wkt(meta.crs_wkt)
    return osr.CoordinateTransformation(wgs84, image_crs)


def _wgs84_srs() -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _spatial_reference_from_wkt(wkt: str) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(wkt) != 0:
        raise ValueError("Could not parse GeoTIFF CRS WKT.")
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def write_csv(tiles: list[dict[str, Any]], csv_path: Path) -> None:
    """Write weak labels as flat CSV rows."""
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "tile_id",
                "tree_id",
                "x_pixel",
                "y_pixel",
                "crown_radius",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "x_long_wgs84",
                "y_lat_wgs84",
                "dbh_cm",
            ]
        )

        for row in _iter_label_rows(tiles=tiles):
            writer.writerow(
                [
                    row["tile_id"],
                    row["tree_id"],
                    row["x_pixel"],
                    row["y_pixel"],
                    row["crown_radius"],
                    row["bbox_x1"],
                    row["bbox_y1"],
                    row["bbox_x2"],
                    row["bbox_y2"],
                    row["x_long_wgs84"],
                    row["y_lat_wgs84"],
                    row["dbh_cm"],
                ]
            )


def write_shapefile(tiles: list[dict[str, Any]], shp_path: Path) -> None:
    """Write weak labels as WGS84 point shapefile."""
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("GDAL ESRI Shapefile driver is not available.")

    if shp_path.exists():
        driver.DeleteDataSource(str(shp_path))

    datasource = driver.CreateDataSource(str(shp_path))
    if datasource is None:
        raise RuntimeError(f"Could not create shapefile: {shp_path}")
    try:
        srs = _wgs84_srs()
        layer = datasource.CreateLayer("labels_tiles", srs=srs, geom_type=ogr.wkbPoint)
        if layer is None:
            raise RuntimeError(f"Could not create layer for shapefile: {shp_path}")

        _create_shapefile_fields(layer=layer)

        definition = layer.GetLayerDefn()
        for row in _iter_label_rows(tiles=tiles):
            feature = ogr.Feature(definition)
            feature.SetField("tile_id", row["tile_id"])
            feature.SetField("tree_id", row["tree_id"])
            feature.SetField("x_pixel", row["x_pixel"])
            feature.SetField("y_pixel", row["y_pixel"])
            feature.SetField("crown_px", row["crown_radius"])
            feature.SetField("bbox_x1", row["bbox_x1"])
            feature.SetField("bbox_y1", row["bbox_y1"])
            feature.SetField("bbox_x2", row["bbox_x2"])
            feature.SetField("bbox_y2", row["bbox_y2"])
            feature.SetField("x_long", row["x_long_wgs84"])
            feature.SetField("y_lat", row["y_lat_wgs84"])
            feature.SetField("dbh_cm", row["dbh_cm"])

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(row["x_long_wgs84"], row["y_lat_wgs84"])
            feature.SetGeometry(point)

            if layer.CreateFeature(feature) != 0:
                raise RuntimeError("Failed to write feature into shapefile.")
            feature = None
    finally:
        datasource = None


def _create_shapefile_fields(layer: ogr.Layer) -> None:
    field_specs = [
        ("tile_id", ogr.OFTString, 48, 0),
        ("tree_id", ogr.OFTString, 48, 0),
        ("x_pixel", ogr.OFTInteger, 0, 0),
        ("y_pixel", ogr.OFTInteger, 0, 0),
        ("crown_px", ogr.OFTReal, 12, 3),
        ("bbox_x1", ogr.OFTReal, 12, 3),
        ("bbox_y1", ogr.OFTReal, 12, 3),
        ("bbox_x2", ogr.OFTReal, 12, 3),
        ("bbox_y2", ogr.OFTReal, 12, 3),
        ("x_long", ogr.OFTReal, 18, 9),
        ("y_lat", ogr.OFTReal, 18, 9),
        ("dbh_cm", ogr.OFTReal, 10, 3),
    ]
    for name, field_type, width, precision in field_specs:
        definition = ogr.FieldDefn(name, field_type)
        if width > 0:
            definition.SetWidth(width)
        if precision > 0:
            definition.SetPrecision(precision)
        if layer.CreateField(definition) != 0:
            raise RuntimeError(f"Could not create shapefile field: {name}")


def _iter_label_rows(tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tile in tiles:
        tile_id = str(tile["tile_id"])
        for tree in tile["trees"]:
            rows.append(
                {
                    "tile_id": tile_id,
                    "tree_id": str(tree["tree_id"]),
                    "x_pixel": int(tree["x_pixel"]),
                    "y_pixel": int(tree["y_pixel"]),
                    "crown_radius": round(float(tree["crown_radius"]), 3),
                    "bbox_x1": round(float(tree["bbox_x1"]), 3),
                    "bbox_y1": round(float(tree["bbox_y1"]), 3),
                    "bbox_x2": round(float(tree["bbox_x2"]), 3),
                    "bbox_y2": round(float(tree["bbox_y2"]), 3),
                    "x_long_wgs84": float(tree["x_long_wgs84"]),
                    "y_lat_wgs84": float(tree["y_lat_wgs84"]),
                    "dbh_cm": float(tree["dbh_cm"]),
                }
            )
    return rows


def export_visualizations_opencv(
    image_tif: Path,
    output_dir: Path,
    projected_trees: list[dict[str, float]],
    tiles: list[dict[str, Any]],
    max_dimension: int = 2400,
    crown_stroke_width: int = 1,
    center_radius: int = 2,
) -> dict[str, str]:
    """Render weak-label visualization overlays using OpenCV."""
    base_bgr = _load_tif_for_visualization(image_tif=image_tif)
    original_height, original_width = base_bgr.shape[:2]
    scale = 1.0
    if max(original_width, original_height) > max_dimension:
        scale = float(max_dimension) / float(max(original_width, original_height))

    if scale < 1.0:
        preview = cv2.resize(
            base_bgr,
            dsize=(
                int(round(original_width * scale)),
                int(round(original_height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = base_bgr

    crowns = preview.copy()
    crowns_tiles = preview.copy()

    for tree in projected_trees:
        x = int(round(tree["x_global"] * scale))
        y = int(round(tree["y_global"] * scale))
        radius = max(1, int(round(tree["crown_radius_px"] * scale)))
        cv2.circle(
            crowns,
            center=(x, y),
            radius=radius,
            color=(0, 0, 255),
            thickness=max(1, crown_stroke_width),
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            crowns,
            center=(x, y),
            radius=max(1, center_radius),
            color=(0, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            crowns_tiles,
            center=(x, y),
            radius=radius,
            color=(0, 0, 255),
            thickness=max(1, crown_stroke_width),
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            crowns_tiles,
            center=(x, y),
            radius=max(1, center_radius),
            color=(0, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    for tile in tiles:
        x0 = int(round(float(tile["x0"]) * scale))
        y0 = int(round(float(tile["y0"]) * scale))
        x1 = int(round(float(tile["x1"]) * scale))
        y1 = int(round(float(tile["y1"]) * scale))
        cv2.rectangle(
            crowns_tiles,
            pt1=(x0, y0),
            pt2=(x1, y1),
            color=(255, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    base_path = viz_dir / "base.tif"
    crowns_path = viz_dir / "labels_crowns.tif"
    crowns_tiles_path = viz_dir / "labels_crowns_tiles.tif"
    for legacy_png in (
        viz_dir / "base.png",
        viz_dir / "labels_crowns.png",
        viz_dir / "labels_crowns_tiles.png",
    ):
        if legacy_png.exists():
            legacy_png.unlink()

    _write_lossless_tiff(path=base_path, image=preview)
    _write_lossless_tiff(path=crowns_path, image=crowns)
    _write_lossless_tiff(path=crowns_tiles_path, image=crowns_tiles)

    return {
        "visualization_base": str(base_path),
        "visualization_crowns": str(crowns_path),
        "visualization_crowns_tiles": str(crowns_tiles_path),
    }


def _load_tif_for_visualization(image_tif: Path) -> np.ndarray:
    """Load a GeoTIFF into an 8-bit BGR image for OpenCV rendering."""
    dataset = gdal.Open(str(image_tif), gdal.GA_ReadOnly)
    if dataset is not None:
        try:
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            band_count = dataset.RasterCount
            if band_count <= 0:
                raise ValueError(f"GeoTIFF has no raster bands: {image_tif}")

            if band_count >= 3:
                red = dataset.GetRasterBand(1).ReadAsArray(0, 0, width, height)
                green = dataset.GetRasterBand(2).ReadAsArray(0, 0, width, height)
                blue = dataset.GetRasterBand(3).ReadAsArray(0, 0, width, height)
                rgb = np.dstack([red, green, blue])
            else:
                gray = dataset.GetRasterBand(1).ReadAsArray(0, 0, width, height)
                rgb = np.dstack([gray, gray, gray])
        finally:
            dataset = None

        rgb_uint8 = _to_uint8_image(image=rgb)
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    image = cv2.imread(str(image_tif), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    raise FileNotFoundError(f"Could not open image for visualization: {image_tif}")


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric image data to uint8 using min-max normalization."""
    if image.dtype == np.uint8:
        return image

    data = image.astype(np.float32)
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros_like(data, dtype=np.uint8)

    min_value = float(np.min(data[finite]))
    max_value = float(np.max(data[finite]))
    if max_value <= min_value:
        return np.zeros_like(data, dtype=np.uint8)

    normalized = (data - min_value) / (max_value - min_value)
    normalized = np.clip(normalized * 255.0, 0.0, 255.0)
    return normalized.astype(np.uint8)


def _write_lossless_tiff(path: Path, image: np.ndarray) -> None:
    """Write image as lossless TIFF (LZW when supported)."""
    compression_flag = getattr(cv2, "IMWRITE_TIFF_COMPRESSION", None)
    if compression_flag is None:
        ok = cv2.imwrite(str(path), image)
    else:
        # 5 = LZW compression for TIFF (lossless).
        ok = cv2.imwrite(str(path), image, [compression_flag, 5])
    if not ok:
        raise RuntimeError(f"Failed to write visualization image: {path}")
