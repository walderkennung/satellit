"""Generate weak tree labels per tile from forest inventory data.

The resulting YAML file has the form:
```yaml
tiles:
  - tile_id: "tile_x0_y0"
    trees:
      - x_pixel: int
        y_pixel: int
        crown_radius: float
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from osgeo import osr

from src.satellit_sam.core.allometry import (
    CrownModel,
    DbhUnit,
    compute_crown_radius_m,
)
from src.satellit_sam.core.geotiff import GeoTiffMeta
from src.satellit_sam.core.inventory import Inventory


def make_weak_labels(
    output_dir: Path | argparse.Namespace,
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
    only_non_empty_tiles: bool = False,
) -> None:
    """Generate tile-wise weak labels and write them to ``labels_tiles.yaml``.

    The function accepts either explicit keyword arguments or an argparse
    namespace (used by the CLI workflow).
    """
    if isinstance(output_dir, argparse.Namespace):
        args = output_dir
        run(
            output_dir=Path(args.output_dir),
            image_tif=Path(args.image_tif),
            tile_size=int(getattr(args, "tile_size", tile_size)),
            tile_overlap=int(getattr(args, "overlap", tile_overlap)),
            min_dbh_cm=float(getattr(args, "min_dbh_cm", min_dbh_cm)),
            max_dbh_cm=float(getattr(args, "max_dbh_cm", max_dbh_cm)),
            crown_model=getattr(args, "crown_model", crown_model),
            export_visualizations=bool(
                getattr(args, "export_visualizations", export_visualizations)
            ),
            inventory_csv=getattr(args, "inventory_csv", inventory_csv),
            inventory_shp=getattr(args, "inventory_shp", inventory_shp),
            x_field=str(getattr(args, "x_field", x_field)),
            y_field=str(getattr(args, "y_field", y_field)),
            tree_id_field=str(getattr(args, "tree_id_field", tree_id_field)),
            species_field=str(getattr(args, "species_field", species_field)),
            status_field=str(getattr(args, "status_field", status_field)),
            status_filter=str(getattr(args, "status_filter", status_filter)),
            dbh_field=str(getattr(args, "dbh_field", dbh_field)),
            dbh_unit=getattr(args, "dbh_unit", dbh_unit),
            deduplicate_tree_id=bool(
                getattr(args, "deduplicate_tree_id", deduplicate_tree_id)
            ),
            default_crown_radius_m=float(
                getattr(args, "default_crown_radius_m", default_crown_radius_m)
            ),
            linear_factor_m_per_cm=float(
                getattr(args, "linear_factor_m_per_cm", linear_factor_m_per_cm)
            ),
            linear_intercept_m=float(
                getattr(args, "linear_intercept_m", linear_intercept_m)
            ),
            power_a=float(getattr(args, "power_a", power_a)),
            power_b=float(getattr(args, "power_b", power_b)),
            min_crown_radius_m=float(
                getattr(args, "min_crown_radius_m", min_crown_radius_m)
            ),
            max_crown_radius_m=float(
                getattr(args, "max_crown_radius_m", max_crown_radius_m)
            ),
            only_non_empty_tiles=bool(
                getattr(args, "only_non_empty_tiles", only_non_empty_tiles)
            ),
        )
        return

    if image_tif is None:
        raise ValueError("`image_tif` must be provided.")
    if inventory_csv is not None and inventory_shp is not None:
        raise ValueError("Provide only one of `inventory_csv` or `inventory_shp`.")
    if inventory_csv is None and inventory_shp is None:
        raise ValueError("Provide either `inventory_csv` or `inventory_shp`.")
    if tile_size <= 0:
        raise ValueError("`tile_size` must be > 0.")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("`tile_overlap` must be >= 0 and < `tile_size`.")

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
    else:
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

    wgs84_to_image = _build_wgs84_to_image_crs_transform(meta=meta)
    tiles = _build_tiles(
        width=meta.width,
        height=meta.height,
        tile_size=tile_size,
        overlap=tile_overlap,
    )
    meter_to_px = _meter_to_pixel_scale(meta=meta)

    for tree in inventory.trees:
        image_x, image_y = _wgs84_tree_to_pixel(
            lon=tree.x_wgs84,
            lat=tree.y_wgs84,
            meta=meta,
            wgs84_to_image=wgs84_to_image,
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

            tile["trees"].append(
                {
                    "x_pixel": int(round(image_x - tile["x0"])),
                    "y_pixel": int(round(image_y - tile["y0"])),
                    "crown_radius": round(crown_radius_px, 3),
                }
            )

    tiles_payload: list[dict[str, Any]] = []
    for tile in tiles:
        if only_non_empty_tiles and not tile["trees"]:
            continue
        tiles_payload.append(
            {
                "tile_id": tile["tile_id"],
                "trees": tile["trees"],
            }
        )

    yaml_path = output_dir / "labels_tiles.yaml"
    write_yaml(tiles=tiles_payload, yaml_path=yaml_path)

    if export_visualizations:
        print("`export_visualizations` is not implemented in this workflow yet.")
    print(f"Weak labels written: {yaml_path}")


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
) -> tuple[float, float]:
    if wgs84_to_image is not None:
        x_geo, y_geo, _ = wgs84_to_image.TransformPoint(lon, lat, 0.0)
    else:
        x_geo, y_geo = lon, lat

    pixel_x = (x_geo - meta.origin_x) / meta.pixel_size_x
    pixel_y = (y_geo - meta.origin_y) / meta.pixel_size_y
    return pixel_x, pixel_y


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
    if wgs84.IsSame(image_crs):
        return None
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


def write_yaml(tiles: list[dict[str, Any]], yaml_path: Path) -> None:
    """Write weak labels to YAML in the documented structure."""
    lines: list[str] = ["tiles:"]
    for tile in tiles:
        tile_id = str(tile["tile_id"])
        trees = tile["trees"]
        lines.append(f'  - tile_id: "{tile_id}"')
        if not trees:
            lines.append("    trees: []")
            continue

        lines.append("    trees:")
        for tree in trees:
            lines.append(f"      - x_pixel: {int(tree['x_pixel'])}")
            lines.append(f"        y_pixel: {int(tree['y_pixel'])}")
            lines.append(f"        crown_radius: {float(tree['crown_radius']):.3f}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
