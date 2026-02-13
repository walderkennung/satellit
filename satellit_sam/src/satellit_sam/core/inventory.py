"""Inventory loaders for CSV and shapefile-based tree records."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Literal

from osgeo import gdal, ogr, osr
from pyproj import Geod

from satellit_sam.core.allometry import DbhUnit, to_dbh_cm
from satellit_sam.core.tree import Tree

gdal.UseExceptions()

InventoryCoordinates = Literal["auto", "local", "utm"]
_WGS84_GEOD = Geod(ellps="WGS84")


class Inventory:
    """In-memory tree inventory loaded from CSV or shapefile sources."""

    trees: list[Tree]

    def __init__(self):
        """Initialize an empty inventory."""
        self.trees = []

    def __len__(self) -> int:
        """Return the number of trees in the inventory."""
        return len(self.trees)

    def load_csv(
        self,
        csv_path: Path,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        status_field="",
        status_filter="",
        x_field="",
        y_field="",
        dbh_field="",
        dbh_unit: DbhUnit = "cm",
        min_dbh_cm: float = 0.0,
        max_dbh_cm: float = math.inf,
        tree_id_field="",
        species_field="",
        deduplicate_tree_id=True,
    ) -> None:
        """Load trees from CSV and map local offsets to WGS84 coordinates.

        Args:
            csv_path: Semicolon-delimited inventory file.
            x_origin: WGS84 longitude origin in decimal degrees.
            y_origin: WGS84 latitude origin in decimal degrees.
            status_field: Column name containing tree status text.
            status_filter: Optional case-insensitive status filter.
            x_field: Column name for local X offset in meters.
            y_field: Column name for local Y offset in meters.
            dbh_field: Column name for DBH values.
            dbh_unit: Unit used by ``dbh_field`` values.
            min_dbh_cm: Minimum DBH threshold in centimeters.
            max_dbh_cm: Maximum DBH threshold in centimeters (<=0 disables upper bound).
            tree_id_field: Column name for tree identifier values.
            species_field: Column name for species values.
            deduplicate_tree_id: Whether to deduplicate by tree id after loading.
        """
        self.trees = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                status = (row.get(status_field) or "").strip().strip("\r")
                if status_filter and status.lower() != status_filter.lower():
                    continue

                x_str = (row.get(x_field) or "").strip().replace(",", ".")
                y_str = (row.get(y_field) or "").strip().replace(",", ".")
                dbh_str = (row.get(dbh_field) or "").strip().replace(",", ".")
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

                tree_id = (row.get(tree_id_field) or "").strip()
                species = (row.get(species_field) or "").strip()
                dbh_cm = to_dbh_cm(dbh_raw=dbh_raw, dbh_unit=dbh_unit)
                if dbh_cm < min_dbh_cm:
                    continue
                if max_dbh_cm > 0 and dbh_cm > max_dbh_cm:
                    continue
                x_wgs84, y_wgs84 = _offset_wgs84(
                    x_origin=x_origin,
                    y_origin=y_origin,
                    local_x_m=local_x,
                    local_y_m=local_y,
                )
                self.trees.append(
                    Tree(
                        tree_id=tree_id,
                        species=species,
                        status=status,
                        x_wgs84=x_wgs84,
                        y_wgs84=y_wgs84,
                        dbh_cm=dbh_cm,
                    )
                )

        if deduplicate_tree_id:
            self.deduplicate_trees()

    def load_shp(
        self,
        shp_path: Path,
        status_field="",
        status_filter="",
        dbh_field="",
        dbh_unit: DbhUnit = "cm",
        min_dbh_cm: float = 0.0,
        max_dbh_cm: float = math.inf,
        tree_id_field="",
        species_field="",
        deduplicate_tree_id=True,
    ) -> None:
        """Load trees from a shapefile via GDAL/OGR.

        Geometry coordinates are transformed to WGS84 (EPSG:4326) and stored as
        tree longitude/latitude.

        Args:
            shp_path: Path to source shapefile.
            status_field: Field name containing tree status text.
            status_filter: Optional case-insensitive status filter.
            dbh_field: Field name for DBH values.
            dbh_unit: Unit used by ``dbh_field`` values.
            min_dbh_cm: Minimum DBH threshold in centimeters.
            max_dbh_cm: Maximum DBH threshold in centimeters (<=0 disables upper bound).
            tree_id_field: Field name for tree identifier values.
            species_field: Field name for species values.
            deduplicate_tree_id: Whether to deduplicate by tree id after loading.

        Raises:
            FileNotFoundError: If the shapefile cannot be opened.
            ValueError: If the first layer cannot be read.
        """
        previous_shx_restore = gdal.GetConfigOption("SHAPE_RESTORE_SHX")
        gdal.SetConfigOption("SHAPE_RESTORE_SHX", "YES")
        data_source = None
        try:
            data_source = ogr.Open(str(shp_path), 0)
            if data_source is None:
                raise FileNotFoundError(f"Could not open shapefile: {shp_path}")

            layer = data_source.GetLayer(0)
            if layer is None:
                raise ValueError(
                    f"Could not read first layer from shapefile: {shp_path}"
                )

            transform = _build_transform_to_wgs84(layer=layer)
            layer_defn = layer.GetLayerDefn()
            field_names = [
                layer_defn.GetFieldDefn(i).GetNameRef()
                for i in range(layer_defn.GetFieldCount())
            ]

            self.trees = []
            for idx, feature in enumerate(layer):
                coords = _feature_xy(feature=feature, transform=transform)
                if coords is None:
                    continue

                row = {
                    name: _field_value_as_text(feature.GetField(name))
                    for name in field_names
                }
                x, y = coords

                status = _get_first_present(row, [status_field, "status"]).strip()
                if not status:
                    status = "alive"
                status = status.strip("\r")
                if status_filter and status.lower() != status_filter.lower():
                    continue

                tree_id = _get_first_present(
                    row, [tree_id_field, "treeid", "tag", "stemtag"]
                ).strip()
                if not tree_id:
                    tree_id = f"shp_{idx}"
                species = _get_first_present(
                    row, [species_field, "species", "latin"]
                ).strip()

                dbh_cm = -1.0
                dbh_raw = -1.0
                dbh_text = _get_first_present(row, [dbh_field, "dbh"]).strip()
                if dbh_text:
                    try:
                        dbh_raw = float(dbh_text.replace(",", "."))
                        if dbh_raw > 0:
                            dbh_cm = to_dbh_cm(dbh_raw=dbh_raw, dbh_unit=dbh_unit)
                    except ValueError:
                        pass

                # If DBH filtering is enabled, require valid DBH and apply thresholds.
                if min_dbh_cm > 0 or max_dbh_cm > 0:
                    if dbh_cm <= 0:
                        continue
                    if dbh_cm < min_dbh_cm:
                        continue
                    if max_dbh_cm > 0 and dbh_cm > max_dbh_cm:
                        continue

                self.trees.append(
                    Tree(
                        tree_id=tree_id,
                        species=species,
                        status=status,
                        x_wgs84=x,
                        y_wgs84=y,
                        dbh_cm=dbh_cm,
                    )
                )
        finally:
            data_source = None
            gdal.SetConfigOption("SHAPE_RESTORE_SHX", previous_shx_restore)

        if deduplicate_tree_id:
            self.deduplicate_trees()

    def deduplicate_trees(self) -> None:
        """Keep one row per tree id, preferring the highest DBH record.

        Rows without a tree id are retained unchanged.
        """
        by_id: dict[str, Tree] = {}
        no_id_rows: list[Tree] = []
        for tree in self.trees:
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
        deduped.sort(key=lambda t: (t.tree_id, t.x_wgs84, t.y_wgs84))
        self.trees = deduped


def _build_transform_to_wgs84(
    layer: ogr.Layer,
) -> osr.CoordinateTransformation | None:
    """Build a layer-to-WGS84 (EPSG:4326) transform when needed."""
    source_srs = layer.GetSpatialRef()
    if source_srs is None:
        return None

    source = source_srs.Clone()
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    if hasattr(source, "SetAxisMappingStrategy"):
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if source.IsSame(target):
        return None

    return osr.CoordinateTransformation(source, target)


def _offset_wgs84(
    x_origin: float,
    y_origin: float,
    local_x_m: float,
    local_y_m: float,
) -> tuple[float, float]:
    """Convert local meter offsets from a WGS84 origin to lon/lat."""
    lon = x_origin
    lat = y_origin

    if local_x_m != 0.0:
        x_azimuth = 90.0 if local_x_m > 0.0 else 270.0
        lon, lat, _ = _WGS84_GEOD.fwd(lon, lat, x_azimuth, abs(local_x_m))

    if local_y_m != 0.0:
        y_azimuth = 0.0 if local_y_m > 0.0 else 180.0
        lon, lat, _ = _WGS84_GEOD.fwd(lon, lat, y_azimuth, abs(local_y_m))

    return lon, lat


def _row_value_case_insensitive(row: dict[str, str], key: str) -> str:
    """Return a row value for `key` with case-insensitive fallback."""
    if key in row and row[key] is not None:
        return row[key]
    lowered = key.lower()
    for candidate, value in row.items():
        if candidate.lower() == lowered and value is not None:
            return value
    return ""


def _get_first_present(row: dict[str, str], keys: list[str]) -> str:
    """Return the first non-empty value found for any candidate key."""
    for key in keys:
        value = _row_value_case_insensitive(row=row, key=key)
        if value != "":
            return value
    return ""


def _field_value_as_text(value: object | None) -> str:
    """Normalize OGR field values to text for downstream row-style parsing."""
    if value is None:
        return ""
    return str(value)


def _feature_xy(
    feature: ogr.Feature, transform: osr.CoordinateTransformation | None
) -> tuple[float, float] | None:
    """Extract representative XY coordinates from a feature geometry."""
    geometry = feature.GetGeometryRef()
    if geometry is None:
        return None

    working_geom = geometry.Clone()
    if transform is not None:
        if working_geom.Transform(transform) != 0:
            return None

    flattened_type = ogr.GT_Flatten(working_geom.GetGeometryType())
    if flattened_type == ogr.wkbPoint:
        return working_geom.GetX(), working_geom.GetY()

    if flattened_type == ogr.wkbMultiPoint and working_geom.GetGeometryCount() > 0:
        point = working_geom.GetGeometryRef(0)
        if point is not None:
            return point.GetX(), point.GetY()

    centroid = working_geom.Centroid()
    if centroid is None:
        return None
    return centroid.GetX(), centroid.GetY()
