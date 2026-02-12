"""Tree data model and coordinate transforms used by SAM workflows."""

from dataclasses import dataclass

from pyproj import Proj, Transformer


@dataclass
class Tree:
    """Represents one inventory tree with geodetic coordinates and stem diameter.

    Attributes:
        tree_id: Inventory identifier for the tree.
        species: Species label when available.
        status: Tree status from the source inventory (for example, alive/dead).
        x_wgs84: Longitude in decimal degrees (EPSG:4326).
        y_wgs84: Latitude in decimal degrees (EPSG:4326).
        dbh_cm: Diameter at breast height in centimeters.
    """

    tree_id: str
    species: str | None
    status: str | None
    x_wgs84: float
    y_wgs84: float
    dbh_cm: float

    def pos_to_utm(
        self, utm_zone: int, northern_hemisphere: bool
    ) -> tuple[float, float]:
        """Project WGS84 coordinates to UTM easting/northing in meters.

        Arguments:
            utm_zone: UTM longitudinal zone number (1-60).
            northern_hemisphere: `True` for northern hemisphere zones, `False` for southern.

        Returns:
            Tuple `(x_utm, y_utm)` containing easting and northing in meters.
        """
        proj_utm = Proj(
            proj="utm", zone=utm_zone, ellps="WGS84", south=not northern_hemisphere
        )
        transformer = Transformer.from_proj(proj_from="epsg:4326", proj_to=proj_utm)
        return transformer.transform(self.y_wgs84, self.x_wgs84)
