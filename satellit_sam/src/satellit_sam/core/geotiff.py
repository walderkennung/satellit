"""GeoTIFF metadata loading utilities for geospatial workflows."""

from dataclasses import dataclass
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()


@dataclass
class GeoTiffMeta:
    """Metadata needed for mapping inventory points into raster pixel space."""

    width: int
    height: int
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float
    crs_wkt: str | None
    nodata_band_1: float | None

    @staticmethod
    def load_tif(tif_path: Path) -> "GeoTiffMeta":
        """Load GeoTIFF metadata from disk.

        Args:
            tif_path: Path to a GeoTIFF raster.

        Returns:
            Parsed geospatial metadata for the raster.

        Raises:
            FileNotFoundError: If the file cannot be opened by GDAL.
            ValueError: If the raster does not provide geotransform metadata.
        """
        dataset = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Could not open GeoTIFF: {tif_path}")

        try:
            geotransform = dataset.GetGeoTransform()
            if geotransform is None:
                raise ValueError(
                    f"GeoTIFF is missing geotransform metadata: {tif_path}"
                )

            crs_wkt = dataset.GetProjectionRef()
            if not crs_wkt or not crs_wkt.strip():
                crs_wkt = None

            nodata_band_1 = None
            band_1 = dataset.GetRasterBand(1)
            if band_1 is not None:
                nodata = band_1.GetNoDataValue()
                if nodata is not None:
                    nodata_band_1 = float(nodata)

            return GeoTiffMeta(
                width=dataset.RasterXSize,
                height=dataset.RasterYSize,
                origin_x=float(geotransform[0]),
                pixel_size_x=float(geotransform[1]),
                origin_y=float(geotransform[3]),
                pixel_size_y=float(geotransform[5]),
                crs_wkt=crs_wkt,
                nodata_band_1=nodata_band_1,
            )
        finally:
            dataset = None
