"""Workflow helpers for image tiling."""

import os
from pathlib import Path

import numpy as np
from osgeo import gdal

from satellit_sam.core import Image, TilesDir, tile_image
from satellit_sam.core.tiling import iter_tile_geometries

gdal.UseExceptions()

_GEOTIFF_EXTENSIONS = {".tif", ".tiff"}


def make_image_tiles(
    image_path: Path,
    output_path: Path,
    tile_width: int,
    tile_height: int,
    overlap: int,
) -> TilesDir:
    """Load an image from disk and save tiled outputs.

    GeoTIFF inputs are tiled via GDAL window reads so only one tile-sized
    raster portion is loaded in memory at a time.

    Args:
        image_path: Path to the source image.
        output_path: Directory where tile outputs are written.
        tile_width: Tile width in pixels.
        tile_height: Tile height in pixels.
        overlap: Overlap between neighboring tiles in pixels.

    Returns:
        Tile directory descriptor for generated outputs.

    Raises:
        ValueError: If tile dimensions or overlap values are invalid.
    """
    if tile_width <= 0:
        raise ValueError("`tile_width` must be > 0.")
    if tile_height <= 0:
        raise ValueError("`tile_height` must be > 0.")
    if overlap < 0:
        raise ValueError("`overlap` must be >= 0.")
    if overlap >= tile_width or overlap >= tile_height:
        raise ValueError("`overlap` must be smaller than both tile dimensions.")

    output_path.mkdir(parents=True, exist_ok=True)
    tile_size = (tile_width, tile_height)

    if image_path.suffix.lower() in _GEOTIFF_EXTENSIONS:
        return _make_geotiff_tiles(
            image_path=image_path,
            output_path=output_path,
            tile_size=tile_size,
            overlap=overlap,
        )

    image = Image.load(str(image_path))
    tiling_dir = tile_image(
        image=image,
        tile_size=tile_size,
        overlap=overlap,
        output_path=str(output_path),
    )
    tiling_dir.save_to_dir()
    return tiling_dir


def _make_geotiff_tiles(
    image_path: Path,
    output_path: Path,
    tile_size: tuple[int, int],
    overlap: int,
) -> TilesDir:
    """Tile a GeoTIFF by streaming one GDAL window at a time."""
    dataset = gdal.Open(str(image_path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Could not open GeoTIFF: {image_path}")

    try:
        width = int(dataset.RasterXSize)
        height = int(dataset.RasterYSize)
        band_count = int(dataset.RasterCount)
        channels = _select_channel_count(band_count)

        tiling_dir = TilesDir(
            output_path=str(output_path),
            tile_size=tile_size,
            overlap=(overlap, overlap),
            original_shape=(height, width, channels),
        )
        tiles_rgb_path = tiling_dir.tiles_rgb_path()
        os.makedirs(tiles_rgb_path, exist_ok=True)

        for tile_geo in iter_tile_geometries(
            image_shape=(width, height),
            tile_size=tile_size,
            overlap=(overlap, overlap),
        ):
            x_start, y_start = tile_geo.start
            x_end, y_end = tile_geo.end
            tile_width = x_end - x_start
            tile_height = y_end - y_start

            tile_data = _read_geotiff_tile(
                dataset=dataset,
                x_start=x_start,
                y_start=y_start,
                width=tile_width,
                height=tile_height,
                channels=channels,
            )

            tile = Image(
                size=(tile_width, tile_height),
                channels=channels,
                data=tile_data,
            )
            tile_filename = f"tile_x{x_start}_y{y_start}.png"
            tile.save(os.path.join(tiles_rgb_path, tile_filename))

        tiling_dir.save_to_dir()
        return tiling_dir
    finally:
        dataset = None


def _select_channel_count(band_count: int) -> int:
    """Map raster band count to output image channels."""
    if band_count <= 0:
        raise ValueError("GeoTIFF has no raster bands.")
    if band_count == 1:
        return 1
    if band_count >= 4:
        return 4
    return 3


def _read_geotiff_tile(
    dataset: gdal.Dataset,
    x_start: int,
    y_start: int,
    width: int,
    height: int,
    channels: int,
) -> np.ndarray:
    """Read one tile window from a GeoTIFF dataset."""
    if channels == 1:
        band = dataset.GetRasterBand(1)
        if band is None:
            raise ValueError("GeoTIFF is missing raster band 1.")
        tile_data = band.ReadAsArray(x_start, y_start, width, height)
        if tile_data is None:
            raise ValueError("Failed to read GeoTIFF tile window.")
        return tile_data

    bands = []
    for band_idx in range(1, channels + 1):
        band = dataset.GetRasterBand(band_idx)
        if band is None:
            raise ValueError(f"GeoTIFF is missing raster band {band_idx}.")
        band_data = band.ReadAsArray(x_start, y_start, width, height)
        if band_data is None:
            raise ValueError(
                f"Failed to read GeoTIFF tile window for band {band_idx}."
            )
        bands.append(band_data)

    return np.stack(bands, axis=2)
