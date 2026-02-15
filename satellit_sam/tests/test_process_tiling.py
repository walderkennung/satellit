"""Tests for process tiling workflow and CLI wiring."""

import numpy as np
import pytest
from typer.testing import CliRunner
from osgeo import gdal

from satellit_sam.cli.root import app
from satellit_sam.workflows.process import make_image_tiles
from satellit_sam.workflows.process import tiling as process_tiling


@pytest.mark.unit
def test_make_image_tiles_saves_tiles_and_metadata(small_test_image, temp_dir):
    """Workflow should load an image and persist tiles plus metadata."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "tiles_out"
    small_test_image.save(str(image_path))

    tiles_dir = make_image_tiles(
        image_path=image_path,
        output_path=output_path,
        tile_width=32,
        tile_height=24,
        overlap=8,
    )

    assert tiles_dir.tile_size == (32, 24)
    assert tiles_dir.overlap == (8, 8)
    assert (output_path / "metadata.json").exists()
    assert len(list((output_path / "tiles_rgb").glob("tile_*.png"))) > 0


@pytest.mark.unit
def test_process_tiles_cli_runs(small_test_image, temp_dir):
    """CLI should expose process tiles and call the tiling workflow."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "tiles_cli"
    small_test_image.save(str(image_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "process",
            "tiles",
            "--image",
            str(image_path),
            "--output-path",
            str(output_path),
            "--tile-width",
            "32",
            "--tile-height",
            "24",
            "--overlap",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert (output_path / "metadata.json").exists()
    assert len(list((output_path / "tiles_rgb").glob("tile_*.png"))) > 0


@pytest.mark.unit
def test_make_image_tiles_streams_geotiff_windows(temp_dir, monkeypatch):
    """GeoTIFF tiling should stream windows instead of using Image.load."""
    image_path = temp_dir / "source.tif"
    output_path = temp_dir / "tiles_tif"
    _write_test_geotiff(image_path=image_path, width=64, height=48, channels=3)

    def _fail_image_load(*_args, **_kwargs):
        raise AssertionError("Image.load should not be used for GeoTIFF tiling.")

    monkeypatch.setattr(
        process_tiling.Image,
        "load",
        staticmethod(_fail_image_load),
    )

    tiles_dir = make_image_tiles(
        image_path=image_path,
        output_path=output_path,
        tile_width=16,
        tile_height=12,
        overlap=4,
    )

    assert tiles_dir.original_shape == (48, 64, 3)
    assert (output_path / "metadata.json").exists()
    assert len(list((output_path / "tiles_rgb").glob("tile_*.png"))) > 0


def _write_test_geotiff(
    image_path,
    width: int,
    height: int,
    channels: int,
) -> None:
    """Create a small deterministic GeoTIFF fixture for tests."""
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("GDAL GTiff driver is not available.")

    dataset = driver.Create(str(image_path), width, height, channels, gdal.GDT_Byte)
    if dataset is None:
        raise RuntimeError("Failed to create GeoTIFF test fixture.")

    try:
        for band_idx in range(1, channels + 1):
            band = dataset.GetRasterBand(band_idx)
            if band is None:
                raise RuntimeError(f"Missing GeoTIFF band {band_idx}.")
            band_data = np.full(
                (height, width), fill_value=band_idx * 20, dtype=np.uint8
            )
            band.WriteArray(band_data)
    finally:
        dataset = None
