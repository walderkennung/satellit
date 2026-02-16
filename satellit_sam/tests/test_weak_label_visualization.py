"""Tests for weak-label tile visualization exports."""

from pathlib import Path

import numpy as np
import pytest

from satellit_sam.workflows.label import weak


def _sample_tiles() -> list[dict[str, object]]:
    """Build minimal tile payloads for visualization tests."""
    return [
        {
            "tile_id": "tile_x0_y0",
            "x0": 0,
            "y0": 0,
            "x1": 6,
            "y1": 6,
            "trees": [
                {
                    "x_pixel": 2,
                    "y_pixel": 3,
                    "crown_radius": 2.0,
                    "bbox_x1": 1.0,
                    "bbox_y1": 1.0,
                    "bbox_x2": 4.0,
                    "bbox_y2": 5.0,
                }
            ],
        },
        {
            "tile_id": "tile_x6_y0",
            "x0": 6,
            "y0": 0,
            "x1": 12,
            "y1": 6,
            "trees": [
                {
                    "x_pixel": 3,
                    "y_pixel": 2,
                    "crown_radius": 1.0,
                    "bbox_x1": 2.0,
                    "bbox_y1": 1.0,
                    "bbox_x2": 4.0,
                    "bbox_y2": 3.0,
                }
            ],
        },
    ]


@pytest.mark.unit
def test_export_visualizations_writes_one_file_per_tile(tmp_path, monkeypatch):
    """Export should produce one visualization image per labeled tile."""
    image_path = tmp_path / "input.png"
    png_data = np.zeros((6, 12, 3), dtype=np.uint8)

    import cv2

    assert cv2.imwrite(str(image_path), png_data)

    written_images: dict[Path, np.ndarray] = {}

    def _capture_write(path: Path, image: np.ndarray) -> None:
        written_images[path] = image.copy()
        path.write_bytes(b"ok")

    monkeypatch.setattr(weak, "_write_lossless_tiff", _capture_write)

    outputs = weak.export_visualizations_opencv(
        image_tif=image_path,
        output_dir=tmp_path,
        tiles=_sample_tiles(),
    )

    tiles_dir = Path(outputs["visualization_tiles_dir"])
    tile_a = tiles_dir / "tile_x0_y0.tif"
    tile_b = tiles_dir / "tile_x6_y0.tif"

    assert tile_a.exists()
    assert tile_b.exists()
    assert outputs["visualization_tiles_count"] == "2"
    assert written_images[tile_a].shape == (6, 6, 3)
    assert written_images[tile_b].shape == (6, 6, 3)
    assert np.any(written_images[tile_a] != 0)
    assert np.any(written_images[tile_b] != 0)


class _FakeBand:
    """Fake GDAL band that records window reads."""

    def __init__(self, band_index: int, calls: list[tuple[int, int, int, int, int]]):
        self.band_index = band_index
        self.calls = calls

    def ReadAsArray(self, x0: int, y0: int, width: int, height: int) -> np.ndarray:
        self.calls.append((self.band_index, x0, y0, width, height))
        return np.full((height, width), self.band_index * 100, dtype=np.uint16)


class _FakeDataset:
    """Fake GDAL dataset for streaming-read assertions."""

    RasterCount = 3

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int, int]] = []

    def GetRasterBand(self, band_index: int) -> _FakeBand:
        return _FakeBand(band_index, self.calls)


@pytest.mark.unit
def test_export_visualizations_streams_tiff_windows(tmp_path, monkeypatch):
    """TIFF exports should read tile windows via GDAL band window calls."""
    fake_dataset = _FakeDataset()

    def _fake_open(*_args, **_kwargs) -> _FakeDataset:
        return fake_dataset

    written_paths: list[Path] = []

    def _capture_write(path: Path, image: np.ndarray) -> None:
        written_paths.append(path)
        path.write_bytes(b"ok")

    monkeypatch.setattr(weak.gdal, "Open", _fake_open)
    monkeypatch.setattr(weak, "_write_lossless_tiff", _capture_write)

    outputs = weak.export_visualizations_opencv(
        image_tif=tmp_path / "input.tif",
        output_dir=tmp_path,
        tiles=_sample_tiles(),
    )

    assert outputs["visualization_tiles_count"] == "2"
    assert len(written_paths) == 2

    expected_calls = {
        (1, 0, 0, 6, 6),
        (2, 0, 0, 6, 6),
        (3, 0, 0, 6, 6),
        (1, 6, 0, 6, 6),
        (2, 6, 0, 6, 6),
        (3, 6, 0, 6, 6),
    }
    assert set(fake_dataset.calls) == expected_calls
