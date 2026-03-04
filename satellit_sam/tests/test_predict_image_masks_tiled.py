"""Tests for streamed tiled image-mask prediction workflow."""

import csv
from pathlib import Path

import numpy as np
import pytest
import supervision as sv
from osgeo import gdal

from satellit_sam.core import Image
from satellit_sam.workflows.predict import image_masks as image_masks_workflow


class _DummySam:
    """Test double for segmentation backends."""

    def __init__(self, predict_impl):
        self._predict_impl = predict_impl

    def print_debug_info(self):
        """No-op debug output for tests."""

    def predict_detections(
        self,
        image: Image,
        text=None,
        boxes=None,
        points=None,
        threshold=0.0,
        confidence_threshold=0.5,
        allow_low_confidence_fallback=False,
    ):
        return self._predict_impl(
            image=image,
            text=text,
            boxes=boxes,
            points=points,
            threshold=threshold,
            confidence_threshold=confidence_threshold,
            allow_low_confidence_fallback=allow_low_confidence_fallback,
        )


@pytest.mark.unit
def test_predict_image_masks_tiled_merges_overlapping_bbox_candidates(
    temp_dir,
    monkeypatch,
):
    """Overlapping tile detections should merge to one global box via NMS."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "predict_out"
    source_data = np.zeros((96, 96, 3), dtype=np.uint8)
    Image(size=(96, 96), channels=3, data=source_data).save(str(image_path))

    prediction_calls: list[tuple[tuple[int, int], list[tuple[float, float, float, float]] | None]] = []

    def _predict_impl(image: Image, boxes=None, **_kwargs):
        prediction_calls.append((image.size, boxes))
        if not boxes:
            return sv.Detections.empty()

        box = boxes[0]
        tile_width, tile_height = image.size
        mask = np.zeros((tile_height, tile_width), dtype=bool)
        x1, y1, x2, y2 = [int(value) for value in box]
        mask[y1:y2, x1:x2] = True
        return sv.Detections(
            xyxy=np.asarray([box], dtype=np.float32),
            confidence=np.asarray([0.9], dtype=np.float32),
            mask=np.asarray([mask], dtype=bool),
        )

    monkeypatch.setattr(
        "satellit_sam.sam3.get_sam",
        lambda model_name="sam3": _DummySam(_predict_impl),
    )
    monkeypatch.setattr(image_masks_workflow, "annotate", lambda image, detections, label=None: image)

    image_masks_workflow.predict_image_masks(
        image_path=image_path,
        output_path=output_path,
        text_prompt=None,
        bbox_prompts=[(40.0, 40.0, 50.0, 50.0)],
        point_prompts=[],
        model="sam3",
        threshold=0.0,
        tile_size=64,
        tile_overlap=32,
        merge_iou_threshold=0.5,
        weak_label_bboxes_by_tile=None,
    )

    assert len(prediction_calls) >= 2

    results = np.load(output_path / "masks" / "image_masks.npz")
    assert int(results["boxes"].shape[0]) == 1
    assert int(results["masks"].shape[0]) == 1
    assert int(results["source_tile_x"].shape[0]) == 1
    assert int(results["source_tile_y"].shape[0]) == 1
    np.testing.assert_allclose(
        results["boxes"][0],
        np.asarray([40.0, 40.0, 50.0, 50.0], dtype=np.float32),
        rtol=0.0,
        atol=1.0,
    )
    possible_origins = {(0, 0), (32, 0), (0, 32), (32, 32)}
    assert (int(results["source_tile_x"][0]), int(results["source_tile_y"][0])) in possible_origins

    tiles_dir = output_path / "masks" / "tiles"
    assert (tiles_dir / "index.csv").exists()
    tile_npz_files = sorted(tiles_dir.glob("tile_x*_y*.npz"))
    assert len(tile_npz_files) == 9


@pytest.mark.unit
def test_predict_image_masks_tiled_projects_points_per_tile(
    temp_dir,
    monkeypatch,
):
    """Point prompts should be projected into tile-local coordinates."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "predict_out"
    source_data = np.zeros((96, 96, 3), dtype=np.uint8)
    Image(size=(96, 96), channels=3, data=source_data).save(str(image_path))

    received_points: list[list[tuple[float, float]] | None] = []

    def _predict_impl(points=None, **_kwargs):
        received_points.append(points)
        return sv.Detections.empty()

    monkeypatch.setattr(
        "satellit_sam.sam3.get_sam",
        lambda model_name="sam2": _DummySam(_predict_impl),
    )
    monkeypatch.setattr(image_masks_workflow, "annotate", lambda image, detections, label=None: image)

    image_masks_workflow.predict_image_masks(
        image_path=image_path,
        output_path=output_path,
        text_prompt=None,
        bbox_prompts=[],
        point_prompts=[(40.0, 40.0)],
        model="sam2",
        threshold=0.5,
        tile_size=64,
        tile_overlap=32,
        merge_iou_threshold=0.5,
        weak_label_bboxes_by_tile=None,
    )

    tile_point_calls = [points for points in received_points if points]
    assert len(tile_point_calls) >= 2
    for points in tile_point_calls:
        assert points is not None
        for point_x, point_y in points:
            assert 0.0 <= point_x < 64.0
            assert 0.0 <= point_y < 64.0


@pytest.mark.unit
def test_predict_image_masks_tiled_uses_matching_weak_label_tile_only(
    temp_dir,
    monkeypatch,
):
    """Weak-label prompts should be consumed only by matching tile ids."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "predict_out"
    source_data = np.zeros((96, 96, 3), dtype=np.uint8)
    Image(size=(96, 96), channels=3, data=source_data).save(str(image_path))

    received_boxes: list[list[tuple[float, float, float, float]] | None] = []

    def _predict_impl(boxes=None, **_kwargs):
        received_boxes.append(boxes)
        return sv.Detections.empty()

    monkeypatch.setattr(
        "satellit_sam.sam3.get_sam",
        lambda model_name="sam3": _DummySam(_predict_impl),
    )
    monkeypatch.setattr(image_masks_workflow, "annotate", lambda image, detections, label=None: image)

    weak_labels = {"tile_x32_y32": [(8.0, 8.0, 20.0, 20.0)]}
    image_masks_workflow.predict_image_masks(
        image_path=image_path,
        output_path=output_path,
        text_prompt=None,
        bbox_prompts=[],
        point_prompts=[],
        model="sam3",
        threshold=0.5,
        tile_size=64,
        tile_overlap=32,
        merge_iou_threshold=0.5,
        weak_label_bboxes_by_tile=weak_labels,
    )

    non_empty_calls = [boxes for boxes in received_boxes if boxes]
    assert len(non_empty_calls) == 1
    assert non_empty_calls[0] == [(8.0, 8.0, 20.0, 20.0)]


@pytest.mark.unit
def test_predict_image_masks_tiled_writes_empty_npz_for_no_detections(
    temp_dir,
    monkeypatch,
):
    """Workflow should write empty mask arrays when no tile detections survive."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "predict_out"
    source_data = np.zeros((64, 64, 3), dtype=np.uint8)
    Image(size=(64, 64), channels=3, data=source_data).save(str(image_path))

    monkeypatch.setattr(
        "satellit_sam.sam3.get_sam",
        lambda model_name="dinov3": _DummySam(lambda **_kwargs: sv.Detections.empty()),
    )
    monkeypatch.setattr(image_masks_workflow, "annotate", lambda image, detections, label=None: image)

    image_masks_workflow.predict_image_masks(
        image_path=image_path,
        output_path=output_path,
        text_prompt="tree",
        bbox_prompts=[],
        point_prompts=[],
        model="dinov3",
        threshold=0.5,
        tile_size=32,
        tile_overlap=16,
        merge_iou_threshold=0.5,
        weak_label_bboxes_by_tile=None,
    )

    results = np.load(output_path / "masks" / "image_masks.npz")
    assert int(results["masks"].shape[0]) == 0
    assert int(results["boxes"].shape[0]) == 0
    assert int(results["scores"].shape[0]) == 0
    assert int(results["source_tile_x"].shape[0]) == 0
    assert int(results["source_tile_y"].shape[0]) == 0

    index_csv = output_path / "masks" / "tiles" / "index.csv"
    assert index_csv.exists()
    with index_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 16
    assert all(int(row["count"]) == 0 for row in rows)


@pytest.mark.unit
def test_predict_image_masks_tiled_writes_per_tile_npz_schema(
    temp_dir,
    monkeypatch,
):
    """Per-tile NPZ artifacts should include canonical schema and local detections."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "predict_out"
    source_data = np.zeros((32, 32, 3), dtype=np.uint8)
    Image(size=(32, 32), channels=3, data=source_data).save(str(image_path))

    def _predict_impl(image: Image, boxes=None, **_kwargs):
        tile_width, tile_height = image.size
        mask = np.zeros((tile_height, tile_width), dtype=bool)
        mask[4:12, 6:14] = True
        return sv.Detections(
            xyxy=np.asarray([[6.0, 4.0, 14.0, 12.0]], dtype=np.float32),
            confidence=np.asarray([0.95], dtype=np.float32),
            mask=np.asarray([mask], dtype=bool),
        )

    monkeypatch.setattr(
        "satellit_sam.sam3.get_sam",
        lambda model_name="sam3": _DummySam(_predict_impl),
    )
    monkeypatch.setattr(
        image_masks_workflow, "annotate", lambda image, detections, label=None: image
    )

    image_masks_workflow.predict_image_masks(
        image_path=image_path,
        output_path=output_path,
        text_prompt="tree",
        bbox_prompts=[],
        point_prompts=[],
        model="sam3",
        threshold=0.0,
        tile_size=32,
        tile_overlap=0,
        merge_iou_threshold=0.5,
        weak_label_bboxes_by_tile=None,
    )

    tile_npz = output_path / "masks" / "tiles" / "tile_x0_y0.npz"
    assert tile_npz.exists()
    with np.load(tile_npz, allow_pickle=False) as tile_data:
        assert set(tile_data.files) == {
            "tile_id",
            "tile_origin",
            "tile_size",
            "masks",
            "boxes",
            "scores",
        }
        assert str(tile_data["tile_id"]) == "tile_x0_y0"
        np.testing.assert_array_equal(
            tile_data["tile_origin"],
            np.asarray([0, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            tile_data["tile_size"],
            np.asarray([32, 32], dtype=np.int32),
        )
        assert tile_data["masks"].shape == (1, 32, 32)
        assert tile_data["masks"].dtype == np.bool_
        assert tile_data["boxes"].shape == (1, 4)
        assert tile_data["boxes"].dtype == np.float32
        assert tile_data["scores"].shape == (1,)
        assert tile_data["scores"].dtype == np.float32

    index_csv = output_path / "masks" / "tiles" / "index.csv"
    with index_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["tile_id"] == "tile_x0_y0"
    assert int(rows[0]["count"]) == 1


@pytest.mark.unit
def test_predict_image_masks_tiled_streams_geotiff_without_image_load(
    temp_dir,
    monkeypatch,
):
    """GeoTIFF tile prediction should not use Image.load for tile inference."""
    image_path = temp_dir / "source.tif"
    output_path = temp_dir / "predict_out"
    _write_test_geotiff(image_path=image_path, width=64, height=48, channels=3)

    def _fail_image_load(*_args, **_kwargs):
        raise AssertionError("Image.load should not be used for GeoTIFF tile inference.")

    monkeypatch.setattr(image_masks_workflow.Image, "load", staticmethod(_fail_image_load))
    monkeypatch.setattr(
        image_masks_workflow,
        "_load_image_for_visualization",
        lambda _path: Image(
            size=(64, 48),
            channels=3,
            data=np.zeros((48, 64, 3), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        "satellit_sam.sam3.get_sam",
        lambda model_name="dinov3": _DummySam(lambda **_kwargs: sv.Detections.empty()),
    )
    monkeypatch.setattr(image_masks_workflow, "annotate", lambda image, detections, label=None: image)

    image_masks_workflow.predict_image_masks(
        image_path=image_path,
        output_path=output_path,
        text_prompt="tree",
        bbox_prompts=[],
        point_prompts=[],
        model="dinov3",
        threshold=0.5,
        tile_size=32,
        tile_overlap=8,
        merge_iou_threshold=0.5,
        weak_label_bboxes_by_tile=None,
    )

    assert (output_path / "image_masks_visualization.png").exists()
    assert (output_path / "masks" / "image_masks.npz").exists()


def _write_test_geotiff(
    image_path: Path,
    width: int,
    height: int,
    channels: int,
) -> None:
    """Create a deterministic GeoTIFF fixture for streamed-prediction tests."""
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
                (height, width),
                fill_value=band_idx * 20,
                dtype=np.uint8,
            )
            band.WriteArray(band_data)
    finally:
        dataset = None
