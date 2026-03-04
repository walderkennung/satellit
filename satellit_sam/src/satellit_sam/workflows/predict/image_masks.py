"""Image-mask prediction workflow for streamed full-image segmentation inference."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import supervision as sv
import torch
from osgeo import gdal
from torchvision.ops import nms

from satellit_sam.core import Image
from satellit_sam.core.tiling import iter_tile_geometries
from satellit_sam.plot import annotate
from satellit_sam.prompts import (
    project_bboxes_to_tile,
    project_points_to_tile,
    tile_id_from_origin,
)
from satellit_sam.workflows.run_metadata import write_run_metadata

gdal.UseExceptions()

_GEOTIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class _TileInput:
    """One prediction tile and its origin in full-image coordinates."""

    image: Image
    origin: tuple[int, int]


@dataclass
class _TileCandidate:
    """One tile-local detection projected to full-image coordinates."""

    global_box: tuple[float, float, float, float]
    score: float
    tile_mask: np.ndarray
    tile_origin: tuple[int, int]


@dataclass
class _TileArtifactIndexRow:
    """One tile artifact manifest row."""

    tile_id: str
    x0: int
    y0: int
    width: int
    height: int
    count: int


def predict_image_masks(
    image_path: Path,
    output_path: Path,
    text_prompt: str | None,
    bbox_prompts: list[tuple[float, float, float, float]],
    point_prompts: list[tuple[float, float]],
    model: Literal["sam3", "sam2", "dinov3"] = "sam3",
    threshold: float = 0.5,
    tile_size: int = 640,
    tile_overlap: int = 64,
    merge_iou_threshold: float = 0.5,
    weak_label_bboxes_by_tile: (
        dict[str, list[tuple[float, float, float, float]]] | None
    ) = None,
    command: str | None = None,
) -> None:
    """Predict image masks from one image and save outputs.

    The workflow:
    1) streams model inference over image tiles,
    2) merges tile detections globally via NMS,
    3) saves per-tile strong-label artifacts,
    4) saves one mask visualization, and
    5) saves merged predicted masks and metadata as one ``.npz`` file.

    Args:
        image_path: Path to the input image.
        output_path: Output directory for all artifacts.
        text_prompt: Optional text prompt for segmentation filtering.
        bbox_prompts: Optional image-space bbox prompts.
        point_prompts: Optional image-space point prompts.
        model: Segmentation model family to use (``sam3``, ``sam2``, ``dinov3``).
        threshold: Confidence threshold for keeping predicted masks.
        tile_size: Prediction tile size in pixels.
        tile_overlap: Overlap between neighboring prediction tiles in pixels.
        merge_iou_threshold: IoU threshold for cross-tile NMS merge.
        weak_label_bboxes_by_tile: Optional tile-local bboxes keyed by tile id.
        command: Optional CLI command string used to start the run.

    Raises:
        ValueError: If inputs or prompt combinations are invalid.
    """
    from satellit_sam.sam3 import get_sam

    if tile_size <= 0:
        raise ValueError("`tile_size` must be > 0.")
    if tile_overlap < 0:
        raise ValueError("`tile_overlap` must be >= 0.")
    if tile_overlap >= tile_size:
        raise ValueError("`tile_overlap` must be smaller than `tile_size`.")
    if merge_iou_threshold < 0.0 or merge_iou_threshold > 1.0:
        raise ValueError("`merge_iou_threshold` must be in the range [0.0, 1.0].")

    has_weak_label_prompts = bool(weak_label_bboxes_by_tile)
    if model == "sam2" and text_prompt is not None:
        raise ValueError(
            "--text is not supported with model 'sam2'. Use --bbox and/or --point."
        )
    if model == "dinov3":
        if text_prompt is None:
            raise ValueError("Model 'dinov3' requires --text.")
        if bbox_prompts or point_prompts or has_weak_label_prompts:
            raise ValueError(
                "Model 'dinov3' supports --text prompts only (no --bbox/--point/--weak-labels-csv)."
            )
    elif text_prompt is None and not bbox_prompts and not point_prompts and not has_weak_label_prompts:
        raise ValueError(
            (
                "At least one prompt is required: --text, --bbox, --point, "
                "and/or a matching --weak-labels-csv entry for at least one tile."
            )
        )

    output_path.mkdir(parents=True, exist_ok=True)
    masks_dir = output_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_tiles_dir = masks_dir / "tiles"
    mask_tiles_dir.mkdir(parents=True, exist_ok=True)

    sam = get_sam(model_name=model)
    sam.print_debug_info()
    print(f"Preparing streamed tile prediction from: {image_path}")

    source_image: Image | None = None
    if image_path.suffix.lower() in _GEOTIFF_EXTENSIONS:
        image_size = _get_geotiff_size(image_path)
        tile_inputs = _iter_geotiff_tile_inputs(
            image_path=image_path,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    else:
        source_image = Image.load(str(image_path))
        image_size = source_image.size
        tile_inputs = _iter_in_memory_tile_inputs(
            image=source_image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

    tile_candidates: list[_TileCandidate] = []
    tile_index_rows: list[_TileArtifactIndexRow] = []
    for tile_input in tile_inputs:
        tile_origin = tile_input.origin
        tile_id = tile_id_from_origin(tile_origin)
        tile_image = tile_input.image
        tile_width, tile_height = tile_image.size
        tile_bbox_prompts = project_bboxes_to_tile(
            image_bboxes=bbox_prompts,
            tile_origin=tile_origin,
            tile_size=tile_image.size,
        )
        if weak_label_bboxes_by_tile:
            tile_bbox_prompts.extend(weak_label_bboxes_by_tile.get(tile_id, []))
        tile_point_prompts = project_points_to_tile(
            image_points=point_prompts,
            tile_origin=tile_origin,
            tile_size=tile_image.size,
        )
        print(
            "Processing tile "
            f"{tile_id}: "
            f"position=({tile_origin[0]}, {tile_origin[1]}), "
            f"size=({tile_width}, {tile_height}), "
            f"bboxes={_format_tile_bboxes(tile_bbox_prompts)}"
        )

        if text_prompt is None and not tile_bbox_prompts and not tile_point_prompts:
            empty_masks = np.empty((0, tile_height, tile_width), dtype=bool)
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            _save_tile_masks_npz(
                tile_masks_path=mask_tiles_dir / f"{tile_id}.npz",
                tile_id=tile_id,
                tile_origin=tile_origin,
                tile_size=tile_image.size,
                masks=empty_masks,
                boxes=empty_boxes,
                scores=empty_scores,
            )
            tile_index_rows.append(
                _TileArtifactIndexRow(
                    tile_id=tile_id,
                    x0=tile_origin[0],
                    y0=tile_origin[1],
                    width=tile_width,
                    height=tile_height,
                    count=0,
                )
            )
            continue

        detections = sam.predict_detections(
            image=tile_image,
            text=text_prompt,
            boxes=tile_bbox_prompts or None,
            points=tile_point_prompts or None,
            threshold=0.0,
            confidence_threshold=threshold,
            allow_low_confidence_fallback=True,
        )
        local_boxes, local_scores, local_masks = _tile_detections_to_arrays(
            detections=detections,
            tile_size=tile_image.size,
        )
        _save_tile_masks_npz(
            tile_masks_path=mask_tiles_dir / f"{tile_id}.npz",
            tile_id=tile_id,
            tile_origin=tile_origin,
            tile_size=tile_image.size,
            masks=local_masks,
            boxes=local_boxes,
            scores=local_scores,
        )
        tile_index_rows.append(
            _TileArtifactIndexRow(
                tile_id=tile_id,
                x0=tile_origin[0],
                y0=tile_origin[1],
                width=tile_width,
                height=tile_height,
                count=int(local_boxes.shape[0]),
            )
        )
        tile_candidates.extend(
            _tile_arrays_to_candidates(
                local_boxes=local_boxes,
                local_scores=local_scores,
                local_masks=local_masks,
                tile_origin=tile_origin,
                image_size=image_size,
            )
        )

    detections, source_tile_x, source_tile_y = _merge_tile_candidates(
        candidates=tile_candidates,
        image_size=image_size,
        merge_iou_threshold=merge_iou_threshold,
    )
    tile_index_path = mask_tiles_dir / "index.csv"
    _write_tile_index(index_path=tile_index_path, rows=tile_index_rows)

    if source_image is None:
        print(f"Loading image from: {image_path}")
        source_image = _load_image_for_visualization(image_path)

    if text_prompt:
        label = text_prompt
    elif bbox_prompts or has_weak_label_prompts:
        label = "bbox"
    elif point_prompts:
        label = "point"
    else:
        label = None

    ann_image = annotate(image=source_image, detections=detections, label=label)

    visualization_path = output_path / "image_masks_visualization.png"
    ann_image.save(str(visualization_path))
    masks_path = masks_dir / "image_masks.npz"
    _save_masks(
        masks_path=masks_path,
        image_size=source_image.size,
        masks=detections.mask,
        boxes=detections.xyxy,
        scores=detections.confidence,
        source_tile_x=source_tile_x,
        source_tile_y=source_tile_y,
    )
    metadata_path = write_run_metadata(
        output_dir=output_path,
        image_path=image_path,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        prompt={
            "text": text_prompt,
            "bbox_prompts": [list(prompt) for prompt in bbox_prompts],
            "point_prompts": [list(prompt) for prompt in point_prompts],
            "weak_label_prompt_count": (
                sum(len(prompts) for prompts in weak_label_bboxes_by_tile.values())
                if weak_label_bboxes_by_tile
                else 0
            ),
        },
        model=model,
        command=command,
    )

    print(f"Finished tile processing. Total masks found: {len(detections)}")
    print("✓ Mask prediction complete.")
    _print_prediction_summary(detections=detections)
    print(f"Visualization saved to: {visualization_path}")
    print(f"Predicted masks saved to: {masks_path}")
    print(f"Per-tile masks saved under: {mask_tiles_dir}")
    print(f"Run metadata saved to: {metadata_path}")


def _iter_in_memory_tile_inputs(
    image: Image,
    tile_size: int,
    tile_overlap: int,
) -> Iterator[_TileInput]:
    """Yield in-memory image crops for streamed prediction."""
    for tile_geo in iter_tile_geometries(
        image_shape=image.size,
        tile_size=(tile_size, tile_size),
        overlap=(tile_overlap, tile_overlap),
    ):
        x_start, y_start = tile_geo.start
        x_end, y_end = tile_geo.end
        yield _TileInput(
            image=image.crop(x_start, y_start, x_end, y_end),
            origin=tile_geo.start,
        )


def _iter_geotiff_tile_inputs(
    image_path: Path,
    tile_size: int,
    tile_overlap: int,
) -> Iterator[_TileInput]:
    """Yield streamed GeoTIFF windows for prediction."""
    dataset = gdal.Open(str(image_path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Could not open GeoTIFF: {image_path}")

    try:
        width = int(dataset.RasterXSize)
        height = int(dataset.RasterYSize)
        channels = _select_channel_count(int(dataset.RasterCount))
        for tile_geo in iter_tile_geometries(
            image_shape=(width, height),
            tile_size=(tile_size, tile_size),
            overlap=(tile_overlap, tile_overlap),
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
            yield _TileInput(
                image=Image(
                    size=(tile_width, tile_height),
                    channels=channels,
                    data=tile_data,
                ),
                origin=(x_start, y_start),
            )
    finally:
        dataset = None


def _tile_detections_to_arrays(
    detections: sv.Detections,
    tile_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize one tile's detections into ``(boxes, scores, masks)`` arrays."""
    tile_width, tile_height = tile_size
    if len(detections) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, tile_height, tile_width), dtype=bool),
        )

    boxes = detections.xyxy
    if boxes is None:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, tile_height, tile_width), dtype=bool),
        )
    boxes = np.asarray(boxes, dtype=np.float32)

    if detections.confidence is None:
        confidences = np.ones((len(boxes),), dtype=np.float32)
    else:
        confidences = np.asarray(detections.confidence, dtype=np.float32).reshape(-1)

    if detections.mask is None:
        tile_masks = _boxes_to_tile_masks(boxes=boxes, tile_size=tile_size)
    else:
        tile_masks = np.asarray(detections.mask, dtype=bool)

    effective_count = min(len(boxes), len(confidences), len(tile_masks))
    if effective_count == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, tile_height, tile_width), dtype=bool),
        )

    normalized_masks: list[np.ndarray] = []
    for idx in range(effective_count):
        normalized_masks.append(_normalize_tile_mask(tile_masks[idx], tile_size=tile_size))

    return (
        boxes[:effective_count].astype(np.float32, copy=False),
        confidences[:effective_count].astype(np.float32, copy=False),
        np.asarray(normalized_masks, dtype=bool),
    )


def _tile_arrays_to_candidates(
    local_boxes: np.ndarray,
    local_scores: np.ndarray,
    local_masks: np.ndarray,
    tile_origin: tuple[int, int],
    image_size: tuple[int, int],
) -> list[_TileCandidate]:
    """Convert one tile's normalized arrays into global candidate objects."""
    candidates: list[_TileCandidate] = []
    for idx, local_box in enumerate(local_boxes):
        global_box = _local_box_to_global(
            local_box=local_box,
            tile_origin=tile_origin,
            image_size=image_size,
        )
        if global_box is None:
            continue
        candidates.append(
            _TileCandidate(
                global_box=global_box,
                score=float(local_scores[idx]),
                tile_mask=np.asarray(local_masks[idx], dtype=bool),
                tile_origin=tile_origin,
            )
        )
    return candidates


def _merge_tile_candidates(
    candidates: list[_TileCandidate],
    image_size: tuple[int, int],
    merge_iou_threshold: float,
) -> tuple[sv.Detections, np.ndarray, np.ndarray]:
    """Merge cross-tile candidates with global box NMS."""
    if not candidates:
        return (
            sv.Detections.empty(),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    boxes = np.asarray([candidate.global_box for candidate in candidates], dtype=np.float32)
    scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float32)
    kept_indices = nms(
        boxes=torch.from_numpy(boxes),
        scores=torch.from_numpy(scores),
        iou_threshold=merge_iou_threshold,
    )
    if kept_indices.numel() == 0:
        return (
            sv.Detections.empty(),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    image_width, image_height = image_size
    kept_boxes: list[tuple[float, float, float, float]] = []
    kept_scores: list[float] = []
    kept_masks: list[np.ndarray] = []
    kept_source_tile_x: list[int] = []
    kept_source_tile_y: list[int] = []
    for index in kept_indices.tolist():
        candidate = candidates[int(index)]
        tile_mask = candidate.tile_mask
        tile_x, tile_y = candidate.tile_origin

        full_mask = np.zeros((image_height, image_width), dtype=bool)
        y_end = min(tile_y + tile_mask.shape[0], image_height)
        x_end = min(tile_x + tile_mask.shape[1], image_width)
        if x_end <= tile_x or y_end <= tile_y:
            continue

        full_mask[tile_y:y_end, tile_x:x_end] = tile_mask[
            : (y_end - tile_y), : (x_end - tile_x)
        ]
        kept_boxes.append(candidate.global_box)
        kept_scores.append(candidate.score)
        kept_masks.append(full_mask)
        kept_source_tile_x.append(tile_x)
        kept_source_tile_y.append(tile_y)

    if not kept_masks:
        return (
            sv.Detections.empty(),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        sv.Detections(
            xyxy=np.asarray(kept_boxes, dtype=np.float32),
            confidence=np.asarray(kept_scores, dtype=np.float32),
            mask=np.asarray(kept_masks, dtype=bool),
        ),
        np.asarray(kept_source_tile_x, dtype=np.int32),
        np.asarray(kept_source_tile_y, dtype=np.int32),
    )


def _local_box_to_global(
    local_box: np.ndarray,
    tile_origin: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Translate one tile-local box to global image coordinates."""
    image_width, image_height = image_size
    tile_x, tile_y = tile_origin
    x1, y1, x2, y2 = [float(value) for value in local_box]

    global_x1 = max(0.0, min(float(image_width), x1 + tile_x))
    global_y1 = max(0.0, min(float(image_height), y1 + tile_y))
    global_x2 = max(0.0, min(float(image_width), x2 + tile_x))
    global_y2 = max(0.0, min(float(image_height), y2 + tile_y))

    if global_x2 <= global_x1 or global_y2 <= global_y1:
        return None
    return (global_x1, global_y1, global_x2, global_y2)


def _boxes_to_tile_masks(
    boxes: np.ndarray,
    tile_size: tuple[int, int],
) -> np.ndarray:
    """Approximate masks from boxes when a backend does not return masks."""
    tile_width, tile_height = tile_size
    masks: list[np.ndarray] = []
    for x1, y1, x2, y2 in boxes.astype(np.float32):
        mask = np.zeros((tile_height, tile_width), dtype=bool)
        x1i = max(0, min(tile_width, int(np.floor(x1))))
        y1i = max(0, min(tile_height, int(np.floor(y1))))
        x2i = max(0, min(tile_width, int(np.ceil(x2))))
        y2i = max(0, min(tile_height, int(np.ceil(y2))))
        if x2i > x1i and y2i > y1i:
            mask[y1i:y2i, x1i:x2i] = True
        masks.append(mask)
    if not masks:
        return np.empty((0, tile_height, tile_width), dtype=bool)
    return np.asarray(masks, dtype=bool)


def _normalize_tile_mask(mask: np.ndarray, tile_size: tuple[int, int]) -> np.ndarray:
    """Return a boolean mask with shape matching ``tile_size``."""
    tile_width, tile_height = tile_size
    target_shape = (tile_height, tile_width)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        return np.zeros(target_shape, dtype=bool)
    if mask_array.shape == target_shape:
        return mask_array

    normalized = np.zeros(target_shape, dtype=bool)
    copy_height = min(mask_array.shape[0], target_shape[0])
    copy_width = min(mask_array.shape[1], target_shape[1])
    normalized[:copy_height, :copy_width] = mask_array[:copy_height, :copy_width]
    return normalized


def _get_geotiff_size(image_path: Path) -> tuple[int, int]:
    """Read image ``(width, height)`` without loading full raster data."""
    dataset = gdal.Open(str(image_path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Could not open GeoTIFF: {image_path}")
    try:
        return int(dataset.RasterXSize), int(dataset.RasterYSize)
    finally:
        dataset = None


def _load_image_for_visualization(image_path: Path) -> Image:
    """Load the full source image used for final visualization export."""
    return Image.load(str(image_path))


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

    bands: list[np.ndarray] = []
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


def _save_masks(
    masks_path: Path,
    image_size: tuple[int, int],
    masks: np.ndarray | None = None,
    boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    source_tile_x: np.ndarray | None = None,
    source_tile_y: np.ndarray | None = None,
) -> None:
    """Save merged model outputs as one compressed ``.npz`` file."""
    image_width, image_height = image_size
    empty_masks = np.empty((0, image_height, image_width), dtype=np.uint8)
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    empty_scores = np.empty((0,), dtype=np.float32)
    empty_source = np.empty((0,), dtype=np.int32)

    np.savez_compressed(
        masks_path,
        masks=np.asarray(masks) if masks is not None else empty_masks,
        boxes=np.asarray(boxes) if boxes is not None else empty_boxes,
        scores=np.asarray(scores) if scores is not None else empty_scores,
        image_size=np.asarray([image_width, image_height], dtype=np.int32),
        source_tile_x=(
            np.asarray(source_tile_x, dtype=np.int32)
            if source_tile_x is not None
            else empty_source
        ),
        source_tile_y=(
            np.asarray(source_tile_y, dtype=np.int32)
            if source_tile_y is not None
            else empty_source
        ),
    )


def _save_tile_masks_npz(
    tile_masks_path: Path,
    tile_id: str,
    tile_origin: tuple[int, int],
    tile_size: tuple[int, int],
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Save one tile's strong-label detections as compressed NPZ."""
    tile_width, tile_height = tile_size
    np.savez_compressed(
        tile_masks_path,
        tile_id=np.asarray(tile_id),
        tile_origin=np.asarray([tile_origin[0], tile_origin[1]], dtype=np.int32),
        tile_size=np.asarray([tile_width, tile_height], dtype=np.int32),
        masks=np.asarray(masks, dtype=bool),
        boxes=np.asarray(boxes, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
    )


def _format_tile_bboxes(
    tile_bboxes: list[tuple[float, float, float, float]],
) -> str:
    """Format tile-local bbox prompts as ``position`` and ``size`` tuples."""
    if not tile_bboxes:
        return "none"

    formatted_bboxes: list[str] = []
    for idx, bbox in enumerate(tile_bboxes, start=1):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        formatted_bboxes.append(
            f"{idx}: pos=({x1:.1f}, {y1:.1f}), size=({width:.1f}, {height:.1f})"
        )
    return "[" + "; ".join(formatted_bboxes) + "]"


def _write_tile_index(
    index_path: Path,
    rows: list[_TileArtifactIndexRow],
) -> None:
    """Write tile-artifact manifest rows as CSV."""
    with index_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tile_id", "x0", "y0", "width", "height", "count"])
        for row in rows:
            writer.writerow(
                [
                    row.tile_id,
                    row.x0,
                    row.y0,
                    row.width,
                    row.height,
                    row.count,
                ]
            )


def _print_prediction_summary(detections: sv.Detections) -> None:
    """Print a compact CLI summary of prediction results."""
    mask_count = len(detections)
    box_count = 0
    if detections.xyxy is not None:
        box_count = int(len(detections.xyxy))

    print("Prediction summary:")
    print(f"- masks: {mask_count}")
    print(f"- boxes: {box_count}")

    scores = detections.confidence
    if scores is None or len(scores) == 0:
        print("- confidence: n/a")
        return

    scores_array = np.asarray(scores, dtype=np.float32)
    print(
        "- confidence: "
        f"min={scores_array.min():.3f}, "
        f"mean={scores_array.mean():.3f}, "
        f"max={scores_array.max():.3f}"
    )
