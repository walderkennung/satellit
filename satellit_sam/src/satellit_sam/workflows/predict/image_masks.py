"""Image-mask prediction workflow for full-image SAM inference."""

from pathlib import Path
from typing import Literal

import numpy as np

from satellit_sam.core import Image
from satellit_sam.plot import annotate


def predict_image_masks(
    image_path: Path,
    output_path: Path,
    text_prompt: str | None,
    bbox_prompts: list[tuple[float, float, float, float]],
    point_prompts: list[tuple[float, float]],
    model: Literal["sam3", "sam2"] = "sam3",
    threshold: float = 0.5,
) -> None:
    """Predict image masks from one image and save outputs.

    The workflow:
    1) loads the input image,
    2) runs SAM3 mask prediction on the full image,
    3) saves one mask visualization, and
    4) saves predicted masks and metadata as one ``.npz`` file.

    Args:
        image_path: Path to the input image.
        output_path: Output directory for all artifacts.
        text_prompt: Optional text prompt for SAM.
        bbox_prompts: Optional image-space bbox prompts.
        point_prompts: Optional image-space point prompts.
        model: SAM model family to use (``sam3`` or ``sam2``).
        threshold: Confidence threshold for keeping predicted masks.

    Raises:
        ValueError: If no prompt is provided.
    """
    from satellit_sam.sam3 import get_sam

    if text_prompt is None and not bbox_prompts and not point_prompts:
        raise ValueError(
            "At least one prompt is required: --text, --bbox, and/or --point."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    masks_dir = output_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    sam = get_sam(model_name=model)
    sam.print_debug_info()
    print(f"Loading image from: {image_path}")
    image = Image.load(str(image_path))

    if text_prompt:
        label = text_prompt
    elif bbox_prompts:
        label = "bbox"
    elif point_prompts:
        label = "point"
    else:
        label = None

    detections = sam.predict_detections(
        image=image,
        text=text_prompt,
        boxes=bbox_prompts or None,
        points=point_prompts or None,
        threshold=0.0,
        confidence_threshold=threshold,
        allow_low_confidence_fallback=True,
    )
    ann_image = annotate(image=image, detections=detections, label=label)

    visualization_path = output_path / "image_masks_visualization.png"
    ann_image.save(str(visualization_path))
    masks_path = masks_dir / "image_masks.npz"
    _save_masks(
        masks_path=masks_path,
        image_size=image.size,
        masks=detections.mask,
        boxes=detections.xyxy,
        scores=detections.confidence,
    )

    print("✓ Mask prediction complete.")
    _print_prediction_summary(detections=detections)
    print(f"Visualization saved to: {visualization_path}")
    print(f"Predicted masks saved to: {masks_path}")


def _save_masks(
    masks_path: Path,
    image_size: tuple[int, int],
    masks: np.ndarray | None = None,
    boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> None:
    """Save SAM outputs as one compressed ``.npz`` file."""
    image_width, image_height = image_size
    empty_masks = np.empty((0, image_height, image_width), dtype=np.uint8)
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    empty_scores = np.empty((0,), dtype=np.float32)

    np.savez_compressed(
        masks_path,
        masks=np.asarray(masks) if masks is not None else empty_masks,
        boxes=np.asarray(boxes) if boxes is not None else empty_boxes,
        scores=np.asarray(scores) if scores is not None else empty_scores,
        image_size=np.asarray([image_width, image_height], dtype=np.int32),
    )


def _print_prediction_summary(detections) -> None:
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
