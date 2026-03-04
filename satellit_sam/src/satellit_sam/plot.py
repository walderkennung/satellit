"""Visualization helpers for SAM detection and mask overlays."""

from typing import Optional

import numpy as np
import supervision as sv
import torch

from satellit_sam.core import Image

COLOR = sv.ColorPalette.from_hex(
    [
        "#ffff00",
        "#ff9b00",
        "#ff8080",
        "#ff66b2",
        "#ff66ff",
        "#b266ff",
        "#9999ff",
        "#3399ff",
        "#66ffff",
        "#33ff99",
        "#66ff66",
        "#99ff00",
    ]
)


def from_sam(sam_result: dict) -> sv.Detections:
    """Convert SAM post-processing output to ``supervision`` detections.

    Args:
        sam_result: Dictionary returned by SAM post-processing utilities.

    Returns:
        Structured detections with masks, boxes, and confidence scores.
    """
    masks = sam_result.get("masks")
    boxes = sam_result.get("boxes")
    scores = sam_result.get("scores")
    if masks is None or boxes is None or scores is None:
        raise ValueError("SAM result must contain 'masks', 'boxes', and 'scores'.")

    if len(masks) == 0:
        return sv.Detections.empty()

    masks_tensor = torch.as_tensor(masks).to(dtype=torch.bool)
    if masks_tensor.ndim == 3:
        # ``sv.Detections.from_transformers`` expects (N, 1, H, W) when boxes exist.
        masks_tensor = masks_tensor.unsqueeze(1)
    elif masks_tensor.ndim != 4:
        raise ValueError(
            f"Expected SAM masks with 3 or 4 dimensions, got {masks_tensor.ndim}."
        )

    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
    labels_tensor = torch.zeros_like(scores_tensor, dtype=torch.int64)
    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)

    detections = sv.Detections.from_transformers(
        {
            "masks": masks_tensor,
            "boxes": boxes_tensor,
            "scores": scores_tensor,
            "labels": labels_tensor,
        }
    )
    detections.class_id = None
    return detections


def annotate(
    image: Image, detections: sv.Detections, label: Optional[str] = None
) -> Image:
    """Draw masks, boxes, and optional labels on top of an image.

    Args:
        image: Source image.
        detections: Detection outputs to render.
        label: Optional class/prompt prefix used in rendered text labels.

    Returns:
        Annotated image copy.
    """
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

    mask_annotator = sv.MaskAnnotator(
        color=COLOR, color_lookup=sv.ColorLookup.INDEX, opacity=0.6
    )
    box_annotator = sv.BoxAnnotator(
        color=COLOR, color_lookup=sv.ColorLookup.INDEX, thickness=1
    )
    label_annotator = sv.LabelAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=0.4,
        text_padding=5,
        text_color=sv.Color.BLACK,
        text_thickness=1,
    )

    annotated_image = image.copy()
    annotated_image.data = mask_annotator.annotate(annotated_image.data, detections)
    annotated_image.data = box_annotator.annotate(annotated_image.data, detections)

    if label:
        labels = [f"{label} {confidence:.2f}" for confidence in detections.confidence]
        annotated_image.data = label_annotator.annotate(
            annotated_image.data, detections, labels
        )

    return annotated_image
