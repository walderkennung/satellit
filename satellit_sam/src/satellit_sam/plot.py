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
    """Convert a SAM post-processing result to ``supervision`` detections.

    Args:
        sam_result: Dictionary returned by SAM post-processing utilities.

    Returns:
        Structured detections with masks, boxes, and confidence scores.
    """
    if len(sam_result["masks"]) == 0:
        return sv.Detections.empty()

    # Convert to supervision Detections
    masks = np.array([m.cpu().numpy() for m in sam_result["masks"]])  # (N, H, W)
    boxes = np.array([b.cpu().numpy() for b in sam_result["boxes"]])  # (N, 4) xyxy
    scores = np.array([s.cpu().item() for s in sam_result["scores"]])  # (N,)

    return sv.Detections(xyxy=boxes, confidence=scores, mask=masks)


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
