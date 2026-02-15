"""SAM model wrappers and inference helpers for SAM3 and SAM2."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torchvision
import supervision as sv
from transformers import Sam2Model, Sam2Processor, Sam3Model, Sam3Processor

from satellit_sam.core import Image
from satellit_sam.plot import annotate, from_sam

POINT_PROMPT_BOX_RADIUS_PX = 8.0
SamVersion = Literal["sam3", "sam2"]


class SamSingleton:
    """Singleton wrapper for loading and running a selected SAM model."""

    def __init__(self, model_name: SamVersion = "sam3"):
        """Initialize model, processor, and device-specific settings."""
        self.model_name = model_name
        if torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            self.device = "cuda"
        else:
            self.device = "cpu"

        if self.model_name == "sam3":
            self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
            self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        else:
            self.model = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(
                self.device
            )
            self.processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")

    def print_debug_info(self):
        """Print runtime versions and CUDA availability for diagnostics."""
        print("SAM model:", self.model_name)
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

    def predict(
        self,
        image: Image,
        text: str | None = None,
        boxes: list[tuple[float, float, float, float]] | None = None,
        box_labels: list[int] | None = None,
        points: list[tuple[float, float]] | None = None,
        point_labels: list[int] | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        allow_low_confidence_fallback: bool = False,
    ) -> Image:
        """Generate and render segmentation predictions for one image.

        Args:
            image: Image to segment.
            text: Optional text prompt.
            boxes: Optional list of box prompts in ``x1,y1,x2,y2`` format.
            box_labels: Optional per-box labels for the SAM processor.
            points: Optional list of point prompts in ``x,y`` format.
            point_labels: Optional per-point labels for the SAM processor.
            threshold: Score threshold used in SAM post-processing.
            mask_threshold: Pixel mask threshold used in SAM post-processing.
            confidence_threshold: Minimum confidence required to keep detections.
            allow_low_confidence_fallback: Whether to keep the top-scoring mask
                when score filtering removes all detections.

        Returns:
            Annotated image with segmentation overlays.
        """
        detections = self.predict_detections(
            image=image,
            text=text,
            boxes=boxes,
            box_labels=box_labels,
            points=points,
            point_labels=point_labels,
            threshold=threshold,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            allow_low_confidence_fallback=allow_low_confidence_fallback,
        )

        if text:
            label = text
        elif boxes:
            label = "bbox"
        elif points:
            label = "point"
        else:
            label = None
        return annotate(image=image, detections=detections, label=label)

    def predict_detections(
        self,
        image: Image,
        text: str | None = None,
        boxes: list[tuple[float, float, float, float]] | None = None,
        box_labels: list[int] | None = None,
        points: list[tuple[float, float]] | None = None,
        point_labels: list[int] | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        allow_low_confidence_fallback: bool = False,
    ) -> sv.Detections:
        """Generate segmentation detections for one image.

        Args:
            image: Image to segment.
            text: Optional text prompt.
            boxes: Optional list of box prompts in ``x1,y1,x2,y2`` format.
            box_labels: Optional per-box labels for the SAM processor.
            points: Optional list of point prompts in ``x,y`` format.
            point_labels: Optional per-point labels for the SAM processor.
            threshold: Score threshold used in SAM post-processing.
            mask_threshold: Pixel mask threshold used in SAM post-processing.
            confidence_threshold: Minimum confidence required to keep detections.
            allow_low_confidence_fallback: Whether to keep the top-scoring mask
                when score filtering removes all detections.

        Returns:
            Filtered detections including masks, boxes, and confidence scores.

        Raises:
            ValueError: If no prompts are provided.
        """
        if text is None and not boxes and not points:
            raise ValueError(
                "At least one prompt is required: text, boxes, and/or points."
            )
        if self.model_name == "sam2" and text is not None:
            raise ValueError(
                "--text is not supported with model 'sam2'. Use --bbox and/or --point."
            )

        processor_kwargs = {"images": image.data, "return_tensors": "pt"}
        if text is not None:
            processor_kwargs["text"] = text

        effective_boxes: list[tuple[float, float, float, float]] = []
        if boxes:
            effective_boxes.extend(boxes)
        if points and self.model_name == "sam3":
            if point_labels is not None:
                print(
                    "Warning: point labels are ignored; this SAM3 processor version "
                    "supports box labels only."
                )
            if not boxes:
                print(
                    "Note: point prompts are approximated as small box prompts "
                    "for SAM3 compatibility."
                )
            effective_boxes.extend(self._points_to_boxes(image=image, points=points))

        if self.model_name == "sam2" and points:
            point_values = [[[list(point)] for point in points]]
            point_value_labels = [
                [[int(label)] for label in (point_labels or [1] * len(points))]
            ]
            processor_kwargs["input_points"] = point_values
            processor_kwargs["input_labels"] = point_value_labels

        if effective_boxes and self.model_name == "sam3":
            processor_kwargs["input_boxes"] = [[list(box) for box in effective_boxes]]
            processor_kwargs["input_boxes_labels"] = [
                self._build_box_labels(
                    box_labels=box_labels, total_boxes=len(effective_boxes)
                )
            ]
        if effective_boxes and self.model_name == "sam2":
            processor_kwargs["input_boxes"] = [[list(box) for box in effective_boxes]]

        inputs = self.processor(**processor_kwargs).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.model_name == "sam3":
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            detections = from_sam(sam_result=results)
            if len(detections) == 0:
                return detections
            if detections.confidence is None:
                return detections
            filtered = detections[detections.confidence >= confidence_threshold]
            if len(filtered) > 0:
                return filtered
            if allow_low_confidence_fallback:
                top_index = int(np.argmax(detections.confidence))
                print(
                    "Note: no detections above confidence "
                    f"{confidence_threshold:.3f}; "
                    "using highest-confidence mask for visualization."
                )
                return detections[np.array([top_index])]
            return filtered

        post_processed_masks = self.processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=inputs.get("original_sizes"),
            mask_threshold=mask_threshold,
            binarize=True,
        )
        raw_masks = post_processed_masks[0]
        raw_scores = outputs.iou_scores[0] if outputs.iou_scores is not None else None
        detections = self._sam2_masks_to_detections(raw_masks=raw_masks, raw_scores=raw_scores)
        if len(detections) == 0:
            return detections
        if detections.confidence is None:
            return detections
        filtered = detections[detections.confidence >= confidence_threshold]
        if len(filtered) > 0:
            return filtered
        if allow_low_confidence_fallback:
            top_index = int(np.argmax(detections.confidence))
            print(
                "Note: no detections above confidence "
                f"{confidence_threshold:.3f}; "
                "using highest-confidence mask for visualization."
            )
            return detections[np.array([top_index])]
        return filtered

    @staticmethod
    def _build_box_labels(
        box_labels: Sequence[int] | None, total_boxes: int
    ) -> list[int]:
        """Return one label per effective box prompt."""
        if box_labels is None:
            return [1] * total_boxes

        labels = [int(label) for label in box_labels[:total_boxes]]
        if len(labels) < total_boxes:
            labels.extend([1] * (total_boxes - len(labels)))
        return labels

    @staticmethod
    def _points_to_boxes(
        image: Image,
        points: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float, float, float]]:
        """Approximate point prompts as tiny boxes for SAM3 compatibility."""
        image_width, image_height = image.size
        max_x = float(image_width)
        max_y = float(image_height)
        radius = POINT_PROMPT_BOX_RADIUS_PX

        boxes: list[tuple[float, float, float, float]] = []
        for point_x, point_y in points:
            x1 = max(0.0, float(point_x) - radius)
            y1 = max(0.0, float(point_y) - radius)
            x2 = min(max_x, float(point_x) + radius)
            y2 = min(max_y, float(point_y) + radius)

            # Keep box geometry valid even when points are on the image edge.
            if x2 <= x1:
                x2 = min(max_x, x1 + 1.0)
                x1 = max(0.0, x2 - 1.0)
            if y2 <= y1:
                y2 = min(max_y, y1 + 1.0)
                y1 = max(0.0, y2 - 1.0)

            boxes.append((x1, y1, x2, y2))
        return boxes

    @staticmethod
    def _sam2_masks_to_detections(
        raw_masks: torch.Tensor | np.ndarray,
        raw_scores: torch.Tensor | np.ndarray | None,
    ) -> sv.Detections:
        """Convert SAM2 post-processed masks to supervision detections."""
        masks_tensor = torch.as_tensor(raw_masks)
        while masks_tensor.ndim > 4:
            masks_tensor = masks_tensor.squeeze(0)

        scores_tensor = torch.as_tensor(raw_scores) if raw_scores is not None else None
        if scores_tensor is not None:
            while scores_tensor.ndim > 2:
                scores_tensor = scores_tensor.squeeze(0)

        if masks_tensor.ndim == 2:
            masks_tensor = masks_tensor.unsqueeze(0)

        mask_confidences: torch.Tensor | None = None
        if masks_tensor.ndim == 4:
            if scores_tensor is not None and scores_tensor.ndim == 2:
                if scores_tensor.shape[:2] == masks_tensor.shape[:2]:
                    best_index = scores_tensor.argmax(dim=1)
                    prompt_index = torch.arange(masks_tensor.shape[0])
                    mask_confidences = scores_tensor[prompt_index, best_index]
                    masks_tensor = masks_tensor[prompt_index, best_index]
                else:
                    masks_tensor = masks_tensor[:, 0]
            else:
                masks_tensor = masks_tensor[:, 0]

        if masks_tensor.ndim != 3:
            return sv.Detections.empty()

        masks_np = masks_tensor.detach().cpu().numpy()
        masks_np = masks_np.astype(bool)

        scores_np: np.ndarray | None = None
        if mask_confidences is not None:
            scores_np = mask_confidences.detach().cpu().numpy().astype(np.float32)
        elif scores_tensor is not None and scores_tensor.ndim == 1:
            if int(scores_tensor.shape[0]) == int(masks_np.shape[0]):
                scores_np = scores_tensor.detach().cpu().numpy().astype(np.float32)

        filtered_masks: list[np.ndarray] = []
        boxes: list[tuple[float, float, float, float]] = []
        confidences: list[float] = []

        for mask_index, mask in enumerate(masks_np):
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                continue

            x1 = float(xs.min())
            y1 = float(ys.min())
            x2 = float(xs.max() + 1)
            y2 = float(ys.max() + 1)

            filtered_masks.append(mask)
            boxes.append((x1, y1, x2, y2))
            if scores_np is None:
                confidences.append(1.0)
            else:
                confidences.append(float(scores_np[mask_index]))

        if not filtered_masks:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            confidence=np.asarray(confidences, dtype=np.float32),
            mask=np.asarray(filtered_masks, dtype=bool),
        )

_sam_instances: dict[SamVersion, SamSingleton] = {}


def get_sam(model_name: SamVersion = "sam3") -> SamSingleton:
    """Return a cached SAM model wrapper for the selected model version."""
    if model_name not in _sam_instances:
        _sam_instances[model_name] = SamSingleton(model_name=model_name)
    return _sam_instances[model_name]


class _DefaultSamProxy:
    """Lazy proxy preserving ``sam`` compatibility for default SAM3 usage."""

    def __getattr__(self, name: str):
        return getattr(get_sam("sam3"), name)


sam = _DefaultSamProxy()
