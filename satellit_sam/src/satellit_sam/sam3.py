"""Singleton SAM3 model wrapper and inference helpers."""

import torch
import torchvision
from transformers import Sam3Model, Sam3Processor

from satellit_sam.core import Image
from satellit_sam.plot import annotate, from_sam


class SamSingleton:
    """Singleton wrapper for loading and running the SAM3 model."""

    def __init__(self):
        """Initialize model, processor, and device-specific settings."""
        if torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def print_debug_info(self):
        """Print runtime versions and CUDA availability for diagnostics."""
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

    def predict(
        self,
        image: Image,
        text: str | None = None,
        boxes: list[tuple[float, float, float, float]] | None = None,
        box_labels: list[int] | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> Image:
        """Generate and render segmentation predictions for one image.

        Args:
            image: Image to segment.
            text: Optional text prompt.
            boxes: Optional list of box prompts in ``x1,y1,x2,y2`` format.
            box_labels: Optional per-box labels for the SAM processor.
            threshold: Score threshold used in SAM post-processing.
            mask_threshold: Pixel mask threshold used in SAM post-processing.

        Returns:
            Annotated image with segmentation overlays.

        Raises:
            ValueError: If neither text nor box prompts are provided.
        """
        if text is None and not boxes:
            raise ValueError("At least one prompt is required: text and/or boxes.")

        processor_kwargs = {"images": image.data, "return_tensors": "pt"}
        if text is not None:
            processor_kwargs["text"] = text

        if boxes:
            processor_kwargs["input_boxes"] = [[list(box) for box in boxes]]
            processor_kwargs["input_boxes_labels"] = [box_labels or [1] * len(boxes)]

        inputs = self.processor(**processor_kwargs).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        detections = from_sam(sam_result=results)
        detections = detections[detections.confidence > 0.5]

        label = text if text else ("bbox" if boxes else None)
        return annotate(image=image, detections=detections, label=label)


sam = SamSingleton()
