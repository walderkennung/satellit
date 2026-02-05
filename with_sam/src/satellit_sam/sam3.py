from dataclasses import dataclass

import torch
import torchvision
from transformers import Sam3Model, Sam3Processor

from satellit_sam.plot import annotate, from_sam

from .image_processing import Image


class SamSingleton:
    def __init__(self):
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
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

    def predict(
        self,
        image: Image,
        text: str,
        threshold: float = 0.01,
        mask_threshold: float = 0.01,
    ) -> Image:
        # Segment using text prompt
        inputs = self.processor(images=image.data, text=text, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        detections = from_sam(sam_result=results)
        detections = detections[detections.confidence > 0.5]

        return annotate(image=image, detections=detections, label=text)


sam = SamSingleton()
