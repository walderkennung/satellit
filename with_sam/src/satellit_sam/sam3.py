from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


@dataclass
class PredictionResult:
    masks: torch.Tensor  # (batch_size, num_queries, height, width)
    boxes: torch.Tensor  # (batch_size, num_queries, 4)
    scores: torch.Tensor  # (batch_size, num_queries)

    def save(self, output_path: str):
        np.savez_compressed(
            output_path,
            masks=self.masks.cpu().numpy(),
            boxes=self.boxes.cpu().numpy(),
            scores=self.scores.cpu().numpy(),
        )


class SamSingleton:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def predict(self, image: Image.Image, text: str):
        # Segment using text prompt
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = outputs["pred_logits"]
        result = PredictionResult(
            masks=outputs["pred_masks"],
            boxes=outputs["pred_boxes"],
            scores=scores if scores is not None else torch.Tensor([]),
        )

        print(f"Found {len(result.masks)} objects")
        return result

    def overlay_masks(self, image, masks):
        image = image.convert("RGBA")
        masks = 255 * masks.cpu().numpy().astype(np.uint8)

        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

        for mask, color in zip(masks, colors):
            mask = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
        return image


sam = SamSingleton()
