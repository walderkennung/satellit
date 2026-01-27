from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


@dataclass
class PredictionResult:
    masks: list[Image.Image]
    boxes: torch.Tensor  # (batch_size, num_queries, 4)
    scores: torch.Tensor  # (batch_size, num_queries)

    def save(self, output_path: str):
        np.savez_compressed(
            output_path,
            masks=np.array(self.masks),
            boxes=self.boxes.cpu().numpy(),
            scores=self.scores.cpu().numpy(),
        )


class SamSingleton:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def predict(
        self,
        image: Image.Image,
        text: str,
        threshold: float = 0.01,
        mask_threshold: float = 0.01,
    ) -> PredictionResult:
        # Segment using text prompt
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = outputs["pred_logits"]
        result = PredictionResult(
            masks=[],
            boxes=outputs["pred_boxes"],
            scores=scores if scores is not None else torch.Tensor([]),
        )

        print("Outputs:")
        print(repr(outputs))

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )

        print("Results:")
        print(repr(results))

        masks = []
        if "masks" in results[0]:
            for i, mask_tensor in enumerate(results[0]["masks"]):
                mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np)
                masks.append(mask_pil)
            result.masks = masks

        return result

    def overlay_masks(self, image, masks):
        image = image.convert("RGBA")
        masks = 255 * np.array(masks).astype(np.uint8)

        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

        for mask, color in zip(masks, colors):
            mask = mask.squeeze()
            if mask.ndim == 1:
                mask_img = Image.fromarray(mask.reshape(1, -1))
            elif mask.ndim == 2:
                mask_img = Image.fromarray(mask)
            else:
                print(f"Unexpected mask shape: {mask.shape}")
                continue
            mask_img = mask_img.resize(image.size, Image.NEAREST)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask_img.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
        return image


sam = SamSingleton()
