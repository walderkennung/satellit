import matplotlib
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def predict(image: Image.Image):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Sam3Processor.from_pretrained("facebook/sam3").to(device)
    model = Sam3Model.from_pretrained("facebook/sam3")

    # Segment using text prompt
    inputs = processor(images=image, text="tree crowns", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    print(f"Found {len(results['masks'])} objects")
    return results
    # Results contain:
    # - masks: Binary masks resized to original image size
    # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
    # - scores: Confidence scores


def overlay_masks(image, masks):
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
