import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import satellit_sam.pytorch as pytorch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random(3)
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def process_tiles(
    image,
    sam,
    output_dir="tiles_output",
    initial_offset=[0, 0],
    max_tiles=10,
    tile_size=1024,
    overlap=256,
):
    """Process large image in tiles and save each tile result."""
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]
    mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=128)

    tile_idx = 0
    for y in range(initial_offset[1] * tile_size, h, tile_size - overlap):
        for x in range(initial_offset[0] * tile_size, w, tile_size - overlap):
            if tile_idx < max_tiles:
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tile = image[y:y_end, x:x_end]

                masks = mask_gen.generate(tile)

                # Save tile result
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(tile)
                show_anns(masks, ax)
                ax.axis("off")
                plt.savefig(
                    f"{output_dir}/tile_{tile_idx:04d}_x{x}_y{y}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=150,
                )
                plt.close(fig)  # Free memory!

                print(f"Processed tile {tile_idx} ({x}, {y}) - {len(masks)} masks")
                tile_idx += 1

                # Clear masks from memory
                del masks


print("Reading image...")
image = cv2.imread("data/orthophoto_wgs84_utm33n_agg200mm.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # SAM expects RGB

pytorch_instance = pytorch.init()

print("Loading model...")
sam = sam_model_registry["vit_h"](checkpoint="models/sam/sam_vit_h_4b8939.pth")
sam.to(device=pytorch_instance.device)

print("Generating masks...")
process_tiles(
    image,
    sam,
    output_dir="output/tiles",
    initial_offset=[2, 2],
    max_tiles=2,
    tile_size=1024,
    overlap=256,
)
