import os
import time
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from satellit_sam.sam3 import sam
from tqdm import tqdm
from typing_extensions import AsyncIterable


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


def reconstruct_from_tiles(
    original_shape: tuple[int, int, int],
    tiles_dir: str = "tiles_output",
    tile_size: int = 1024,
    overlap: int = 256,
) -> Image.Image:
    """
    Reconstruct the full image from processed tiles.

    Args:
        original_shape: Shape of the original image (height, width, channels).
        tiles_dir: Directory containing the processed tile images.
        tile_size: Size of each tile.
        overlap: Overlap between adjacent tiles.

    Returns:
        Reconstructed image as a numpy array.
    """
    h, w, c = original_shape
    reconstructed = np.zeros((h, w, 4), dtype=np.float32)  # RGBA for blending
    weight_map = np.zeros((h, w), dtype=np.float32)

    # Create a weight mask for blending (higher weight in center, lower at edges)
    blend_mask = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap > 0:
        # Create linear ramp for overlap regions
        ramp = np.linspace(0, 1, overlap)
        # Apply ramp to edges
        blend_mask[:overlap, :] *= ramp[:, np.newaxis]  # top edge
        blend_mask[-overlap:, :] *= ramp[::-1, np.newaxis]  # bottom edge
        blend_mask[:, :overlap] *= ramp[np.newaxis, :]  # left edge
        blend_mask[:, -overlap:] *= ramp[::-1][np.newaxis, :]  # right edge

    # Find all tile files and extract their positions
    tile_files = []
    for filename in os.listdir(tiles_dir):
        if filename.startswith("tile_") and filename.endswith(".png"):
            # Parse position from filename: tile_0000_x0_y0.png
            parts = filename.replace(".png", "").split("_")
            x = int(parts[2][1:])  # Remove 'x' prefix
            y = int(parts[3][1:])  # Remove 'y' prefix
            tile_files.append((filename, x, y))

    # Sort by position for consistent processing
    tile_files.sort(key=lambda t: (t[2], t[1]))

    for filename, x, y in tile_files:
        tile_path = os.path.join(tiles_dir, filename)
        tile = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)

        if tile is None:
            print(f"Warning: Could not read {tile_path}")
            continue

        # Convert BGR(A) to RGB(A)
        if tile.shape[2] == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            # Add alpha channel
            tile = np.dstack([tile, np.full(tile.shape[:2], 255, dtype=np.uint8)])
        else:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2RGBA)

        tile = tile.astype(np.float32) / 255.0

        # Calculate target tile dimensions (may be smaller at edges of image)
        x_end = min(x + tile_size, w)
        y_end = min(y + tile_size, h)
        target_w = x_end - x
        target_h = y_end - y

        # Get the appropriate portion of the blend mask for target dimensions
        current_blend = blend_mask[:target_h, :target_w]

        # Resize tile to match target dimensions (tiles from matplotlib may have different resolution due to DPI)
        tile_h, tile_w = tile.shape[:2]
        if tile_h != target_h or tile_w != target_w:
            tile = cv2.resize(
                tile, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        # Apply weighted blending
        for c_idx in range(4):
            reconstructed[y:y_end, x:x_end, c_idx] += tile[:, :, c_idx] * current_blend
        weight_map[y:y_end, x:x_end] += current_blend

    # Normalize by weight map to complete the blending
    weight_map = np.maximum(weight_map, 1e-6)  # Avoid division by zero
    for c_idx in range(4):
        reconstructed[:, :, c_idx] /= weight_map

    # Convert back to uint8
    reconstructed = (reconstructed * 255).astype(np.uint8)

    return Image.fromarray(reconstructed)


def _get_cached_tiles(output_dir: str) -> set[tuple[int, int]]:
    """Get set of (x, y) positions for already processed tiles."""
    cached = set()
    if not os.path.exists(output_dir):
        return cached

    for filename in os.listdir(output_dir):
        if filename.startswith("tile_") and filename.endswith(".png"):
            # Parse position from filename: tile_0000_x0_y0.png
            parts = filename.replace(".png", "").split("_")
            x = int(parts[2][1:])  # Remove 'x' prefix
            y = int(parts[3][1:])  # Remove 'y' prefix
            cached.add((x, y))

    return cached


@dataclass
class ProcessInfo:
    original_shape: tuple[int, int, int]
    total_prediction_time: float
    tiles_processed: int
    tiles_skipped: int
    output_dir: str
    tile_overlap: int
    tile_size: int


@dataclass
class TileFile:
    FILE_TYPES = ("png", "npz")

    position: tuple[int, int]  # (x, y)
    tile_idx: int
    tile_size: int
    overlap: int
    output_dir: str

    def filename(self, file_type: Literal["png"] | Literal["npz"]) -> str:
        return f"{self.output_dir}/tile_{self.tile_idx:04d}_x{self.position[0]}_y{self.position[1]}_overlap{self.overlap}.{file_type}"

    @staticmethod
    def parse_filename(filepath: str) -> "TileFile | None":
        fp = PurePath(filepath)
        suffix = fp.suffix[1:]  # remove dot
        if not fp.name.startswith("tile_") and suffix not in TileFile.FILE_TYPES:
            return None

        # Parse position from filename: tile_0000_x0_y0_overlap0.png
        parts = fp.stem.split("_")
        tile_idx = int(parts[1])
        x = int(parts[2][1:])  # Remove 'x' prefix
        y = int(parts[3][1:])  # Remove 'y' prefix
        overlap = int(parts[4][7:])  # Remove 'overlap' prefix
        return TileFile(
            position=(x, y),
            tile_idx=tile_idx,
            tile_size=0,
            overlap=overlap,
            output_dir=str(fp.parent),
        )


@dataclass
class TileInfo:
    prediction_time: float
    total_prediction_time: float
    total_tiles: int
    tiles_processed: int
    tiles_skipped: int
    number_of_masks: int


async def process_tiles(
    image,
    output_dir="tiles_output",
    initial_offset=[0, 0],
    max_tiles=None,
    tile_size=1024,
    overlap=256,
    use_cache=True,
    prompt: str | None = None,
) -> AsyncIterable[TileInfo | ProcessInfo]:
    """Process large image in tiles with SAM (Segment Anything Model) and save each tile result.

    Args:
        image: Input image as a numpy array (RGB format).
        sam: Loaded SAM model instance.
        output_dir: Directory to save processed tile images.
        initial_offset: Starting offset [x, y] in tile units for processing.
        max_tiles: Maximum number of tiles to process (None for all tiles).
        tile_size: Size of each square tile in pixels.
        overlap: Overlap between adjacent tiles in pixels for seamless reconstruction.
        use_cache: If True, skip tiles that already exist in output_dir.
        prompt: Optional prompt for SAM.

    Returns:
        Dictionary containing reconstruction information:
            - original_shape: Shape of the input image
            - tile_size: Size of tiles used
            - overlap: Overlap between tiles
            - output_dir: Directory where tiles were saved
            - total_prediction_time: Total time spent on SAM predictions
            - tiles_processed: Number of tiles processed
            - tiles_skipped: Number of tiles skipped (cached)
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]

    # Get cached tiles if caching is enabled
    cached_tiles = _get_cached_tiles(output_dir) if use_cache else set()
    if cached_tiles:
        print(f"Found {len(cached_tiles)} cached tiles in {output_dir}")

    # Calculate all tile positions
    tile_positions = []
    for y in range(initial_offset[1] * tile_size, h, tile_size - overlap):
        for x in range(initial_offset[0] * tile_size, w, tile_size - overlap):
            if max_tiles is None or len(tile_positions) < max_tiles:
                tile_positions.append((x, y))

    # Separate cached and uncached tiles
    tiles_to_process = [
        (i, x, y)
        for i, (x, y) in enumerate(tile_positions)
        if (x, y) not in cached_tiles
    ]
    tiles_skipped = len(tile_positions) - len(tiles_to_process)

    if tiles_skipped > 0:
        print(f"Skipping {tiles_skipped} cached tiles")

    total_prediction_time = 0.0
    tiles_processed = 0

    for tile_idx, x, y in tiles_to_process:
        x_end = min(x + tile_size, w)
        y_end = min(y + tile_size, h)
        tile = image[y:y_end, x:x_end]
        tile_image = Image.fromarray(tile)

        timestamp_start = time.perf_counter()

        result = sam.predict(tile_image, text=prompt or "tree crowns")

        prediction_time = time.perf_counter() - timestamp_start
        total_prediction_time += prediction_time

        tile_file = TileFile(
            position=(x, y),
            tile_idx=tile_idx,
            tile_size=tile_size,
            overlap=overlap,
            output_dir=output_dir,
        )
        result.save(tile_file.filename("npz"))

        overlay = sam.overlay_masks(tile_image, result.masks)
        overlay.save(tile_file.filename("png"))

        tiles_processed += 1

        yield TileInfo(
            total_tiles=len(tiles_to_process),
            prediction_time=prediction_time,
            total_prediction_time=total_prediction_time,
            tiles_processed=tiles_processed,
            tiles_skipped=tiles_skipped,
            number_of_masks=len(result.masks),
        )

    # Return info needed for reconstruction
    yield ProcessInfo(
        original_shape=image.shape,
        tile_size=tile_size,
        tile_overlap=overlap,
        output_dir=output_dir,
        total_prediction_time=total_prediction_time,
        tiles_processed=tiles_processed,
        tiles_skipped=tiles_skipped,
    )
    return


async def main():
    print("Loading image...")
    # Load image
    image_path = "../data/orthophoto_wgs84_utm33n_agg200mm.tif"
    image = Image.open(image_path).convert("RGB")

    print("Generating masks...")
    process_info = None
    with tqdm(total=float("inf"), desc="Processing tiles", unit="tile") as pbar:
        async for info in process_tiles(
            np.array(image),
            output_dir="output/tiles",
            tile_size=256,
            overlap=32,
            use_cache=True,
            prompt="tree crowns",
        ):
            if isinstance(info, TileInfo):
                if info.total_tiles != pbar.total:
                    pbar.total = info.total_tiles
                pbar.set_postfix(
                    masks=info.number_of_masks,
                    time=f"{info.prediction_time:.2f}s",
                    avg=f"{info.total_prediction_time / info.tiles_processed:.2f}s",
                    skipped=info.tiles_skipped,
                )
                pbar.update(1)
            if isinstance(info, ProcessInfo):
                process_info = info
                print(f"\nTotal prediction time: {info.total_prediction_time:.2f}s")
                print(
                    f"Tiles processed: {info.tiles_processed}, Tiles skipped (cached): {info.tiles_skipped}"
                )
                if info.tiles_processed > 0:
                    print(
                        f"Average time per tile: {info.total_prediction_time / info.tiles_processed:.2f}s"
                    )

        if process_info is None:
            print("Tile processing did not complete successfully.")
            return

    print("Reconstructing full image from tiles...")
    reconstructed_image = reconstruct_from_tiles(
        original_shape=process_info.original_shape,
        tiles_dir=process_info.output_dir,
        tile_size=process_info.tile_size,
        overlap=process_info.tile_overlap,
    )
    reconstructed_image.save("output/reconstructed.png")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
