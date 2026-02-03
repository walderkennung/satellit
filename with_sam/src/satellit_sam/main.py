import os
import time
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm
from typing_extensions import AsyncIterable

from satellit_sam.sam3 import sam
from src.satellit_sam.image_processing import Image, tile_image


async def main():
    print("Loading image...")
    image_path = "../data/orthophoto_wgs84_utm33n_agg200mm.tif"
    image = Image.load(image_path)

    tiling_dir = tile_image(
        image, tile_size=256, overlap=32, output_path="output/test_tiles"
    )
    tiling_dir.save_to_dir()

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
