from tqdm import tqdm

from src.satellit_sam.image_processing import Image, tile_image

from .sam3 import sam


async def main():
    sam.print_debug_info()

    print("Loading image...")
    image_path = "../data/orthophoto_wgs84_utm33n_agg200mm.tif"
    image = Image.load(image_path)

    tiling_dir = tile_image(
        image, tile_size=2048, overlap=64, output_path="output/test_tiles"
    )
    tiling_dir.save_to_dir()

    print("Generating masks...")

    with tqdm(total=len(tiling_dir), desc="Processing tiles", unit="tile") as pbar:
        for tile in tiling_dir:
            ann_image = sam.predict(tile.image, text="trees")
            tiling_dir.save_annotated_tile(tile, ann_image)
            pbar.update(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
