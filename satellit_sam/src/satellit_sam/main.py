import asyncio
from pathlib import Path

from tqdm import tqdm

from .prompts import parse_tile_origin, project_bboxes_to_tile
from .workflows.process import (
    Image,
    create_heightmap_from_las,
    tile_image,
)


def predict_masks(
    image_path: Path,
    tile_size: int,
    overlap: int,
    output_path: Path,
    text_prompt: str | None,
    bbox_prompts: list[tuple[float, float, float, float]],
) -> None:
    asyncio.run(
        predict_masks_async(
            image_path=image_path,
            tile_size=tile_size,
            overlap=overlap,
            output_path=output_path,
            text_prompt=text_prompt,
            bbox_prompts=bbox_prompts,
        )
    )


async def predict_masks_async(
    image_path: Path,
    tile_size: int,
    overlap: int,
    output_path: Path,
    text_prompt: str | None,
    bbox_prompts: list[tuple[float, float, float, float]],
) -> None:
    from .sam3 import sam

    sam.print_debug_info()

    print(f"Loading image from: {image_path}")
    image = Image.load(str(image_path))

    las_file = Path("../data/Traunstein/2018/inventory_plot_normalized.las")

    heightmap = create_heightmap_from_las(
        las_file, width=image.size[0], height=image.size[1], method="max"
    )
    print(f"Height map shape: {heightmap.shape}")
    print(f"Height range: {heightmap.z_range}")
    print(f"Resolution: {heightmap.resolution}m per pixel")

    heightmap_path = f"{output_path}/heightmap.png"
    heightmap.save(heightmap_path)
    print(f"Saved to: {heightmap_path}")

    image.data = image.data * heightmap.to_rgb()
    image.save(f"{output_path}/heightmap_overlay.png")

    print(f"Tiling image (tile_size={tile_size}, overlap={overlap})...")
    tiling_dir = tile_image(
        image, tile_size=tile_size, overlap=overlap, output_path=str(output_path)
    )
    tiling_dir.save_to_dir()

    if text_prompt and bbox_prompts:
        print(
            "Generating masks with text prompt "
            f"'{text_prompt}' and {len(bbox_prompts)} bbox prompt(s)..."
        )
    elif text_prompt:
        print(f"Generating masks with text prompt: '{text_prompt}'...")
    else:
        print(f"Generating masks with {len(bbox_prompts)} bbox prompt(s)...")

    with tqdm(total=len(tiling_dir), desc="Processing tiles", unit="tile") as pbar:
        for tile in tiling_dir:
            tile_bboxes = None
            if bbox_prompts:
                tile_origin = parse_tile_origin(tile.path)
                projected_bboxes = project_bboxes_to_tile(
                    image_bboxes=bbox_prompts,
                    tile_origin=tile_origin,
                    tile_size=tile.image.size,
                )
                if projected_bboxes:
                    tile_bboxes = projected_bboxes
                elif text_prompt is None:
                    tiling_dir.save_annotated_tile(tile, tile.image.copy())
                    pbar.update(1)
                    continue

            ann_image = sam.predict(tile.image, text=text_prompt, boxes=tile_bboxes)
            tiling_dir.save_annotated_tile(tile, ann_image)
            pbar.update(1)

    print(f"✓ Processing complete! Results saved to: {output_path}")


def main(argv: list[str] | None = None) -> None:
    from .cli import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main()
