import asyncio
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from src.satellit_sam.image_processing import (
    Image,
    create_heightmap_from_las,
    tile_image,
)

from .prompts import parse_bbox_prompts, parse_tile_origin, project_bboxes_to_tile
from .sam3 import sam

app = typer.Typer()


@app.command()
def main(
    image_path: Annotated[
        Path,
        typer.Option(
            "--image",
            help="Path to the input image file (e.g., GeoTIFF)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    tile_size: Annotated[
        int,
        typer.Option(
            "--tile-size",
            help="Size of tiles to split the image into (in pixels)",
        ),
    ] = 2048,
    overlap: Annotated[
        int,
        typer.Option(
            "--overlap",
            help="Overlap between tiles (in pixels)",
        ),
    ] = 64,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            help="Directory path where output tiles will be saved",
        ),
    ] = Path("output/test_tiles"),
    text_prompt: Annotated[
        str | None,
        typer.Option(
            "--text-prompt",
            help=(
                "Optional text prompt for object detection (e.g., 'trees'). "
                "Defaults to 'trees' when --bbox is not provided."
            ),
        ),
    ] = None,
    bbox_prompts: Annotated[
        list[str] | None,
        typer.Option(
            "--bbox",
            help=(
                "Bounding-box prompt in image coordinates as x1,y1,x2,y2. "
                "Repeat --bbox to provide multiple boxes."
            ),
        ),
    ] = None,
):
    """
    Process satellite imagery using SAM (Segment Anything Model).

    This tool loads a satellite image, tiles it into smaller chunks,
    and generates segmentation masks based on the provided text and/or bbox prompts.
    """
    normalized_text_prompt = text_prompt.strip() if text_prompt else None
    if normalized_text_prompt == "":
        normalized_text_prompt = None

    try:
        parsed_bbox_prompts = parse_bbox_prompts(bbox_prompts)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--bbox") from err

    if normalized_text_prompt is None and not parsed_bbox_prompts:
        normalized_text_prompt = "trees"

    asyncio.run(
        async_main(
            image_path,
            tile_size,
            overlap,
            output_path,
            normalized_text_prompt,
            parsed_bbox_prompts,
        )
    )


async def async_main(
    image_path: Path,
    tile_size: int,
    overlap: int,
    output_path: Path,
    text_prompt: str | None,
    bbox_prompts: list[tuple[float, float, float, float]],
):
    sam.print_debug_info()

    print(f"Loading image from: {image_path}")
    image = Image.load(str(image_path))

    las_file = Path("../data/Traunstein/2018/inventory_plot_normalized.las")

    # Simple one-line creation and saving
    heightmap = create_heightmap_from_las(
        las_file, width=image.size[0], height=image.size[1], method="max"
    )
    print(f"Height map shape: {heightmap.shape}")
    print(f"Height range: {heightmap.z_range}")
    print(f"Resolution: {heightmap.resolution}m per pixel")

    # Save as image
    heightmap_path = f"{output_path}/heightmap.png"
    heightmap.save(heightmap_path)
    print(f"Saved to: {heightmap_path}")

    # Multiply the RGB image with the grayscale heightmap
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


if __name__ == "__main__":
    app()
