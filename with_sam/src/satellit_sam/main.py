import asyncio
from pathlib import Path

import typer
from tqdm import tqdm
from typing_extensions import Annotated

from src.satellit_sam.image_processing import Image, tile_image

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
        str,
        typer.Option(
            "--text-prompt",
            help="Text prompt for object detection (e.g., 'trees')",
        ),
    ] = "trees",
):
    """
    Process satellite imagery using SAM (Segment Anything Model).

    This tool loads a satellite image, tiles it into smaller chunks,
    and generates segmentation masks based on the provided text prompt.
    """
    asyncio.run(async_main(image_path, tile_size, overlap, output_path, text_prompt))


async def async_main(
    image_path: Path,
    tile_size: int,
    overlap: int,
    output_path: Path,
    text_prompt: str,
):
    sam.print_debug_info()

    print(f"Loading image from: {image_path}")
    image = Image.load(str(image_path))

    print(f"Tiling image (tile_size={tile_size}, overlap={overlap})...")
    tiling_dir = tile_image(
        image, tile_size=tile_size, overlap=overlap, output_path=str(output_path)
    )
    tiling_dir.save_to_dir()

    print(f"Generating masks with prompt: '{text_prompt}'...")

    with tqdm(total=len(tiling_dir), desc="Processing tiles", unit="tile") as pbar:
        for tile in tiling_dir:
            ann_image = sam.predict(tile.image, text=text_prompt)
            tiling_dir.save_annotated_tile(tile, ann_image)
            pbar.update(1)

    print(f"✓ Processing complete! Results saved to: {output_path}")


if __name__ == "__main__":
    app()
