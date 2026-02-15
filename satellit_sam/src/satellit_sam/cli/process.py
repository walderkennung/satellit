"""Process command group and command handlers for the satellit CLI."""

from pathlib import Path
from typing import Annotated

import typer

from satellit_sam.workflows.process import make_image_tiles

app = typer.Typer(no_args_is_help=True)


@app.command()
def tiles(
    image_path: Annotated[
        Path,
        typer.Option(
            "--image",
            help="Path to the input image file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            help="Directory path where tile outputs will be saved.",
        ),
    ] = Path("output/tiles"),
    tile_width: Annotated[
        int,
        typer.Option("--tile-width", min=1, help="Tile width in pixels."),
    ] = 2048,
    tile_height: Annotated[
        int,
        typer.Option("--tile-height", min=1, help="Tile height in pixels."),
    ] = 2048,
    overlap: Annotated[
        int,
        typer.Option(
            "--overlap",
            min=0,
            help="Overlap between neighboring tiles in pixels.",
        ),
    ] = 64,
) -> None:
    """Split one image into tiles and save them to disk.

    Args:
        image_path: Path to the input image.
        output_path: Output directory for generated tiles.
        tile_width: Tile width in pixels.
        tile_height: Tile height in pixels.
        overlap: Neighbor tile overlap in pixels.

    Raises:
        typer.BadParameter: If tiling parameters are invalid.
    """
    try:
        make_image_tiles(
            image_path=image_path,
            output_path=output_path,
            tile_width=tile_width,
            tile_height=tile_height,
            overlap=overlap,
        )
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--overlap") from err

    print(f"✓ Tiles saved to: {output_path}")
