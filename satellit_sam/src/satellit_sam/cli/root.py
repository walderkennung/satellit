from pathlib import Path
from typing import Annotated

import typer

from ..prompts import parse_bbox_prompts
from . import label as label_cli

app = typer.Typer(no_args_is_help=True)
app.add_typer(label_cli.app, name="label")


@app.command()
def image_processing(
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
) -> None:
    """Process satellite imagery using SAM (Segment Anything Model)."""
    normalized_text_prompt = text_prompt.strip() if text_prompt else None
    if normalized_text_prompt == "":
        normalized_text_prompt = None

    try:
        parsed_bbox_prompts = parse_bbox_prompts(bbox_prompts)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--bbox") from err

    if normalized_text_prompt is None and not parsed_bbox_prompts:
        normalized_text_prompt = "trees"

    from ..main import predict_masks

    predict_masks(
        image_path=image_path,
        tile_size=tile_size,
        overlap=overlap,
        output_path=output_path,
        text_prompt=normalized_text_prompt,
        bbox_prompts=parsed_bbox_prompts,
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        app()
        return
    app(args=argv, prog_name="satellit_sam")


if __name__ == "__main__":
    main()
