"""Predict command group and command handlers for the satellit CLI."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from satellit_sam.prompts import parse_bbox_prompts, parse_point_prompts
from satellit_sam.workflows.predict import predict_image_masks

app = typer.Typer(no_args_is_help=True)


@app.command("image-masks")
def image_masks(
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
            help="Directory path where prediction outputs will be saved.",
        ),
    ] = Path("output/predict"),
    text_prompt: Annotated[
        str | None,
        typer.Option(
            "--text",
            help="Text prompt for SAM (e.g. 'tree crowns').",
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
    point_prompts: Annotated[
        list[str] | None,
        typer.Option(
            "--point",
            help=(
                "Point prompt in image coordinates as x,y. "
                "Repeat --point to provide multiple points. "
                "Current SAM3 processor support approximates point prompts "
                "as small box prompts."
            ),
        ),
    ] = None,
    model: Annotated[
        Literal["sam3", "sam2"],
        typer.Option(
            "--model",
            help="SAM model version to use: sam3 (default) or sam2.",
        ),
    ] = "sam3",
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            max=1.0,
            help="Confidence threshold for keeping predicted masks (0.0-1.0).",
        ),
    ] = 0.5,
) -> None:
    """Predict and visualize image masks from text, bbox, and/or point prompts.

    Args:
        image_path: Input image path.
        output_path: Output directory for generated artifacts.
        text_prompt: Optional text prompt.
        bbox_prompts: Optional image-space bbox prompts.
        point_prompts: Optional image-space point prompts.
        model: SAM model version selector.
        threshold: Confidence threshold for keeping predicted masks.

    Raises:
        typer.BadParameter: If prompt arguments are invalid.
    """
    try:
        parsed_bbox_prompts = parse_bbox_prompts(bbox_prompts)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--bbox") from err

    try:
        parsed_point_prompts = parse_point_prompts(point_prompts)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--point") from err

    if text_prompt is None and not parsed_bbox_prompts and not parsed_point_prompts:
        raise typer.BadParameter(
            "Provide at least one prompt via --text, --bbox, or --point.",
            param_hint="--text / --bbox / --point",
        )
    if model == "sam2" and text_prompt is not None:
        raise typer.BadParameter(
            "--text is not supported with model 'sam2'. Use --bbox and/or --point.",
            param_hint="--model / --text",
        )

    try:
        predict_image_masks(
            image_path=image_path,
            output_path=output_path,
            text_prompt=text_prompt,
            bbox_prompts=parsed_bbox_prompts,
            point_prompts=parsed_point_prompts,
            model=model,
            threshold=threshold,
        )
    except ValueError as err:
        raise typer.BadParameter(
            str(err),
            param_hint="--model / --text / --bbox / --point",
        ) from err
