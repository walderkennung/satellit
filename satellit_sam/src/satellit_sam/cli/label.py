"""Label command group and command handlers for the satellit CLI."""

from pathlib import Path
from typing import Annotated

import typer

from satellit_sam.core.allometry import CrownModel, DbhUnit
from satellit_sam.prompts import load_weak_label_bboxes, parse_bbox_prompts
from satellit_sam.workflows.label.by_bounding_box import (
    make_labels_by_bounding_box,
)
from satellit_sam.workflows.label.validate_predictions import (
    validate_sam3_predictions,
)
from satellit_sam.workflows.label.weak import make_weak_labels

app = typer.Typer(no_args_is_help=True)


@app.command()
def weak(
    image_tif: Annotated[
        Path,
        typer.Option("--image-tif", help="Path to orthophoto GeoTIFF."),
    ],
    inventory_csv: Annotated[
        Path | None,
        typer.Option(
            "--inventory-csv", help="Path to inventory CSV (semicolon-delimited)."
        ),
    ] = None,
    inventory_shp: Annotated[
        Path | None,
        typer.Option(
            "--inventory-shp", help="Path to inventory ESRI Shapefile (.shp)."
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", help="Output directory for labels and optional tiles."
        ),
    ] = ...,
    tile_size: Annotated[
        int,
        typer.Option("--tile-size", help="Tile size in px."),
    ] = 1024,
    overlap: Annotated[
        int,
        typer.Option("--overlap", help="Tile overlap in px."),
    ] = 128,
    x_field: Annotated[
        str,
        typer.Option(
            "--x-field",
            help="Inventory field containing x coordinate (CSV or SHP attribute).",
        ),
    ] = "PX",
    y_field: Annotated[
        str,
        typer.Option(
            "--y-field",
            help="Inventory field containing y coordinate (CSV or SHP attribute).",
        ),
    ] = "PY",
    tree_id_field: Annotated[
        str,
        typer.Option("--tree-id-field", help="Inventory field containing tree id."),
    ] = "TreeID",
    species_field: Annotated[
        str,
        typer.Option("--species-field", help="Inventory field containing species."),
    ] = "Latin",
    status_field: Annotated[
        str,
        typer.Option("--status-field", help="Inventory field containing status."),
    ] = "Status",
    status_filter: Annotated[
        str,
        typer.Option(
            "--status-filter",
            help="Only keep rows with this status (case-insensitive). Empty disables filter.",
        ),
    ] = "alive",
    dbh_field: Annotated[
        str,
        typer.Option(
            "--dbh-field", help="Inventory field containing diameter at breast height."
        ),
    ] = "DBH",
    dbh_unit: Annotated[
        DbhUnit,
        typer.Option("--dbh-unit", help="Unit of DBH values in CSV."),
    ] = "mm",
    default_crown_radius_m: Annotated[
        float,
        typer.Option(
            "--default-crown-radius-m",
            help=(
                "Fallback crown radius in meters when DBH is unavailable "
                "(common for SHP-only input)."
            ),
        ),
    ] = 2.5,
    min_dbh_cm: Annotated[
        float,
        typer.Option(
            "--min-dbh-cm", help="Exclude trees with DBH below this threshold (in cm)."
        ),
    ] = 0.0,
    max_dbh_cm: Annotated[
        float,
        typer.Option(
            "--max-dbh-cm",
            help="Exclude trees with DBH above this threshold (in cm). Use 0 to disable.",
        ),
    ] = 0.0,
    deduplicate_tree_id: Annotated[
        bool,
        typer.Option(
            "--deduplicate-tree-id",
            is_flag=True,
            help="Keep only one row per tree id (highest DBH).",
        ),
    ] = False,
    crown_model: Annotated[
        CrownModel,
        typer.Option("--crown-model", help="Crown radius model."),
    ] = "linear",
    linear_factor_m_per_cm: Annotated[
        float,
        typer.Option(
            "--linear-factor-m-per-cm",
            help="Linear model slope: crown_radius_m = intercept + factor * dbh_cm.",
        ),
    ] = 0.08,
    linear_intercept_m: Annotated[
        float,
        typer.Option("--linear-intercept-m", help="Linear model intercept in meters."),
    ] = 0.0,
    power_a: Annotated[
        float,
        typer.Option(
            "--power-a", help="Power model factor: crown_radius_m = a * (dbh_cm ^ b)."
        ),
    ] = 0.15,
    power_b: Annotated[
        float,
        typer.Option("--power-b", help="Power model exponent."),
    ] = 0.8,
    min_crown_radius_m: Annotated[
        float,
        typer.Option("--min-crown-radius-m", help="Lower clamp for crown radius."),
    ] = 0.5,
    max_crown_radius_m: Annotated[
        float,
        typer.Option("--max-crown-radius-m", help="Upper clamp for crown radius."),
    ] = 15.0,
    bbox_padding_px: Annotated[
        float,
        typer.Option(
            "--bbox-padding-px",
            help="Extra padding (in px) added to each weak-label crown bbox.",
        ),
    ] = 4.0,
    export_visualizations: Annotated[
        bool,
        typer.Option(
            "--export-visualizations",
            is_flag=True,
            help="Generate labeling visualization PNGs.",
        ),
    ] = False,
) -> None:
    """Generate weak labels from tree inventory data.

    Args:
        image_tif: Orthophoto GeoTIFF path.
        inventory_csv: Optional inventory CSV path.
        inventory_shp: Optional inventory shapefile path.
        output_dir: Output directory for generated artifacts.
        tile_size: Tile size in pixels.
        overlap: Tile overlap in pixels.
        x_field: Inventory x-coordinate field.
        y_field: Inventory y-coordinate field.
        tree_id_field: Inventory tree id field.
        species_field: Inventory species field.
        status_field: Inventory status field.
        status_filter: Optional status filter.
        dbh_field: Inventory DBH field.
        dbh_unit: DBH input unit.
        default_crown_radius_m: Fallback radius for missing DBH.
        min_dbh_cm: Minimum DBH filter.
        max_dbh_cm: Maximum DBH filter.
        deduplicate_tree_id: Keep one row per tree id.
        crown_model: Crown radius model.
        linear_factor_m_per_cm: Linear model slope.
        linear_intercept_m: Linear model intercept.
        power_a: Power model factor.
        power_b: Power model exponent.
        min_crown_radius_m: Lower crown-radius clamp.
        max_crown_radius_m: Upper crown-radius clamp.
        bbox_padding_px: Extra bbox padding in pixels.
        export_visualizations: Whether visualization files are exported.

    Raises:
        typer.BadParameter: If inventory sources are misconfigured.
    """
    if inventory_csv is not None and inventory_shp is not None:
        raise typer.BadParameter(
            "Provide only one of --inventory-csv or --inventory-shp.",
            param_hint="--inventory-csv / --inventory-shp",
        )
    if inventory_csv is None and inventory_shp is None:
        raise typer.BadParameter(
            "Provide either --inventory-csv or --inventory-shp.",
            param_hint="--inventory-csv / --inventory-shp",
        )

    make_weak_labels(
        output_dir=output_dir,
        image_tif=image_tif,
        tile_size=tile_size,
        tile_overlap=overlap,
        min_dbh_cm=min_dbh_cm,
        max_dbh_cm=max_dbh_cm,
        crown_model=crown_model,
        export_visualizations=export_visualizations,
        inventory_csv=inventory_csv,
        inventory_shp=inventory_shp,
        x_field=x_field,
        y_field=y_field,
        tree_id_field=tree_id_field,
        species_field=species_field,
        status_field=status_field,
        status_filter=status_filter,
        dbh_field=dbh_field,
        dbh_unit=dbh_unit,
        deduplicate_tree_id=deduplicate_tree_id,
        default_crown_radius_m=default_crown_radius_m,
        linear_factor_m_per_cm=linear_factor_m_per_cm,
        linear_intercept_m=linear_intercept_m,
        power_a=power_a,
        power_b=power_b,
        min_crown_radius_m=min_crown_radius_m,
        max_crown_radius_m=max_crown_radius_m,
        bbox_padding_px=bbox_padding_px,
    )


@app.command("validate-predictions")
def validate_predictions(
    image_tif: Annotated[
        Path,
        typer.Option("--image-tif", help="Path to orthophoto GeoTIFF."),
    ],
    predictions_npz: Annotated[
        Path,
        typer.Option(
            "--predictions-npz",
            help="Path to SAM3 output NPZ (for example masks/image_masks.npz).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    inventory_csv: Annotated[
        Path | None,
        typer.Option(
            "--inventory-csv", help="Path to inventory CSV (semicolon-delimited)."
        ),
    ] = None,
    inventory_shp: Annotated[
        Path | None,
        typer.Option(
            "--inventory-shp", help="Path to inventory ESRI Shapefile (.shp)."
        ),
    ] = None,
    output_csv: Annotated[
        Path,
        typer.Option(
            "--output-csv",
            help="Output CSV path for validation results.",
        ),
    ] = Path("output/validation/label_validation.csv"),
    x_field: Annotated[
        str,
        typer.Option(
            "--x-field",
            help="Inventory field containing x coordinate (CSV offset or SHP attribute).",
        ),
    ] = "PX",
    y_field: Annotated[
        str,
        typer.Option(
            "--y-field",
            help="Inventory field containing y coordinate (CSV offset or SHP attribute).",
        ),
    ] = "PY",
    tree_id_field: Annotated[
        str,
        typer.Option("--tree-id-field", help="Inventory field containing tree id."),
    ] = "TreeID",
    stem_id_field: Annotated[
        str,
        typer.Option(
            "--stem-id-field",
            help=(
                "Inventory field containing stem id. "
                "Fallback chain: explicit field -> stemtag -> tree_id."
            ),
        ),
    ] = "StemTag",
    species_field: Annotated[
        str,
        typer.Option("--species-field", help="Inventory field containing species."),
    ] = "Latin",
    status_field: Annotated[
        str,
        typer.Option("--status-field", help="Inventory field containing status."),
    ] = "Status",
    status_filter: Annotated[
        str,
        typer.Option(
            "--status-filter",
            help="Only keep rows with this status (case-insensitive). Empty disables filter.",
        ),
    ] = "alive",
    dbh_field: Annotated[
        str,
        typer.Option(
            "--dbh-field", help="Inventory field containing diameter at breast height."
        ),
    ] = "DBH",
    dbh_unit: Annotated[
        DbhUnit,
        typer.Option("--dbh-unit", help="Unit of DBH values in inventory."),
    ] = "mm",
    min_dbh_cm: Annotated[
        float,
        typer.Option(
            "--min-dbh-cm", help="Exclude trees with DBH below this threshold (in cm)."
        ),
    ] = 0.0,
    max_dbh_cm: Annotated[
        float,
        typer.Option(
            "--max-dbh-cm",
            help="Exclude trees with DBH above this threshold (in cm). Use 0 to disable.",
        ),
    ] = 0.0,
    deduplicate_tree_id: Annotated[
        bool,
        typer.Option(
            "--deduplicate-tree-id",
            is_flag=True,
            help="Keep only one row per tree id (highest DBH).",
        ),
    ] = False,
) -> None:
    """Validate SAM3 strong labels against inventory stem positions.

    Args:
        image_tif: Orthophoto GeoTIFF path.
        predictions_npz: SAM3 output NPZ path.
        inventory_csv: Optional inventory CSV path.
        inventory_shp: Optional inventory shapefile path.
        output_csv: CSV path for validation rows.
        x_field: Inventory x-coordinate field.
        y_field: Inventory y-coordinate field.
        tree_id_field: Inventory tree id field.
        stem_id_field: Inventory stem id field.
        species_field: Inventory species field.
        status_field: Inventory status field.
        status_filter: Optional status filter.
        dbh_field: Inventory DBH field.
        dbh_unit: DBH input unit.
        min_dbh_cm: Minimum DBH filter.
        max_dbh_cm: Maximum DBH filter.
        deduplicate_tree_id: Keep one row per tree id.

    Raises:
        typer.BadParameter: If inventory sources or prediction payload are invalid.
    """
    if inventory_csv is not None and inventory_shp is not None:
        raise typer.BadParameter(
            "Provide only one of --inventory-csv or --inventory-shp.",
            param_hint="--inventory-csv / --inventory-shp",
        )
    if inventory_csv is None and inventory_shp is None:
        raise typer.BadParameter(
            "Provide either --inventory-csv or --inventory-shp.",
            param_hint="--inventory-csv / --inventory-shp",
        )

    try:
        validate_sam3_predictions(
            image_tif=image_tif,
            predictions_npz=predictions_npz,
            output_csv=output_csv,
            inventory_csv=inventory_csv,
            inventory_shp=inventory_shp,
            x_field=x_field,
            y_field=y_field,
            tree_id_field=tree_id_field,
            stem_id_field=stem_id_field,
            species_field=species_field,
            status_field=status_field,
            status_filter=status_filter,
            dbh_field=dbh_field,
            dbh_unit=dbh_unit,
            min_dbh_cm=min_dbh_cm,
            max_dbh_cm=max_dbh_cm,
            deduplicate_tree_id=deduplicate_tree_id,
        )
    except ValueError as err:
        raise typer.BadParameter(
            str(err),
            param_hint=(
                "--predictions-npz / --inventory-csv / --inventory-shp "
                "/ --image-tif / --dbh-field / --dbh-unit / --min-dbh-cm / --max-dbh-cm"
            ),
        ) from err


@app.command()
def by_bounding_boxes(
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
    weak_labels_csv: Annotated[
        Path | None,
        typer.Option(
            "--weak-labels-csv",
            help=(
                "Path to labels_tiles.csv generated by 'label weak'. "
                "Uses stored per-tree tile-local bboxes as box prompts."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """Generate label overlays from explicit or weak-label box prompts.

    Deprecated:
        Use ``predict image-masks`` to persist strong labels as NPZ artifacts:
        ``predict image-masks --bbox ...`` or
        ``predict image-masks --weak-labels-csv ...``.

    Args:
        image_path: Input image path.
        tile_size: Tile size in pixels.
        overlap: Tile overlap in pixels.
        output_path: Output directory for generated tiles.
        bbox_prompts: Optional image-space box prompts.
        weak_labels_csv: Optional weak-label CSV with tile-local prompts.

    Raises:
        typer.BadParameter: If prompt arguments are invalid.
    """
    typer.echo(
        "DEPRECATED: `label by-bounding-boxes` is deprecated and does not persist "
        "canonical strong-label masks. "
        "Use `predict image-masks --bbox ...` or "
        "`predict image-masks --weak-labels-csv ...`."
    )

    try:
        parsed_bbox_prompts = parse_bbox_prompts(bbox_prompts)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--bbox") from err

    weak_label_bboxes_by_tile: (
        dict[str, list[tuple[float, float, float, float]]] | None
    ) = None
    if weak_labels_csv is not None:
        try:
            weak_label_bboxes_by_tile = load_weak_label_bboxes(weak_labels_csv)
        except ValueError as err:
            raise typer.BadParameter(str(err), param_hint="--weak-labels-csv") from err

    make_labels_by_bounding_box(
        image_path=image_path,
        tile_size=tile_size,
        overlap=overlap,
        output_path=output_path,
        bbox_prompts=parsed_bbox_prompts,
        weak_label_bboxes_by_tile=weak_label_bboxes_by_tile,
    )
