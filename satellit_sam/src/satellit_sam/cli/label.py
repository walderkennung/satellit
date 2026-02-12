from pathlib import Path
from typing import Annotated

import typer

from src.satellit_sam.core.allometry import CrownModel, DbhUnit
from src.satellit_sam.workflows.label.weak import make_weak_labels

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
    only_non_empty_tiles: Annotated[
        bool,
        typer.Option(
            "--only-non-empty-tiles",
            is_flag=True,
            help="Write only tiles that contain at least one tree.",
        ),
    ] = False,
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
    export_visualizations: Annotated[
        bool,
        typer.Option(
            "--export-visualizations",
            is_flag=True,
            help="Generate labeling visualization PNGs.",
        ),
    ] = False,
) -> None:
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
        only_non_empty_tiles=only_non_empty_tiles,
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        app()
        return
    app(args=argv, prog_name="satellit_sam")


if __name__ == "__main__":
    main()
