#!/usr/bin/env python3
"""Summarize an inventory CSV by tree species."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = REPO_ROOT / (
    "data/Traunstein/inventory/original/PlotDataReport10-04-2018_1323085911.csv"
)
LATIN_TO_ENGLISH = {
    "Abies alba": "silver fir",
    "Acer campestre": "field maple",
    "Acer platanoides": "Norway maple",
    "Acer pseudoplatanus": "sycamore maple",
    "Aesculus hippocastanum": "horse chestnut",
    "Alnus glutinose": "black alder",
    "Betula": "birch",
    "Carpinus betulus": "European hornbeam",
    "Fagus sylvatica": "European beech",
    "Fraxinus excelsior": "European ash",
    "Juglans regia": "English walnut",
    "Larix decidua": "European larch",
    "Picea abies": "Norway spruce",
    "Pinus sylvestris": "Scots pine",
    "Populus": "poplar",
    "Populus tremula": "European aspen",
    "Prunus avium": "wild cherry",
    "Pseudotsuga menziesii": "Douglas fir",
    "Quercus": "oak",
    "Quercus rubra": "northern red oak",
    "Salix": "willow",
    "Sorbus aria": "whitebeam",
    "Sorbus aucuparia": "rowan",
    "Sorbus torminalis": "wild service tree",
    "Thuja plicata": "western red cedar",
    "Tilia": "lime",
    "Ulmus glabra": "wych elm",
    "Unidentified broadleaf": "unidentified broadleaf",
    "Unidentified conifer": "unidentified conifer",
    "(missing)": "(missing)",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Print a per-species summary for a semicolon-delimited inventory CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the inventory CSV. Defaults to {DEFAULT_CSV_PATH}.",
    )
    parser.add_argument(
        "--species-column",
        default="Latin",
        help="Column containing the species name.",
    )
    parser.add_argument(
        "--tree-id-column",
        default="TreeID",
        help="Column containing the tree identifier.",
    )
    parser.add_argument(
        "--stem-id-column",
        default="StemID",
        help="Column containing the stem identifier.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Show only the top N species. Use 0 to show all species.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write the summarized table as CSV.",
    )
    return parser.parse_args()


def normalize_species(raw_value: str | None) -> str:
    """Normalize a species cell value for reporting.

    Args:
        raw_value: Raw species value from the CSV row.

    Returns:
        Normalized species label.
    """
    value = (raw_value or "").strip()
    return value if value else "(missing)"


def capitalize_species_name(name: str) -> str:
    """Capitalize a species label for display.

    Args:
        name: Species label to format.

    Returns:
        Display-ready capitalized species label.
    """
    if name.startswith("(") and name.endswith(")"):
        return name
    return name.title()


def summarize_inventory(
    csv_path: Path,
    species_column: str,
    tree_id_column: str,
    stem_id_column: str,
) -> tuple[pd.DataFrame, int]:
    """Load the inventory CSV and aggregate counts by species.

    Args:
        csv_path: Path to the semicolon-delimited inventory CSV.
        species_column: Column containing species labels.
        tree_id_column: Column containing tree IDs.
        stem_id_column: Column containing stem IDs.

    Returns:
        Species summary table plus the total number of records.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        ValueError: If a required column is missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Inventory CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False)
    required_columns = {species_column, tree_id_column, stem_id_column}
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing_text}")

    dataframe[species_column] = dataframe[species_column].map(normalize_species)
    dataframe[tree_id_column] = dataframe[tree_id_column].str.strip()
    dataframe[stem_id_column] = dataframe[stem_id_column].str.strip()

    summary = (
        dataframe.groupby(species_column, dropna=False)
        .agg(
            records=(species_column, "size"),
            unique_trees=(tree_id_column, lambda series: series[series != ""].nunique()),
            unique_stems=(stem_id_column, lambda series: series[series != ""].nunique()),
        )
        .reset_index()
        .rename(columns={species_column: "latin_name"})
    )
    total_records = int(summary["records"].sum())
    summary["share"] = summary["records"] / total_records * 100.0
    summary["english_name"] = summary["latin_name"].map(
        lambda name: LATIN_TO_ENGLISH.get(name, name)
    )
    summary = summary.sort_values(
        by=["records", "latin_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    return summary, total_records


def format_summary_for_output(species_stats: pd.DataFrame) -> pd.DataFrame:
    """Prepare a display-ready summary table.

    Args:
        species_stats: Raw species summary rows.

    Returns:
        Copy of the summary with display column names and formatting applied.
    """
    formatted = species_stats.copy()
    formatted["latin_name"] = formatted["latin_name"].map(capitalize_species_name)
    formatted["english_name"] = formatted["english_name"].map(capitalize_species_name)
    formatted["share"] = formatted["share"].map(lambda value: f"{value:.2f}%")
    return formatted.rename(
        columns={
            "latin_name": "Latin name",
            "english_name": "English name",
            "records": "Records",
            "share": "Share",
            "unique_trees": "Unique trees",
            "unique_stems": "Unique stems",
        }
    )


def format_table(
    species_stats: pd.DataFrame,
    total_records: int,
) -> str:
    """Format the summary as a plain-text table.

    Args:
        species_stats: Sorted species summary rows.
        total_records: Total number of CSV records.

    Returns:
        Formatted text table.
    """
    display_stats = format_summary_for_output(species_stats)
    headers = tuple(str(column) for column in display_stats.columns)
    rows = []
    for row in display_stats.itertuples(index=False, name=None):
        rows.append(tuple(str(value) for value in row))

    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    )
    lines.append("")
    lines.append(f"Total records: {total_records}")
    return "\n".join(lines)


def main() -> int:
    """Run the command-line entry point.

    Returns:
        Process exit code.
    """
    args = parse_args()
    species_stats, total_records = summarize_inventory(
        csv_path=args.csv_path,
        species_column=args.species_column,
        tree_id_column=args.tree_id_column,
        stem_id_column=args.stem_id_column,
    )

    if args.limit > 0:
        species_stats = species_stats.head(args.limit)

    if args.output_csv is not None:
        output_csv = args.output_csv
        if not output_csv.is_absolute():
            output_csv = Path.cwd() / output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        format_summary_for_output(species_stats).to_csv(output_csv, index=False)

    print(format_table(species_stats=species_stats, total_records=total_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
