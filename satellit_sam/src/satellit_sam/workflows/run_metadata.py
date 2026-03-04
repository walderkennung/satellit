"""Helpers for writing workflow run metadata artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_metadata(
    output_dir: Path,
    *,
    image_path: Path,
    tile_size: int,
    tile_overlap: int,
    prompt: dict[str, Any] | None,
    model: str | None,
    command: str | None,
) -> Path:
    """Write one workflow metadata artifact at ``<output_dir>/metadata.json``.

    Args:
        output_dir: Root output directory for the workflow run.
        image_path: Source image path used for the run.
        tile_size: Configured tile size in pixels.
        tile_overlap: Configured tile overlap in pixels.
        prompt: Prompt payload used by the workflow, if applicable.
        model: Model identifier used by the workflow, if applicable.
        command: CLI command used to start the run, if available.

    Returns:
        Path to the written ``metadata.json`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    payload = {
        "tile": {"size": int(tile_size), "overlap": int(tile_overlap)},
        "image_path": str(image_path.resolve()),
        "prompt": prompt,
        "model": model,
        "command": command,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    return metadata_path
