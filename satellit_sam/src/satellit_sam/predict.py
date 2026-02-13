"""Tile-based prediction helpers for running SAM inference over large images."""

import os
import time
from dataclasses import dataclass
from pathlib import PurePath
from typing import AsyncIterable, Literal

from PIL import Image

from satellit_sam.sam3 import sam


@dataclass
class ProcessInfo:
    """Summary metrics emitted after tile processing completes."""

    original_shape: tuple[int, int, int]
    total_prediction_time: float
    tiles_processed: int
    tiles_skipped: int
    output_dir: str
    tile_overlap: int
    tile_size: int


@dataclass
class TileFile:
    """Metadata and naming helpers for tile artifacts on disk."""

    FILE_TYPES = ("png", "npz")

    position: tuple[int, int]  # (x, y)
    tile_idx: int
    tile_size: int
    overlap: int
    output_dir: str

    def filename(self, file_type: Literal["png"] | Literal["npz"]) -> str:
        """Return the output filename for the tile artifact.

        Args:
            file_type: Artifact extension, either ``png`` or ``npz``.

        Returns:
            Absolute or relative artifact path under ``output_dir``.
        """
        return f"{self.output_dir}/tile_{self.tile_idx:04d}_x{self.position[0]}_y{self.position[1]}_overlap{self.overlap}.{file_type}"

    @staticmethod
    def parse_filename(filepath: str) -> "TileFile | None":
        """Parse a tile artifact filename into a ``TileFile`` instance.

        Args:
            filepath: Tile artifact path following the naming convention.

        Returns:
            Parsed tile metadata, or ``None`` when the path does not match.
        """
        fp = PurePath(filepath)
        suffix = fp.suffix[1:]
        if not fp.name.startswith("tile_") and suffix not in TileFile.FILE_TYPES:
            return None

        parts = fp.stem.split("_")
        tile_idx = int(parts[1])
        x = int(parts[2][1:])
        y = int(parts[3][1:])
        overlap = int(parts[4][7:])
        return TileFile(
            position=(x, y),
            tile_idx=tile_idx,
            tile_size=0,
            overlap=overlap,
            output_dir=str(fp.parent),
        )


@dataclass
class TileInfo:
    """Progress payload emitted after each processed tile."""

    prediction_time: float
    total_prediction_time: float
    total_tiles: int
    tiles_processed: int
    tiles_skipped: int
    number_of_masks: int


async def process_tiles(
    image,
    output_dir="tiles_output",
    initial_offset=[0, 0],
    max_tiles=None,
    tile_size=1024,
    overlap=256,
    use_cache=True,
    prompt: str | None = None,
) -> AsyncIterable[TileInfo | ProcessInfo]:
    """Process large image in tiles with SAM and persist per-tile outputs.

    Args:
        image: Input image as a numpy RGB array.
        output_dir: Directory where processed tiles are written.
        initial_offset: Starting offset ``[x, y]`` in tile units.
        max_tiles: Optional maximum number of tiles to process.
        tile_size: Size of each square tile in pixels.
        overlap: Overlap between adjacent tiles in pixels.
        use_cache: Whether to skip existing tile outputs in ``output_dir``.
        prompt: Optional text prompt passed to SAM.

    Yields:
        ``TileInfo`` entries during processing, followed by one ``ProcessInfo``.
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]

    cached_tiles = _get_cached_tiles(output_dir) if use_cache else set()
    if cached_tiles:
        print(f"\x1b[2K\rFound {len(cached_tiles)} cached tiles in '{output_dir}'")

    tile_positions = []
    for y in range(initial_offset[1] * tile_size, h, tile_size - overlap):
        for x in range(initial_offset[0] * tile_size, w, tile_size - overlap):
            if max_tiles is None or len(tile_positions) < max_tiles:
                tile_positions.append((x, y))

    tiles_to_process = [
        (i, x, y)
        for i, (x, y) in enumerate(tile_positions)
        if (x, y) not in cached_tiles
    ]
    tiles_skipped = len(tile_positions) - len(tiles_to_process)

    if tiles_skipped > 0:
        print(f"Skipping {tiles_skipped} cached tiles")

    total_prediction_time = 0.0
    tiles_processed = 0

    for tile_idx, x, y in tiles_to_process:
        x_end = min(x + tile_size, w)
        y_end = min(y + tile_size, h)
        tile = image[y:y_end, x:x_end]
        tile_image = Image.fromarray(tile)

        timestamp_start = time.perf_counter()
        result = sam.predict(tile_image, text=prompt or "tree crowns")

        prediction_time = time.perf_counter() - timestamp_start
        total_prediction_time += prediction_time

        tile_file = TileFile(
            position=(x, y),
            tile_idx=tile_idx,
            tile_size=tile_size,
            overlap=overlap,
            output_dir=output_dir,
        )
        result.save(tile_file.filename("npz"))

        overlay = sam.overlay_masks(tile_image, result.masks)
        overlay.save(tile_file.filename("png"))

        tiles_processed += 1

        yield TileInfo(
            total_tiles=len(tiles_to_process),
            prediction_time=prediction_time,
            total_prediction_time=total_prediction_time,
            tiles_processed=tiles_processed,
            tiles_skipped=tiles_skipped,
            number_of_masks=len(result.masks),
        )

    yield ProcessInfo(
        original_shape=image.shape,
        tile_size=tile_size,
        tile_overlap=overlap,
        output_dir=output_dir,
        total_prediction_time=total_prediction_time,
        tiles_processed=tiles_processed,
        tiles_skipped=tiles_skipped,
    )
    return


def _get_cached_tiles(output_dir: str) -> set[tuple[int, int]]:
    """Collect cached tile coordinates from existing output files.

    Args:
        output_dir: Directory that may already contain tile artifacts.

    Returns:
        Set of ``(x, y)`` tile origins that are already available as PNG files.
    """
    cached_tiles: set[tuple[int, int]] = set()
    for filename in os.listdir(output_dir):
        tile = TileFile.parse_filename(os.path.join(output_dir, filename))
        if tile is not None:
            cached_tiles.add(tile.position)
    return cached_tiles
