import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .image import Image


@dataclass
class Tile:
    image: Image
    path: str


def _tiles_rgb_path(output_path: str) -> str:
    """Get path to RGB tiles directory."""
    return os.path.join(output_path, "tiles_rgb")


def _tiles_annotated_path(output_path: str) -> str:
    """Get path to annotated tiles directory."""
    return os.path.join(output_path, "tiles_annotated")


@dataclass
class TileGeometry:
    start: tuple[int, int]
    end: tuple[int, int]


def _tile_geometries(
    image_shape: tuple[int, int],
    tile_size: tuple[int, int],
    overlap: tuple[int, int],
) -> list[TileGeometry]:
    x_tile_offset = tile_size[0] - overlap[0]
    y_tile_offset = tile_size[1] - overlap[1]

    tile_positions = []
    for y_start in range(0, image_shape[1], y_tile_offset):
        for x_start in range(0, image_shape[0], x_tile_offset):
            y_end = min(y_start + tile_size[1], image_shape[1])
            x_end = min(x_start + tile_size[0], image_shape[0])

            tile_positions.append(
                TileGeometry(start=(x_start, y_start), end=(x_end, y_end))
            )

    return tile_positions


def tile_image(
    image: Image,
    tile_size: int | tuple[int, int],
    overlap: int | tuple[int, int],
    output_path: str,
) -> "TilesDir":
    """Tile the input image and return a TilingDir object representing the tiles and metadata."""
    total_height, total_width = image.size[1], image.size[0]

    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    for tile_geo in _tile_geometries(
        image_shape=(total_width, total_height),
        tile_size=tile_size,
        overlap=overlap,
    ):
        tile = image.crop(
            tile_geo.start[0], tile_geo.start[1], tile_geo.end[0], tile_geo.end[1]
        )
        tile_filename = f"tile_x{tile_geo.start[0]}_y{tile_geo.start[1]}.png"

        tiles_rgb_dir = _tiles_rgb_path(output_path)
        if not os.path.exists(tiles_rgb_dir):
            os.mkdir(tiles_rgb_dir)

        tile.save(os.path.join(tiles_rgb_dir, tile_filename))

    return TilesDir(
        output_path=output_path,
        tile_size=tile_size,
        overlap=overlap,
        original_shape=(total_height, total_width, image.channels),
    )


class TilesDir:
    """
    Class representing a directory containing tiled images and metadata for reconstruction.

    Tile directory structure:
     - tiles_rgb/
     - metadata.json

    ```metadata.json
    {
        "original_shape": [height, width, channels],
        "tile_size": [1024, 1024],
        "overlap": [256, 256],
    }
    ```
    """

    output_path: str
    tile_size: tuple[int, int]
    overlap: tuple[int, int]
    original_shape: tuple[int, int, int]

    def __init__(
        self,
        output_path: str,
        tile_size: tuple[int, int],
        overlap: tuple[int, int],
        original_shape: tuple[int, int, int],
    ) -> None:
        self.output_path = output_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.original_shape = original_shape

    @staticmethod
    def load_from_dir(output_path: str) -> "TilesDir":
        metadata_path = os.path.join(output_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return TilesDir(
            output_path=output_path,
            tile_size=metadata["tile_size"],
            overlap=metadata["overlap"],
            original_shape=tuple(metadata["original_shape"]),
        )

    def save_to_dir(self) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        metadata_path = os.path.join(self.output_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "original_shape": list(self.original_shape),
                    "tile_size": list(self.tile_size),
                    "overlap": list(self.overlap),
                },
                f,
            )

    def save_annotated_tile(self, tile: Tile, annotated_image: Image) -> None:
        """
        Save the annotated tile image to the tiles_annotated directory.

        Args:
            tile: Tile object containing the original tile image and its path.
            annotated_image: Annotated Image object to be saved.
        """
        if not os.path.exists(self.tiles_annotated_path()):
            os.mkdir(self.tiles_annotated_path())
        tile_filename = os.path.basename(tile.path)
        annotated_image.save(os.path.join(self.tiles_annotated_path(), tile_filename))

    def reconstruct_image(self) -> Image:
        """
        Reconstruct the full image from processed tiles.

        Args:
            original_shape: Shape of the original image (height, width, channels).
            tiles_dir: Directory containing the processed tile images.
            tile_size: Size of each tile.
            overlap: Overlap between adjacent tiles.

        Returns:
            Reconstructed Image object.
        """
        h, w, c = self.original_shape
        reconstructed = np.zeros((h, w, 4), dtype=np.float32)  # RGBA for blending
        weight_map = np.zeros((h, w), dtype=np.float32)

        # Create a weight mask for blending (higher weight in center, lower at edges)
        blend_mask = np.ones(self.tile_size, dtype=np.float32)
        if self.overlap[0] > 0:
            # Create linear ramp for overlap regions
            x_ramp = np.linspace(0, 1, self.overlap[0])

            # Apply ramp to edges
            blend_mask[:, : self.overlap[0]] *= x_ramp[np.newaxis, :]  # left edge
            blend_mask[:, -self.overlap[0] :] *= x_ramp[::-1][
                np.newaxis, :
            ]  # right edge

        if self.overlap[1] > 0:
            # Create linear ramp for overlap regions

            y_ramp = np.linspace(0, 1, self.overlap[1])
            blend_mask[: self.overlap[1], :] *= y_ramp[:, np.newaxis]  # top edge
            blend_mask[-self.overlap[1] :, :] *= y_ramp[::-1, np.newaxis]  # bottom edge

        # Find all tile files and extract their positions
        tile_files = []
        for filename in os.listdir(self.tiles_rgb_path()):
            if filename.startswith("tile_") and filename.endswith(".png"):
                # Parse position from filename: tile_x0_y0.png
                parts = filename.replace(".png", "").split("_")
                x = int(parts[1][1:])  # Remove 'x' prefix
                y = int(parts[2][1:])  # Remove 'y' prefix
                tile_files.append((filename, x, y))

        # Sort by position for consistent processing
        tile_files.sort(key=lambda t: (t[2], t[1]))

        for filename, x, y in tile_files:
            tile_path = os.path.join(self.tiles_rgb_path(), filename)
            try:
                tile_image = Image.load(tile_path)
            except FileNotFoundError:
                print(f"Warning: Could not read {tile_path}")
                continue

            # Convert to RGBA for blending
            tile_data = tile_image.data
            if tile_image.channels == 3:
                # Add alpha channel
                tile = np.dstack(
                    [tile_data, np.full(tile_data.shape[:2], 255, dtype=np.uint8)]
                )
            else:
                tile = tile_data

            tile = tile.astype(np.float32) / 255.0

            # Calculate target tile dimensions (may be smaller at edges of image)
            x_end = min(x + self.tile_size[0], w)
            y_end = min(y + self.tile_size[1], h)
            target_w = x_end - x
            target_h = y_end - y

            # Get the appropriate portion of the blend mask for target dimensions
            current_blend = blend_mask[:target_h, :target_w]

            # Resize tile to match target dimensions (tiles from matplotlib may have different resolution due to DPI)
            tile_h, tile_w = tile.shape[:2]
            if tile_h != target_h or tile_w != target_w:
                tile = cv2.resize(
                    tile, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )

            # Apply weighted blending
            for c_idx in range(4):
                reconstructed[y:y_end, x:x_end, c_idx] += (
                    tile[:, :, c_idx] * current_blend
                )
            weight_map[y:y_end, x:x_end] += current_blend

        # Normalize by weight map to complete the blending
        weight_map = np.maximum(weight_map, 1e-6)  # Avoid division by zero
        for c_idx in range(4):
            reconstructed[:, :, c_idx] /= weight_map

        # Convert back to uint8
        reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)

        # Create Image object
        return Image(size=(w, h), channels=4, data=reconstructed_uint8)

    def tiles_rgb_path(self) -> str:
        """Get path to RGB tiles directory."""
        return _tiles_rgb_path(self.output_path)

    def tiles_annotated_path(self) -> str:
        """Get path to annotated tiles directory."""
        return _tiles_annotated_path(self.output_path)

    def get_tile_positions(self) -> set[tuple[int, int]]:
        """Get set of (x, y) positions for already processed tiles."""
        cached = set()
        if not os.path.exists(self.tiles_rgb_path()):
            return cached

        for filename in os.listdir(self.tiles_rgb_path()):
            if filename.startswith("tile_") and filename.endswith(".png"):
                # Parse position from filename: tile_x0_y0.png
                parts = filename.replace(".png", "").split("_")
                x = int(parts[1][1:])  # Remove 'x' prefix
                y = int(parts[2][1:])  # Remove 'y' prefix
                cached.add((x, y))

        return cached

    def __iter__(self) -> "_TilesDirIterator":
        return _TilesDirIterator(self)

    def __len__(self) -> int:
        return len(self.__iter__())


class _TilesDirIterator:
    _tile_paths: list[str]
    _current_index: int

    def __init__(self, tiling_dir: TilesDir) -> None:
        tiles_dir = tiling_dir.tiles_rgb_path()
        self._tile_paths = [
            os.path.join(tiles_dir, f)
            for f in os.listdir(tiles_dir)
            if f.startswith("tile_") and f.endswith(".png")
        ]
        self._current_index = 0

    def __iter__(self) -> "_TilesDirIterator":
        return self

    def __len__(self) -> int:
        return len(self._tile_paths)

    def __next__(self) -> Tile:
        if self._current_index >= len(self._tile_paths):
            raise StopIteration

        image_path = self._tile_paths[self._current_index]
        self._current_index += 1
        image = Image.load(image_path)
        return Tile(image=image, path=image_path)
