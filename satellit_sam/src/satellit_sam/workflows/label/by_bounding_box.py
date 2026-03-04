"""Create label overlays from bounding-box prompts."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from satellit_sam.core import (
    Image,
    tile_image,
)
from satellit_sam.prompts import (
    parse_tile_origin,
    project_bboxes_to_tile,
    tile_id_from_origin,
)


def make_labels_by_bounding_box(
    image_path: Path,
    tile_size: int,
    overlap: int,
    output_path: Path,
    bbox_prompts: list[tuple[float, float, float, float]],
    weak_label_bboxes_by_tile: dict[str, list[tuple[float, float, float, float]]]
    | None = None,
) -> None:
    """Generate tile-wise annotated labels from bbox prompts.

    Args:
        image_path: Path to the input image.
        tile_size: Tile edge length in pixels.
        overlap: Tile overlap in pixels.
        output_path: Directory where outputs are written.
        bbox_prompts: Global image-space bbox prompts (`x1,y1,x2,y2`).
        weak_label_bboxes_by_tile: Optional tile-local bboxes keyed by tile id.
    """
    from satellit_sam.sam3 import sam

    print(
        "DEPRECATED workflow: `label by-bounding-boxes` is deprecated. "
        "Use `predict image-masks --bbox ...` or "
        "`predict image-masks --weak-labels-csv ...` for canonical strong-label outputs."
    )

    output_path.mkdir(parents=True, exist_ok=True)

    sam.print_debug_info()
    print(f"Loading image from: {image_path}")
    image = Image.load(str(image_path))

    print(f"Tiling image (tile_size={tile_size}, overlap={overlap})...")
    tiling_dir = tile_image(
        image, tile_size=tile_size, overlap=overlap, output_path=str(output_path)
    )
    tiling_dir.save_to_dir()

    weak_bbox_count = (
        sum(len(boxes) for boxes in weak_label_bboxes_by_tile.values())
        if weak_label_bboxes_by_tile
        else 0
    )
    total_bbox_prompts = len(bbox_prompts) + weak_bbox_count
    print(f"Generating masks with {total_bbox_prompts} bbox prompt(s)...")

    with tqdm(total=len(tiling_dir), desc="Processing tiles", unit="tile") as pbar:
        for tile in tiling_dir:
            tile_origin: tuple[int, int] | None = None
            tile_bboxes: list[tuple[float, float, float, float]] = []

            if bbox_prompts:
                tile_origin = parse_tile_origin(tile.path)
                tile_bboxes.extend(
                    project_bboxes_to_tile(
                        image_bboxes=bbox_prompts,
                        tile_origin=tile_origin,
                        tile_size=tile.image.size,
                    )
                )

            if weak_label_bboxes_by_tile:
                if tile_origin is None:
                    tile_origin = parse_tile_origin(tile.path)
                tile_id = tile_id_from_origin(tile_origin)
                tile_bboxes.extend(weak_label_bboxes_by_tile.get(tile_id, []))

            if not tile_bboxes:
                tiling_dir.save_annotated_tile(tile, tile.image.copy())
                pbar.update(1)
                continue

            ann_image = sam.predict(tile.image, boxes=tile_bboxes)
            tiling_dir.save_annotated_tile(tile, ann_image)
            pbar.update(1)

    print(f"✓ Processing complete! Results saved to: {output_path}")
