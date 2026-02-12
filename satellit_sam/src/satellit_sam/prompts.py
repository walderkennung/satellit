import os
import re


BBOX_DELIMITER = ","
TILE_FILENAME_PATTERN = re.compile(r"^tile_x(-?\d+)_y(-?\d+)\.png$")


def parse_bbox_prompt(raw_bbox: str) -> tuple[float, float, float, float]:
    """Parse one bbox prompt in x1,y1,x2,y2 format."""
    parts = [part.strip() for part in raw_bbox.split(BBOX_DELIMITER)]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid bbox '{raw_bbox}'. Expected format: x1,y1,x2,y2."
        )

    try:
        x1, y1, x2, y2 = (float(value) for value in parts)
    except ValueError as err:
        raise ValueError(
            f"Invalid bbox '{raw_bbox}'. Coordinates must be numeric."
        ) from err

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bbox '{raw_bbox}'. Expected x2>x1 and y2>y1."
        )

    return x1, y1, x2, y2


def parse_bbox_prompts(raw_bboxes: list[str] | None) -> list[tuple[float, float, float, float]]:
    """Parse repeatable bbox prompt options."""
    if not raw_bboxes:
        return []

    return [parse_bbox_prompt(raw_bbox) for raw_bbox in raw_bboxes]


def parse_tile_origin(tile_path: str) -> tuple[int, int]:
    """Extract tile origin (x, y) from tile filename."""
    filename = os.path.basename(tile_path)
    match = TILE_FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(
            f"Could not parse tile origin from '{tile_path}'. "
            "Expected filename like tile_x0_y0.png."
        )

    return int(match.group(1)), int(match.group(2))


def project_bboxes_to_tile(
    image_bboxes: list[tuple[float, float, float, float]],
    tile_origin: tuple[int, int],
    tile_size: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    """Project image-space bboxes into one tile as clipped tile-space boxes."""
    tile_x, tile_y = tile_origin
    tile_width, tile_height = tile_size
    tile_x_end = tile_x + tile_width
    tile_y_end = tile_y + tile_height

    tile_bboxes: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in image_bboxes:
        clipped_x1 = max(x1, tile_x)
        clipped_y1 = max(y1, tile_y)
        clipped_x2 = min(x2, tile_x_end)
        clipped_y2 = min(y2, tile_y_end)

        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            continue

        tile_bboxes.append(
            (
                clipped_x1 - tile_x,
                clipped_y1 - tile_y,
                clipped_x2 - tile_x,
                clipped_y2 - tile_y,
            )
        )

    return tile_bboxes
