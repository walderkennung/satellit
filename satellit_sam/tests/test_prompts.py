"""Tests for prompt parsing and tile-space bbox projection."""

import pytest

from satellit_sam.prompts import (
    parse_bbox_prompt,
    parse_bbox_prompts,
    parse_tile_origin,
    project_bboxes_to_tile,
)


@pytest.mark.unit
class TestBBoxParsing:
    """Tests for bbox prompt parsing."""

    def test_parse_bbox_prompt(self):
        assert parse_bbox_prompt("10,20,30,40") == (10.0, 20.0, 30.0, 40.0)

    def test_parse_bbox_prompt_rejects_invalid_shape(self):
        with pytest.raises(ValueError, match="Expected format"):
            parse_bbox_prompt("10,20,30")

    def test_parse_bbox_prompt_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="numeric"):
            parse_bbox_prompt("10,foo,30,40")

    def test_parse_bbox_prompt_rejects_invalid_order(self):
        with pytest.raises(ValueError, match="x2>x1 and y2>y1"):
            parse_bbox_prompt("10,20,5,40")

    def test_parse_bbox_prompts(self):
        assert parse_bbox_prompts(["0,0,10,10", "20,30,40,50"]) == [
            (0.0, 0.0, 10.0, 10.0),
            (20.0, 30.0, 40.0, 50.0),
        ]

    def test_parse_bbox_prompts_none(self):
        assert parse_bbox_prompts(None) == []


@pytest.mark.unit
class TestTileProjection:
    """Tests for tile filename parsing and bbox projection."""

    def test_parse_tile_origin(self):
        assert parse_tile_origin("/tmp/tile_x128_y256.png") == (128, 256)

    def test_parse_tile_origin_rejects_invalid_filename(self):
        with pytest.raises(ValueError, match="Could not parse tile origin"):
            parse_tile_origin("/tmp/tile_foo.png")

    def test_project_bboxes_to_tile(self):
        image_bboxes = [(10.0, 10.0, 90.0, 90.0), (120.0, 120.0, 180.0, 180.0)]
        tile_bboxes = project_bboxes_to_tile(
            image_bboxes=image_bboxes,
            tile_origin=(64, 64),
            tile_size=(64, 64),
        )

        assert tile_bboxes == [(0.0, 0.0, 26.0, 26.0), (56.0, 56.0, 64.0, 64.0)]

    def test_project_bboxes_to_tile_drops_non_overlapping_boxes(self):
        tile_bboxes = project_bboxes_to_tile(
            image_bboxes=[(200.0, 200.0, 250.0, 250.0)],
            tile_origin=(0, 0),
            tile_size=(64, 64),
        )

        assert tile_bboxes == []
