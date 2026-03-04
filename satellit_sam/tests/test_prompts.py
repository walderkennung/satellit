"""Tests for prompt parsing and tile-space bbox projection."""

import pytest

from satellit_sam.prompts import (
    load_weak_label_bboxes,
    parse_bbox_prompt,
    parse_bbox_prompts,
    parse_point_prompt,
    parse_point_prompts,
    parse_tile_origin,
    project_bboxes_to_tile,
    project_points_to_tile,
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
class TestPointParsing:
    """Tests for point prompt parsing."""

    def test_parse_point_prompt(self):
        assert parse_point_prompt("10,20") == (10.0, 20.0)

    def test_parse_point_prompt_rejects_invalid_shape(self):
        with pytest.raises(ValueError, match="Expected format"):
            parse_point_prompt("10")

    def test_parse_point_prompt_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="numeric"):
            parse_point_prompt("10,foo")

    def test_parse_point_prompts(self):
        assert parse_point_prompts(["0,0", "20.5,30.5"]) == [
            (0.0, 0.0),
            (20.5, 30.5),
        ]

    def test_parse_point_prompts_none(self):
        assert parse_point_prompts(None) == []


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

    def test_project_points_to_tile(self):
        tile_points = project_points_to_tile(
            image_points=[(10.0, 10.0), (120.0, 120.0), (128.0, 120.0)],
            tile_origin=(64, 64),
            tile_size=(64, 64),
        )

        assert tile_points == [(56.0, 56.0)]


@pytest.mark.unit
class TestWeakLabelBboxes:
    """Tests for loading weak-label bbox prompts from CSV."""

    def test_load_weak_label_bboxes(self, tmp_path):
        csv_path = tmp_path / "labels_tiles.csv"
        csv_path.write_text(
            (
                "tile_id,tree_id,x_pixel,y_pixel,crown_radius,bbox_x1,bbox_y1,bbox_x2,bbox_y2\n"
                "tile_x0_y0,t1,10,20,5,4.0,14.0,16.0,26.0\n"
                "tile_x0_y0,t2,30,40,6,22.0,32.0,38.0,48.0\n"
                "tile_x64_y0,t3,8,9,3,2.5,3.5,12.5,13.5\n"
            ),
            encoding="utf-8",
        )

        assert load_weak_label_bboxes(csv_path) == {
            "tile_x0_y0": [(4.0, 14.0, 16.0, 26.0), (22.0, 32.0, 38.0, 48.0)],
            "tile_x64_y0": [(2.5, 3.5, 12.5, 13.5)],
        }

    def test_load_weak_label_bboxes_rejects_missing_columns(self, tmp_path):
        csv_path = tmp_path / "labels_tiles.csv"
        csv_path.write_text(
            "tile_id,tree_id,x_pixel,y_pixel,crown_radius\n"
            "tile_x0_y0,t1,10,20,5\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing required bbox columns"):
            load_weak_label_bboxes(csv_path)

    def test_load_weak_label_bboxes_rejects_invalid_order(self, tmp_path):
        csv_path = tmp_path / "labels_tiles.csv"
        csv_path.write_text(
            (
                "tile_id,tree_id,x_pixel,y_pixel,crown_radius,bbox_x1,bbox_y1,bbox_x2,bbox_y2\n"
                "tile_x0_y0,t1,10,20,5,4.0,14.0,4.0,26.0\n"
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="invalid bbox order"):
            load_weak_label_bboxes(csv_path)
