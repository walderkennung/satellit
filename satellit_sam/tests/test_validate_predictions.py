"""Tests for SAM3 strong-label validation against inventory stems."""

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, ogr, osr

from satellit_sam.workflows.label.validate_predictions import validate_sam3_predictions


def _write_test_geotiff(image_path: Path, width: int, height: int) -> None:
    """Create a deterministic WGS84 GeoTIFF fixture."""
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("GDAL GTiff driver is not available.")

    dataset = driver.Create(str(image_path), width, height, 1, gdal.GDT_Byte)
    if dataset is None:
        raise RuntimeError("Failed to create GeoTIFF test fixture.")

    try:
        dataset.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())

        band = dataset.GetRasterBand(1)
        if band is None:
            raise RuntimeError("Missing GeoTIFF band 1.")
        band.WriteArray(np.zeros((height, width), dtype=np.uint8))
    finally:
        dataset = None


def _write_inventory_shp(shp_path: Path, trees: list[dict[str, object]]) -> None:
    """Write a minimal WGS84 inventory shapefile for tests."""
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("GDAL ESRI Shapefile driver is not available.")

    if shp_path.exists():
        driver.DeleteDataSource(str(shp_path))

    datasource = driver.CreateDataSource(str(shp_path))
    if datasource is None:
        raise RuntimeError(f"Could not create shapefile: {shp_path}")

    try:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = datasource.CreateLayer("inventory", srs=srs, geom_type=ogr.wkbPoint)
        if layer is None:
            raise RuntimeError("Could not create inventory layer.")

        for field_name, field_type, width in [
            ("TreeID", ogr.OFTString, 64),
            ("StemTag", ogr.OFTString, 64),
            ("Status", ogr.OFTString, 32),
            ("DBH", ogr.OFTReal, 0),
        ]:
            definition = ogr.FieldDefn(field_name, field_type)
            if width > 0:
                definition.SetWidth(width)
            if layer.CreateField(definition) != 0:
                raise RuntimeError(f"Could not create field: {field_name}")

        layer_defn = layer.GetLayerDefn()
        for tree in trees:
            feature = ogr.Feature(layer_defn)
            feature.SetField("TreeID", str(tree["tree_id"]))
            feature.SetField("StemTag", str(tree.get("stem_id", "")))
            feature.SetField("Status", str(tree.get("status", "alive")))
            feature.SetField("DBH", float(tree.get("dbh", 30.0)))

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(float(tree["x"]), float(tree["y"]))
            feature.SetGeometry(point)

            if layer.CreateFeature(feature) != 0:
                raise RuntimeError("Failed to write inventory feature.")
            feature = None
    finally:
        datasource = None


@pytest.mark.unit
def test_validate_predictions_matches_stem_inside_mask(temp_dir):
    """Stem pixels inside a SAM3 mask should get a label assignment."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    npz_path = temp_dir / "image_masks.npz"
    output_csv = temp_dir / "validation.csv"

    _write_test_geotiff(image_path=image_path, width=6, height=6)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 2.0, "y": 2.0, "dbh": 30.0}],
    )

    masks = np.zeros((1, 6, 6), dtype=bool)
    masks[0, 2, 2] = True
    np.savez_compressed(npz_path, masks=masks, scores=np.asarray([0.9], dtype=np.float32))

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_npz=npz_path,
        inventory_shp=shp_path,
        output_csv=output_csv,
        dbh_unit="cm",
    )

    assert result.loc[0, "label_id"] == 0
    assert result.loc[0, "tree_id"] == "t1"
    assert result.loc[0, "stem_id"] == "s1"
    assert bool(result.loc[0, "prediction_coverage"]) is True


@pytest.mark.unit
def test_validate_predictions_marks_tree_without_matching_mask(temp_dir):
    """Trees outside all masks should remain unmatched."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    npz_path = temp_dir / "image_masks.npz"

    _write_test_geotiff(image_path=image_path, width=6, height=6)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 1.0, "y": 1.0, "dbh": 30.0}],
    )

    masks = np.zeros((1, 6, 6), dtype=bool)
    masks[0, 4, 4] = True
    np.savez_compressed(npz_path, masks=masks, scores=np.asarray([0.9], dtype=np.float32))

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_npz=npz_path,
        inventory_shp=shp_path,
        dbh_unit="cm",
    )

    assert result.loc[0, "label_id"] is None
    assert bool(result.loc[0, "prediction_coverage"]) is True


@pytest.mark.unit
def test_validate_predictions_does_not_reuse_label_for_multiple_trees(temp_dir):
    """A label already matched once must not match another tree."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    npz_path = temp_dir / "image_masks.npz"

    _write_test_geotiff(image_path=image_path, width=6, height=6)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[
            {"tree_id": "a", "stem_id": "s1", "x": 2.0, "y": 2.0, "dbh": 30.0},
            {"tree_id": "b", "stem_id": "s2", "x": 2.0, "y": 2.0, "dbh": 30.0},
        ],
    )

    masks = np.zeros((1, 6, 6), dtype=bool)
    masks[0, 2, 2] = True
    np.savez_compressed(npz_path, masks=masks, scores=np.asarray([0.9], dtype=np.float32))

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_npz=npz_path,
        inventory_shp=shp_path,
        dbh_unit="cm",
        deduplicate_tree_id=False,
    )

    assert list(result["label_id"]) == [0, None]
    assert list(result["prediction_coverage"]) == [True, True]


@pytest.mark.unit
def test_validate_predictions_uses_score_and_area_tiebreak(temp_dir):
    """When multiple labels fit one stem, score and area ranking is applied."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    npz_path = temp_dir / "image_masks.npz"

    _write_test_geotiff(image_path=image_path, width=6, height=6)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 2.0, "y": 2.0, "dbh": 30.0}],
    )

    masks = np.zeros((2, 6, 6), dtype=bool)
    masks[0, 1:4, 1:4] = True  # area 9
    masks[1, 2, 2] = True  # area 1
    scores = np.asarray([0.8, 0.8], dtype=np.float32)
    np.savez_compressed(npz_path, masks=masks, scores=scores)

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_npz=npz_path,
        inventory_shp=shp_path,
        dbh_unit="cm",
    )

    assert result.loc[0, "label_id"] == 1
    assert bool(result.loc[0, "prediction_coverage"]) is True


@pytest.mark.unit
def test_validate_predictions_applies_dbh_filter(temp_dir):
    """DBH filtering should limit which inventory trees are validated."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    npz_path = temp_dir / "image_masks.npz"

    _write_test_geotiff(image_path=image_path, width=6, height=6)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[
            {"tree_id": "small", "stem_id": "s1", "x": 1.0, "y": 1.0, "dbh": 5.0},
            {"tree_id": "large", "stem_id": "s2", "x": 2.0, "y": 2.0, "dbh": 25.0},
        ],
    )

    masks = np.zeros((1, 6, 6), dtype=bool)
    masks[0, 2, 2] = True
    np.savez_compressed(npz_path, masks=masks, scores=np.asarray([0.9], dtype=np.float32))

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_npz=npz_path,
        inventory_shp=shp_path,
        dbh_unit="cm",
        min_dbh_cm=10.0,
    )

    assert len(result) == 1
    assert result.loc[0, "tree_id"] == "large"
    assert bool(result.loc[0, "prediction_coverage"]) is True


@pytest.mark.unit
def test_validate_predictions_stem_id_falls_back_to_tree_id(temp_dir):
    """Missing stem ids should fall back to the tree id in result rows."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    npz_path = temp_dir / "image_masks.npz"

    _write_test_geotiff(image_path=image_path, width=6, height=6)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "", "x": 1.0, "y": 1.0, "dbh": 30.0}],
    )

    masks = np.zeros((1, 6, 6), dtype=bool)
    np.savez_compressed(npz_path, masks=masks, scores=np.asarray([0.9], dtype=np.float32))

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_npz=npz_path,
        inventory_shp=shp_path,
        stem_id_field="MissingStemField",
        dbh_unit="cm",
    )

    assert result.loc[0, "stem_id"] == "t1"
    assert bool(result.loc[0, "prediction_coverage"]) is True


def _write_tile_npz(
    tile_path: Path,
    tile_origin: tuple[int, int],
    tile_size: tuple[int, int],
    masks: np.ndarray,
    boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> None:
    """Write one per-tile prediction NPZ matching predict workflow schema."""
    mask_count = int(masks.shape[0])
    if boxes is None:
        boxes = np.zeros((mask_count, 4), dtype=np.float32)
    if scores is None:
        scores = np.ones((mask_count,), dtype=np.float32)

    np.savez_compressed(
        tile_path,
        tile_id=np.asarray(tile_path.stem),
        tile_origin=np.asarray([tile_origin[0], tile_origin[1]], dtype=np.int32),
        tile_size=np.asarray([tile_size[0], tile_size[1]], dtype=np.int32),
        masks=np.asarray(masks, dtype=bool),
        boxes=np.asarray(boxes, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
    )


@pytest.mark.unit
def test_validate_predictions_from_tile_dir_basic_match(temp_dir):
    """Validation should match trees using per-tile NPZ masks."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    tiles_dir = temp_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    _write_test_geotiff(image_path=image_path, width=8, height=8)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 2.0, "y": 2.0, "dbh": 30.0}],
    )

    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 2, 2] = True
    _write_tile_npz(
        tile_path=tiles_dir / "tile_x0_y0.npz",
        tile_origin=(0, 0),
        tile_size=(4, 4),
        masks=masks,
        scores=np.asarray([0.9], dtype=np.float32),
    )

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_tiles_dir=tiles_dir,
        inventory_shp=shp_path,
        dbh_unit="cm",
    )

    assert result.loc[0, "label_id"] == 0
    assert bool(result.loc[0, "prediction_coverage"]) is True


@pytest.mark.unit
def test_validate_predictions_from_tile_dir_marks_uncovered_tree(temp_dir):
    """Trees in unprocessed areas should be marked as uncovered."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    tiles_dir = temp_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    _write_test_geotiff(image_path=image_path, width=8, height=8)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 6.0, "y": 6.0, "dbh": 30.0}],
    )

    masks = np.zeros((1, 4, 4), dtype=bool)
    _write_tile_npz(
        tile_path=tiles_dir / "tile_x0_y0.npz",
        tile_origin=(0, 0),
        tile_size=(4, 4),
        masks=masks,
    )

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_tiles_dir=tiles_dir,
        inventory_shp=shp_path,
        dbh_unit="cm",
    )

    assert result.loc[0, "label_id"] is None
    assert bool(result.loc[0, "prediction_coverage"]) is False


@pytest.mark.unit
def test_validate_predictions_from_tile_dir_handles_overlapping_tiles(temp_dir):
    """One-to-one matching should still apply with overlapping per-tile labels."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    tiles_dir = temp_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    _write_test_geotiff(image_path=image_path, width=8, height=8)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[
            {"tree_id": "a", "stem_id": "s1", "x": 3.0, "y": 3.0, "dbh": 30.0},
            {"tree_id": "b", "stem_id": "s2", "x": 3.0, "y": 3.0, "dbh": 30.0},
        ],
    )

    mask_a = np.zeros((1, 6, 6), dtype=bool)
    mask_a[0, 3, 3] = True
    _write_tile_npz(
        tile_path=tiles_dir / "tile_x0_y0.npz",
        tile_origin=(0, 0),
        tile_size=(6, 6),
        masks=mask_a,
        scores=np.asarray([0.6], dtype=np.float32),
    )
    mask_b = np.zeros((1, 6, 6), dtype=bool)
    mask_b[0, 1, 1] = True  # global (3,3) in tile starting at (2,2)
    _write_tile_npz(
        tile_path=tiles_dir / "tile_x2_y2.npz",
        tile_origin=(2, 2),
        tile_size=(6, 6),
        masks=mask_b,
        scores=np.asarray([0.8], dtype=np.float32),
    )

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_tiles_dir=tiles_dir,
        inventory_shp=shp_path,
        dbh_unit="cm",
    )

    assert list(result["label_id"]) == [1, 0]
    assert list(result["prediction_coverage"]) == [True, True]


@pytest.mark.unit
def test_validate_predictions_from_tile_dir_without_index_csv(temp_dir):
    """Validation should not require index.csv when tile NPZ files exist."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    tiles_dir = temp_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    _write_test_geotiff(image_path=image_path, width=8, height=8)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 1.0, "y": 1.0, "dbh": 30.0}],
    )
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1, 1] = True
    _write_tile_npz(
        tile_path=tiles_dir / "tile_x0_y0.npz",
        tile_origin=(0, 0),
        tile_size=(4, 4),
        masks=masks,
    )

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_tiles_dir=tiles_dir,
        inventory_shp=shp_path,
        dbh_unit="cm",
    )

    assert result.loc[0, "label_id"] == 0


@pytest.mark.unit
def test_validate_predictions_prefers_merged_npz_when_both_sources_given(temp_dir):
    """Merged NPZ should be used when both prediction sources are provided."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    merged_npz = temp_dir / "image_masks.npz"
    tiles_dir = temp_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    _write_test_geotiff(image_path=image_path, width=8, height=8)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 2.0, "y": 2.0, "dbh": 30.0}],
    )

    merged_masks = np.zeros((1, 8, 8), dtype=bool)
    merged_masks[0, 2, 2] = True
    np.savez_compressed(merged_npz, masks=merged_masks, scores=np.asarray([0.9], dtype=np.float32))

    tile_masks = np.zeros((1, 4, 4), dtype=bool)
    _write_tile_npz(
        tile_path=tiles_dir / "tile_x0_y0.npz",
        tile_origin=(0, 0),
        tile_size=(4, 4),
        masks=tile_masks,
        scores=np.asarray([0.1], dtype=np.float32),
    )

    result = validate_sam3_predictions(
        image_tif=image_path,
        predictions_npz=merged_npz,
        predictions_tiles_dir=tiles_dir,
        inventory_shp=shp_path,
        dbh_unit="cm",
    )

    assert result.loc[0, "label_id"] == 0
    assert bool(result.loc[0, "prediction_coverage"]) is True


@pytest.mark.unit
def test_validate_predictions_rejects_missing_predictions_sources(temp_dir):
    """Validation should fail when no predictions source is provided."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"

    _write_test_geotiff(image_path=image_path, width=6, height=6)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 2.0, "y": 2.0, "dbh": 30.0}],
    )

    with pytest.raises(ValueError, match="at least one predictions source"):
        validate_sam3_predictions(
            image_tif=image_path,
            inventory_shp=shp_path,
            dbh_unit="cm",
        )


@pytest.mark.unit
def test_validate_predictions_prints_confusion_matrix_and_unmatched_labels(
    temp_dir,
    capsys,
):
    """Summary should include unmatched mask labels and TP/FP/FN metrics."""
    image_path = temp_dir / "source.tif"
    shp_path = temp_dir / "inventory.shp"
    npz_path = temp_dir / "image_masks.npz"

    _write_test_geotiff(image_path=image_path, width=6, height=6)
    _write_inventory_shp(
        shp_path=shp_path,
        trees=[{"tree_id": "t1", "stem_id": "s1", "x": 2.0, "y": 2.0, "dbh": 30.0}],
    )

    masks = np.zeros((2, 6, 6), dtype=bool)
    masks[0, 2, 2] = True
    masks[1, 4, 4] = True
    np.savez_compressed(npz_path, masks=masks, scores=np.asarray([0.9, 0.7], dtype=np.float32))

    validate_sam3_predictions(
        image_tif=image_path,
        predictions_npz=npz_path,
        inventory_shp=shp_path,
        dbh_unit="cm",
    )

    output = capsys.readouterr().out
    assert "- unmatched mask labels: 1" in output
    assert "- true positives: 1" in output
    assert "- false positives: 1" in output
    assert "- false negatives: 0" in output
