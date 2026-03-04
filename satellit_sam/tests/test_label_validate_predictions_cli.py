"""CLI tests for label validate-predictions command."""

import numpy as np
import pytest
from typer.testing import CliRunner

from satellit_sam.cli import label as label_cli
from satellit_sam.cli.root import app


@pytest.mark.unit
def test_label_validate_predictions_cli_passes_arguments(temp_dir, monkeypatch):
    """CLI should pass parsed arguments through to the validation workflow."""
    image_path = temp_dir / "source.tif"
    predictions_path = temp_dir / "image_masks.npz"
    inventory_path = temp_dir / "inventory.shp"
    output_csv = temp_dir / "out.csv"

    image_path.write_bytes(b"not-read-by-mock")
    inventory_path.write_bytes(b"not-read-by-mock")
    np.savez_compressed(predictions_path, masks=np.zeros((0, 2, 2), dtype=bool))

    captured: dict[str, object] = {}

    def _fake_validate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(label_cli, "validate_sam3_predictions", _fake_validate)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "label",
            "validate-predictions",
            "--image-tif",
            str(image_path),
            "--predictions-npz",
            str(predictions_path),
            "--inventory-shp",
            str(inventory_path),
            "--output-csv",
            str(output_csv),
            "--stem-id-field",
            "StemTag",
            "--min-dbh-cm",
            "10",
            "--max-dbh-cm",
            "30",
        ],
    )

    assert result.exit_code == 0
    assert captured["image_tif"] == image_path
    assert captured["predictions_npz"] == predictions_path
    assert captured["inventory_shp"] == inventory_path
    assert captured["output_csv"] == output_csv
    assert captured["stem_id_field"] == "StemTag"
    assert captured["min_dbh_cm"] == 10.0
    assert captured["max_dbh_cm"] == 30.0


@pytest.mark.unit
def test_label_validate_predictions_cli_rejects_invalid_inventory_source_combo(temp_dir):
    """CLI should reject both or neither inventory source options."""
    image_path = temp_dir / "source.tif"
    predictions_path = temp_dir / "image_masks.npz"
    csv_path = temp_dir / "inventory.csv"
    shp_path = temp_dir / "inventory.shp"

    image_path.write_bytes(b"x")
    csv_path.write_text("TreeID;PX;PY;DBH\n", encoding="utf-8")
    shp_path.write_bytes(b"x")
    np.savez_compressed(predictions_path, masks=np.zeros((0, 2, 2), dtype=bool))

    runner = CliRunner()

    both_result = runner.invoke(
        app,
        [
            "label",
            "validate-predictions",
            "--image-tif",
            str(image_path),
            "--predictions-npz",
            str(predictions_path),
            "--inventory-csv",
            str(csv_path),
            "--inventory-shp",
            str(shp_path),
        ],
    )
    none_result = runner.invoke(
        app,
        [
            "label",
            "validate-predictions",
            "--image-tif",
            str(image_path),
            "--predictions-npz",
            str(predictions_path),
        ],
    )

    assert both_result.exit_code != 0
    assert "Provide only one" in both_result.output
    assert none_result.exit_code != 0
    assert "Provide either" in none_result.output


@pytest.mark.unit
def test_label_validate_predictions_cli_rejects_invalid_npz_payload(
    temp_dir,
    monkeypatch,
):
    """CLI should surface NPZ validation errors from the workflow."""
    image_path = temp_dir / "source.tif"
    predictions_path = temp_dir / "image_masks.npz"
    inventory_path = temp_dir / "inventory.shp"

    image_path.write_bytes(b"x")
    inventory_path.write_bytes(b"x")
    np.savez_compressed(predictions_path, boxes=np.zeros((1, 4), dtype=np.float32))

    def _raise_invalid_npz(**_kwargs):
        raise ValueError("Predictions NPZ is missing required 'masks' array.")

    monkeypatch.setattr(
        label_cli,
        "validate_sam3_predictions",
        _raise_invalid_npz,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "label",
            "validate-predictions",
            "--image-tif",
            str(image_path),
            "--predictions-npz",
            str(predictions_path),
            "--inventory-shp",
            str(inventory_path),
        ],
    )

    assert result.exit_code != 0
    assert "missing required 'masks'" in result.output
