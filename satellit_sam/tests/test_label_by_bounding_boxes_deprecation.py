"""CLI tests for deprecated label by-bounding-boxes command."""

import pytest
from typer.testing import CliRunner

from satellit_sam.cli import label as label_cli
from satellit_sam.cli.root import app


@pytest.mark.unit
def test_label_by_bounding_boxes_cli_prints_deprecation_warning(
    small_test_image,
    temp_dir,
    monkeypatch,
):
    """Command should emit migration guidance toward predict image-masks."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "label_out"
    small_test_image.save(str(image_path))

    captured: dict[str, object] = {}

    def _fake_make_labels_by_bounding_box(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        label_cli,
        "make_labels_by_bounding_box",
        _fake_make_labels_by_bounding_box,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "label",
            "by-bounding-boxes",
            "--image",
            str(image_path),
            "--output-path",
            str(output_path),
            "--bbox",
            "1,1,5,5",
        ],
    )

    assert result.exit_code == 0
    assert "DEPRECATED" in result.output
    assert "predict image-masks" in result.output
    assert captured["image_path"] == image_path
    assert captured["output_path"] == output_path
