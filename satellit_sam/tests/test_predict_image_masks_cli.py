"""Tests for predict image-masks CLI model selection and validation."""

import pytest
from typer.testing import CliRunner

from satellit_sam.cli import predict as predict_cli
from satellit_sam.cli.root import app


@pytest.mark.unit
def test_predict_image_masks_cli_accepts_dinov3_with_text(
    small_test_image,
    temp_dir,
    monkeypatch,
):
    """CLI should pass `dinov3` model through to workflow when text is set."""
    image_path = temp_dir / "source.png"
    output_path = temp_dir / "predict_out"
    small_test_image.save(str(image_path))

    captured: dict[str, object] = {}

    def _fake_predict_image_masks(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(predict_cli, "predict_image_masks", _fake_predict_image_masks)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "predict",
            "image-masks",
            "--image",
            str(image_path),
            "--output-path",
            str(output_path),
            "--model",
            "dinov3",
            "--text",
            "tree",
        ],
    )

    assert result.exit_code == 0
    assert captured["model"] == "dinov3"
    assert captured["text_prompt"] == "tree"
    assert captured["bbox_prompts"] == []
    assert captured["point_prompts"] == []
    assert isinstance(captured["command"], str)
    assert captured["command"] != ""


@pytest.mark.unit
def test_predict_image_masks_cli_rejects_dinov3_with_bbox(
    small_test_image,
    temp_dir,
):
    """CLI should reject box prompts with `dinov3`."""
    image_path = temp_dir / "source.png"
    small_test_image.save(str(image_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "predict",
            "image-masks",
            "--image",
            str(image_path),
            "--model",
            "dinov3",
            "--text",
            "tree",
            "--bbox",
            "1,1,5,5",
        ],
    )

    assert result.exit_code != 0
    assert "Model 'dinov3' supports --text prompts only." in result.output


@pytest.mark.unit
def test_predict_image_masks_cli_passes_tiling_options(
    small_test_image,
    temp_dir,
    monkeypatch,
):
    """CLI should pass tile-streaming options to the workflow."""
    image_path = temp_dir / "source.png"
    small_test_image.save(str(image_path))
    captured: dict[str, object] = {}

    def _fake_predict_image_masks(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(predict_cli, "predict_image_masks", _fake_predict_image_masks)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "predict",
            "image-masks",
            "--image",
            str(image_path),
            "--text",
            "tree crowns",
            "--tile-size",
            "512",
            "--tile-overlap",
            "32",
            "--merge-iou-threshold",
            "0.3",
        ],
    )

    assert result.exit_code == 0
    assert captured["tile_size"] == 512
    assert captured["tile_overlap"] == 32
    assert captured["merge_iou_threshold"] == 0.3
    assert isinstance(captured["command"], str)
    assert captured["command"] != ""


@pytest.mark.unit
def test_predict_image_masks_cli_rejects_tile_overlap_gte_tile_size(
    small_test_image,
    temp_dir,
):
    """CLI should reject overlap that is not smaller than tile size."""
    image_path = temp_dir / "source.png"
    small_test_image.save(str(image_path))
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "predict",
            "image-masks",
            "--image",
            str(image_path),
            "--text",
            "tree",
            "--tile-size",
            "64",
            "--tile-overlap",
            "64",
        ],
    )

    assert result.exit_code != 0
    assert "tile-overlap" in result.output
    assert "tile-size" in result.output


@pytest.mark.unit
def test_predict_image_masks_cli_rejects_invalid_merge_iou_threshold(
    small_test_image,
    temp_dir,
):
    """CLI should enforce merge IoU threshold range."""
    image_path = temp_dir / "source.png"
    small_test_image.save(str(image_path))
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "predict",
            "image-masks",
            "--image",
            str(image_path),
            "--text",
            "tree",
            "--merge-iou-threshold",
            "1.2",
        ],
    )

    assert result.exit_code != 0
