"""Tests for SAM runtime device selection behavior."""

from satellit_sam import sam3
from satellit_sam.pytorch import PytorchInstance


class _DummyModel:
    """Simple model stub that records the selected device."""

    def __init__(self):
        self.device = None

    def to(self, device: str):
        self.device = device
        return self


class _DummyModelLoader:
    """Model loader stub compatible with ``from_pretrained`` usage."""

    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return _DummyModel()


class _DummyProcessorLoader:
    """Processor loader stub compatible with ``from_pretrained`` usage."""

    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return object()


def _patch_transformer_loaders(monkeypatch) -> None:
    """Replace heavyweight model/processor loaders with lightweight stubs."""
    monkeypatch.setattr(sam3, "Sam3Model", _DummyModelLoader)
    monkeypatch.setattr(sam3, "Sam2Model", _DummyModelLoader)
    monkeypatch.setattr(sam3, "EomtDinov3ForUniversalSegmentation", _DummyModelLoader)
    monkeypatch.setattr(sam3, "Sam3Processor", _DummyProcessorLoader)
    monkeypatch.setattr(sam3, "Sam2Processor", _DummyProcessorLoader)
    monkeypatch.setattr(sam3, "EomtImageProcessor", _DummyProcessorLoader)


def test_sam_singleton_uses_mps_when_runtime_reports_mps(monkeypatch):
    """SAM should place the model on MPS when the runtime selects it."""
    _patch_transformer_loaders(monkeypatch)
    monkeypatch.setattr(
        sam3.pytorch_runtime,
        "init",
        lambda: PytorchInstance(device="mps", cuda_available=False, mps_available=True),
    )
    monkeypatch.setattr(
        sam3.SamSingleton,
        "_enable_mps_roi_align_cpu_fallback",
        staticmethod(lambda: None),
    )

    instance = sam3.SamSingleton(model_name="sam3")

    assert instance.device == "mps"
    assert instance.model.device == "mps"
    assert instance.mps_roi_align_fallback_enabled


def test_sam_singleton_keeps_mps_with_roi_align_fallback_patch(monkeypatch):
    """SAM3 should remain on MPS and enable op-level CPU fallback."""
    _patch_transformer_loaders(monkeypatch)
    called = {"value": False}

    def _mark_fallback_enabled():
        called["value"] = True

    monkeypatch.setattr(
        sam3.pytorch_runtime,
        "init",
        lambda: PytorchInstance(device="mps", cuda_available=False, mps_available=True),
    )
    monkeypatch.setattr(
        sam3.SamSingleton,
        "_enable_mps_roi_align_cpu_fallback",
        staticmethod(_mark_fallback_enabled),
    )

    instance = sam3.SamSingleton(model_name="sam3")

    assert instance.device == "mps"
    assert instance.model.device == "mps"
    assert called["value"]
    assert instance.mps_roi_align_fallback_enabled


def test_move_roi_align_boxes_to_cpu_accepts_tuple_input():
    """Fallback helper should support tuple-of-tensor boxes from unbind()."""
    boxes = (
        sam3.torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=sam3.torch.float32),
        sam3.torch.tensor([[1.0, 1.0, 2.0, 2.0]], dtype=sam3.torch.float32),
    )

    converted = sam3.SamSingleton._move_roi_align_boxes_to_cpu(boxes)

    assert isinstance(converted, tuple)
    assert len(converted) == 2
    assert converted[0].device.type == "cpu"
