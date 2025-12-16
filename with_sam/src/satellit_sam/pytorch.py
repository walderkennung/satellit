import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class PytorchInstance:
    device: str
    cuda_available: bool
    mps_available: bool

    def debug_info(self) -> None:
        """Log debug information about the PyTorch configuration."""
        logger.debug("PyTorch Configuration:")
        logger.debug(f"  Version: {torch.__version__}")
        logger.debug(f"  Device: {self.device}")
        logger.debug(f"  CUDA available: {self.cuda_available}")
        if self.cuda_available:
            logger.debug(f"    CUDA version: {torch.version.cuda}")
            logger.debug(f"    CUDA device count: {torch.cuda.device_count()}")
            logger.debug(f"    CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.debug(f"  MPS available: {self.mps_available}")
        logger.debug(f"  Default dtype: {torch.get_default_dtype()}")


def init() -> PytorchInstance:
    # Initialize PyTorch
    torch.set_default_dtype(torch.float32)

    # Also patch as_tensor for SAM's internal calls
    _original_as_tensor = torch.as_tensor

    def _patched_as_tensor(data, dtype=None, device=None):
        if device is not None and "mps" in str(device):
            if dtype == torch.float64:
                dtype = torch.float32
            elif dtype is None and hasattr(data, "dtype") and data.dtype == "float64":
                dtype = torch.float32
        return _original_as_tensor(data, dtype=dtype, device=device)

    torch.as_tensor = _patched_as_tensor

    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    device = "mps" if mps_available else "cuda" if cuda_available else "cpu"

    return PytorchInstance(
        device=device,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
