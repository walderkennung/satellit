"""Shared fixtures for tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from satellit_sam.image_processing import Image


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def forest_image_path(test_data_dir):
    """Return the path to the forest test image."""
    return test_data_dir / "forest1.jpg"


@pytest.fixture
def forest_image(forest_image_path):
    """Load and return the forest test image."""
    return Image.load(str(forest_image_path))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_test_image():
    """Create a small synthetic test image (100x100 RGB)."""
    data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image(size=(100, 100), channels=3, data=data)


@pytest.fixture
def medium_test_image():
    """Create a medium synthetic test image (512x512 RGB)."""
    # Create a gradient pattern for visual verification
    x = np.linspace(0, 255, 512)
    y = np.linspace(0, 255, 512)
    xx, yy = np.meshgrid(x, y)

    r = xx.astype(np.uint8)
    g = yy.astype(np.uint8)
    b = ((xx + yy) / 2).astype(np.uint8)

    data = np.stack([r, g, b], axis=-1)
    return Image(size=(512, 512), channels=3, data=data)


@pytest.fixture
def rgba_test_image():
    """Create a small RGBA test image."""
    data = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    return Image(size=(100, 100), channels=4, data=data)
