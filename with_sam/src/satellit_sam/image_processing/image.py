from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Image:
    size: tuple[int, int]  # (width, height)
    channels: int
    data: np.ndarray

    def copy(self) -> "Image":
        """Return a copy of the image."""
        return Image(size=self.size, channels=self.channels, data=self.data.copy())

    def save(self, path: str) -> None:
        """Save the image to the specified path."""
        if self.channels == 4:
            cv2.imwrite(path, cv2.cvtColor(self.data, cv2.COLOR_RGBA2BGRA))
        elif self.channels == 3:
            cv2.imwrite(path, cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(path, self.data)

    @staticmethod
    def load(path: str) -> "Image":
        """Load an image from the specified path."""
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Image not found at path: {path}")

        # Handle both RGB and RGBA images
        if len(raw.shape) == 2:
            # Grayscale image
            data = raw
            height, width = data.shape
            channels = 1
        elif raw.shape[2] == 3:
            # RGB image
            data = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            height, width, channels = data.shape
        else:
            # RGBA image
            data = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
            height, width, channels = data.shape

        return Image(size=(width, height), channels=channels, data=data)

    def crop(self, start_x: int, start_y: int, end_x: int, end_y: int) -> "Image":
        """Crop the image to the specified rectangle."""
        cropped_data = self.data[start_y:end_y, start_x:end_x]
        height, width, channels = cropped_data.shape
        return Image(size=(width, height), channels=channels, data=cropped_data)
