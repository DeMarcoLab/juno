
import numpy as np


class Lens:
    def __init__(self, width, height, medium) -> None:

        self.width = width
        self.height = width
        self.medium = medium

    def generate_profile(self) -> np.ndarray:

        return NotImplemented
    
    def __repr__(self):

        return f"Lens (width: {self.width}, height: {self.height}, medium: {self.medium})"