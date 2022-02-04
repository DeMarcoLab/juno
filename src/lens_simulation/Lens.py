
import numpy as np

class Lens:
    def __init__(self, width: float, height: float, exponent: float) -> None:

        self.width = width
        self.height = height
        self.exponent = exponent

    def __repr__(self):

        return f""" Lens (width: {self.width}, height: {self.height}, medium: {self.exponent}"""

    def generate_profile(self) -> np.ndarray:
        """[summary]

        Returns:
            np.ndarray: [description]
        """

        radius = self.width / 2
        # n_pixels = 10000
        radius_step_size = 10e-6
        n_pixels = int(radius / radius_step_size)

        radius_px = np.linspace(0, radius, n_pixels)
    
        # axicon only
        # heights = (self.height / self.width ) * radius_px
        coefficient = self.height / (radius ** self.exponent)

        heights = self.height - coefficient * radius_px ** self.exponent
        
        profile =  np.append(np.flip(heights[1:]), heights)
        profile = profile - np.max(profile) # 

        return profile
    