from juno.Lens import Lens

from element_template import ElementTemplate

KEYS = []


class CustomLens(ElementTemplate):
    def __init__(self, lens: Lens) -> None:
        super().__init__(lens)
        self.name = "CustomLens"

    def __repr__(self) -> str:
        return f"""Custom Lens"""

    def generate_profile(self):
        pass

    def analyse(self):
        print("Custom analysis.")

    def __keys__(self) -> list:
        return KEYS
