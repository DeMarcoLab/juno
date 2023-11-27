# from juno.Lens import Lens

# from juno_custom.element_template import ElementTemplate


# AXICON_KEYS = ["height", "width", "exponent", "coefficient", "pixel_size"]

# class Axicon(ElementTemplate):
#     def __init__(self, lens: Lens) -> None:
#         super().__init__(lens)
#         self.name = "Axicon"

#     def __identifier__(self) -> str:
#         return f"""Axicon"""

#     def __repr__(self) -> str:
#         return f"""Axicon (Exponent = 1.0)"""

#     def generate_profile(self):
#         self.lens.exponent = 1

#     def analyse(self):
#         print("Axicon analysis.")

#     def __keys__(self) -> list:
#         return AXICON_KEYS