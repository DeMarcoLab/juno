# from juno_custom.element_template import ElementTemplate
# from juno.Lens import Lens

# FOCUSING_KEYS = ["height", "width", "exponent", "coefficient", "pixel_size"]

# class FocusingLens(ElementTemplate):
#     def __init__(self, lens: Lens) -> None:
#         super().__init__(lens)
#         self.name = "Focusing"

#     def __repr__(self) -> str:
#         return f"""Focusing Lens (Exponent = 2.0)"""

#     def generate_profile(self):
#         self.lens.exponent = 2
    
#     def analyse(self):
#         print("Focusing")

#     def __keys__(self) -> list:
#         return FOCUSING_KEYS
