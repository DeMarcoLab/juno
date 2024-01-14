import importlib
import logging

from juno.juno_custom.elements.available_elements import AVAILABLE_ELEMENTS
from juno.juno_custom.elements.element_template import ElementTemplate


def import_element_module(element: str):
    if element not in AVAILABLE_ELEMENTS:
        raise ValueError(f"Element {element} is not available")

    element_folder_name, element_name = AVAILABLE_ELEMENTS[element]
    module_name = f"juno.juno_custom.elements.{element_folder_name}"
    module = importlib.import_module(module_name)
    cls = getattr(module, AVAILABLE_ELEMENTS[element][1])
    logging.info(f"imported {element_name} {cls}")


def get_custom_elements():
    for element in AVAILABLE_ELEMENTS:
        import_element_module(element)
    return ElementTemplate.__subclasses__()
