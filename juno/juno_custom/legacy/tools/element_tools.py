import glob
import importlib
import os

from juno.Lens import Lens
from juno.Medium import Medium

import juno_custom.elements
from juno_custom.element_template import ElementTemplate

# These are the modules that will be ignored when importing custom lenses
IGNORED_MODULES = ["__init__", "template"]


def import_custom_elements():
    # get all the modules in the custom_lenses folder
    module_names = [
        os.path.basename(f)[:-3]
        for f in glob.glob(os.path.join(juno_custom.elements.__path__[0], "*.py"))
    ]

    # iterate through the modules, removing any that have a substring in IGNORED_MODULES
    module_names = [
        juno_custom.elements.__name__ + "." + module
        for module in module_names
        if not any([a in module for a in IGNORED_MODULES])
    ]

    # iterate through the modules and try to import them
    for module_ in module_names:
        try:
            importlib.import_module(module_)

        except Exception as e:
            print(f"Failed to import {module_}, reason being: {e}")


def get_custom_elements():
    import_custom_elements()
    return ElementTemplate.__subclasses__()


def generate_base_element(config: dict):
    diameter = config.get("diameter")
    height = config.get("height")
    medium = config.get("medium")
    exponent = config.get("exponent")
    base_element = Lens(diameter, height, exponent, Medium(medium))
    base_element.generate_profile(config.get("pixel_size"))
    return base_element


def generate_element(config: dict):
    base_element = generate_base_element(config=config)
    element = ElementFactory.create_element(
        element_type=config.get("element_type"), lens=base_element
    )
    return element


def generate_element_config(config: dict):
    config["lenses"] = {}


class ElementFactory:
    @staticmethod
    def create_element(element_type: str = None, lens: Lens = None):
        if element_type is None:
            raise ValueError("Element type must be provided.")

        for custom_element in get_custom_elements():
            if element_type.lower() == custom_element.__name__.lower():
                return custom_element(lens)
        raise ValueError(f"Element type {element_type} not found.")
