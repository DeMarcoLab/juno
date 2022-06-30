# """Basic example of using magicgui to create an Image Arithmetic GUI in napari."""
# from enum import Enum

# import napari
# import numpy
# from napari.layers import Image
# from napari.types import ImageData

# from magicgui import magicgui


# class Operation(Enum):
#     """A set of valid arithmetic operations for image_arithmetic.

#     To create nice dropdown menus with magicgui, it's best (but not required) to use
#     Enums.  Here we make an Enum class for all of the image math operations we want to
#     allow.
#     """

#     add = numpy.add
#     subtract = numpy.subtract
#     multiply = numpy.multiply
#     divide = numpy.divide


# # create a viewer and add a couple image layers
# viewer = napari.Viewer()
# viewer.add_image(numpy.random.rand(20, 20), name="Layer 1")
# viewer.add_image(numpy.random.rand(20, 20), name="Layer 2")


# # for details on why the `-> ImageData` return annotation works:
# # https://napari.org/guides/magicgui.html#return-annotations
# @magicgui(call_button="execute", layout="horizontal")
# def image_arithmetic(layerA: Image, operation: Operation, layerB: Image) -> ImageData:
#     """Add, subtracts, multiplies, or divides to image layers with equal shape."""
#     return operation.value(layerA.data, layerB.data)


# # add our new magicgui widget to the viewer
# viewer.window.add_dock_widget(image_arithmetic, area="bottom")


# # note: the function may still be called directly as usual!
# # new_image = image_arithmetic(img_a, Operation.add, img_b)

# napari.run()


"""Example showing how to accomplish a napari parameter sweep with magicgui.

It demonstrates:
1. overriding the default widget type with a custom class
2. the `auto_call` option, which calls the function whenever a parameter changes

"""
import napari
import skimage.data
import skimage.filters
from napari.layers import Image
from napari.types import ImageData

from magicgui import magicgui

# create a viewer and add some images
viewer = napari.Viewer()
# viewer.add_image(skimage.data.astronaut().mean(-1), name="astronaut")
# viewer.add_image(skimage.data.grass().astype("float"), name="grass")

from lens_simulation import plotting

path = r"C:\Users\pcle0002\Documents\repos\lens_simulation\src\lens_simulation\log\weekly-famous-lizard/fit-jennet"

full_sim = plotting.load_full_sim_propagation_v2(path)
# view = plotting.slice_simulation_view(full_sim, axis=0, prop=0.5)
viewer.add_image(full_sim, name="simulation")


# turn the gaussian blur function into a magicgui
# # for details on why the `-> ImageData` return annotation works:
# # https://napari.org/guides/magicgui.html#return-annotations
# @magicgui(
#     # tells magicgui to call the function whenever a parameter changes
#     auto_call=True,
#     # `widget_type` to override the default (spinbox) "float" widget
#     sigma={"widget_type": "FloatSlider", "max": 6},
#     # contstrain the possible choices for `mode`
#     mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
#     layout="horizontal",
# )
# def gaussian_blur(layer: Image, sigma: float = 1.0, mode="nearest") -> ImageData:
#     """Apply a gaussian blur to ``layer``."""
#     if layer:
#         return skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode)

from pathlib import Path

# turn the gaussian blur function into a magicgui
# for details on why the `-> ImageData` return annotation works:
# https://napari.org/guides/magicgui.html#return-annotations
@magicgui(
    # tells magicgui to call the function whenever a parameter changes
    auto_call=True,
    # `widget_type` to override the default (spinbox) "float" widget
    prop={"widget_type": "FloatSlider", "max": 1.0},
    axis={"choices": [0, 1, 2]},
    layout="horizontal",
)
def slice_image(layer: Image, prop: float = 0.5, axis: int = 0) -> ImageData:
    """Slice the volume along the selected axis"""
    if layer:
        return plotting.slice_simulation_view(layer.data, axis=axis, prop=prop)



# Add it to the napari viewer
viewer.window.add_dock_widget(slice_image, area="bottom")

napari.run()