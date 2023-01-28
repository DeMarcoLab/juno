# Documentation

This document contains the explanations for the simulation structure, coordinate systems and general overview

N.B: This software is currently in development and breaking changes may be made.

## Simulation Structure
Lens simulations operates using free space wave propagation.

The structure of the simulation can be clasified into three main components:
- Beam: The initial wave propagation into the system
- Lens: The lens profile under analysis 
- Output Medium: the output medium which the lens propagates into. 

<figure>
  <img
  src="/img/sim_config.png"
  alt="Simulation Structure Definitions">
  <figcaption style="text-align:center">Simulation Structure Definitions</figcaption>
</figure>


We refer to the collection of Lens -> Ouput Medium pairs as a simulation stage. Multiple simulation stages may be chained together to form long optical systems. 

Therefore, the simulation can be summarised as having the basic structure:

Beam -> Lens -> Output -> Lens -> Output

Beam -> Stage -> Stage (where Stage = Lens/Beam -> Output)

(Note: under the hood, the beam is also a stage that can be customised in the same way (e.g refractive index, propagation distances). However, it is a special case as it defines the initial propagation into the system and so it is better to think of it separately.)

## Implementation Details
The following describes the implementation details including coordinate systems used, and operations.

### Coordinate Systems
Propagation along the optical axis is refered to as the z-axis. The wave propagates along the z-axis. The width (x-axis) and height (y-axis) are defined perpendicular to the z-axis. The beam and lens are defining in the x-y coordinate system, with the height extending into the z-axis.

- z-axis: out of the page, positive out
- x-axis: horizontal, positive right
- y-axis: vertical, positive down

<figure>
  <img
  src="/img/coordinate_system.png"
  alt="Coordinate System">
  <figcaption style="text-align:center">Coordinate System</figcaption>
</figure>

All properties of lens, beams, or stages are defined in a similar manner. In general the definitions follow the convention:
- width/diameter : x-axis
- height: y-axis
- depth: z-axis

### Global Parameters and Padding 
A core set of parameters define the limits of the simulation, and restrict the elements that can be simulated and influence simulation performance. 

**Global Parameters**

The simulation shares a number of core parameters:
- Simulation Width
- Simulation Height
- Simulation Pixel Size
- Simulation Wavelength
- Simulation Amplitude 

These parameters are fixed across the entire simulation, and all stages share these parameters for definition. These parameters restrict the maximum size of the lenses, and beams that can be used in the simulation.  

**Padding**

Lenses must fit within the simulation (width / height). However, lenses that are smaller than the simulation will be padded to the simulation size. This padding is apertured and prevents light from propagating through (please see the aperture section for details on the apertures). The simulation will raise an error if you try to load a lens larger than the simulation.

<figure>
  <img
  src="/img/lens_padding.png"
  alt="Lens Padding">
  <figcaption style="text-align:center">Lens Padding</figcaption>
</figure>

N.B. the lenses above are the same size.

In general, if you expect light to hit the 'walls' of the simulation you should make your simulation much larger (padding) than your lenses. This is because any light that touches the edge of the simulation will wrap around to the other side of the simulation, invalidating the results. (N.B. this will likely be dealt with in a future iteration, but currently is undefined behaviour.) 


### Lens Creation
The creation of lenses is separated into two parts: profile generation and applying modifications.

**Profile Generation**

The profile of a lens can be generated in two ways: generated from parameters or from a numpy array.

- To generate a profile, choose the required parameters: diameter, exponent, height, lens_type, medium.
- To load a numpy array, select the .npy file to load. It is your responsibility to ensure that the sizes and shapes of the lense fit within the simulation.

<figure>
  <img
  src="/img/lens_profile.png"
  alt="Lens Profile">
  <figcaption style="text-align:center">Lens Profile 2D</figcaption>
</figure>


For more detail about the lens configuration please see [Configuration.md](Configuration.md)

**Modifications**

Modifications are additional operations that can be applied to a lens profile, and include:
- Inversion
- Escape Path 
- Gratings
- Truncation
- Aperture

All modifications are calculated based on the base profile, and are applied independently. For example, a truncation will be calculated on the base profile height, and not after applying gratings.

Specifically, the lens are generated in the following order:
1. Generate / Load Profile
2. Invert Profile
3. Create Escape Path
4. Calculate Grating, Truncation, and Aperture modifications
5. Apply Gratings
6. Apply Truncation
7. Apply Apertures

<figure>
  <img
  src="/img/lens_modifications.png"
  alt="Lens Modifications">
  <figcaption style="text-align:center">Lens Modifications</figcaption>
</figure>

**Aperture**

Apertures are a special type of modification that prevent light from propagating through. There are several kinds of apertures than can be applied to lenses:

1. Non Lens Area: The area outside of the lens profile, but within the rectangular bounds of the profile dimensions. It is automatically apertured.
2. Custom Aperture: The aperture defined by the user in the configuration. 
3. Truncation Aperture: The aperture defined by the user in the configuration, if truncation aperture is selected.
4. Simulation Aperture: The area outside the lens profile, but within the simulation bounds (i.e. padded area). It is automatically apertured.
5. Loaded Aperture: The aperture that has been loaded alongside a loaded lens profile (see loading profiles section). 


<figure>
  <img
  src="/img/lens_aperture.png"
  alt="Lens Aperture">
  <figcaption style="text-align:center">Lens Aperture</figcaption>
</figure>


All these aperture plots are available in the logging directory once the simulation completes.

**Loading Profiles**

If you are unable to create a lens using the available tools in the package, you can create it by some other means and load the profile as a numpy array. 

To load a custom profile:
- In the lens config:
    - add the "custom" key and provide a path to the lens profile .npy file.
    - (optional) add a custom aperture to the same directory as the lens profile, and rename it to {profile_name}.aperture.npy. This aperture will be loaded and applied to your custom lens profile.

For example, your config and file system should look like:


```bash

path/to/custom/lens/
    profile.npy

```

```yaml

lenses:
- name: custom_lens:
    ...
    custom: path/to/custom/lens/profile.npy
...
```

It is your responsibility to check that loaded profiles match the simulation dimensions.

### Beam Creation

The beams share the same coordinate system as Lenses, with two additional properties, tilt, and convergence.

**Tilt**
The beam tilt refers to the initial angle the of the beam compared to the propagation axis (z-axis). The beam supports two different tilts, x and y and they are defined as follows:

<figure>
  <img
  src="/img/beam_tilt.png"
  alt="Beam Tilt">
  <figcaption style="text-align:center">Beam Tilt</figcaption>
</figure>

An alternative way to think about this, is that the beam can pitch and yaw, but cannot roll.
- y-tilt controls pitch.
- x-tilt controls yaw.

TODO: double check the wording on this.

**Convergence**
The convergence angle (theta) is defined as half the total internal propaagation angle. This defines the convergence / divergence of the beam.

<figure>
  <img
  src="/img/convergence.png"
  alt="Beam Convergence">
  <figcaption style="text-align:center">Beam Convergence</figcaption>
</figure>



## User Interface
We provide several user interfaces for creating each element of the simulation.

### Lens Creator
This interface allows you to create, save and load lens profiles. Standard profiles are defined by a diameter, height, exponent and lens type (cylindrical or spherical). Additional modifiations such as gratings, escape paths, truncation and apertures also be applied.

<figure>
  <img
  src="/img/tutorial_lens.png"
  alt="Lens Creator">
  <figcaption style="text-align:center">Lens Creator</figcaption>
</figure>

### Beam Creator
This interface allows you to create and save the initial beam setup. A beam defines the initial propagation into the simulation and can be modified in a number of ways, including size, shape, spread, distance, tilt and other properties.

<figure>
  <img
  src="/img/tutorial_beam.png"
  alt="Beam Creator">
  <figcaption style="text-align:center">Beam Creator</figcaption>
</figure>

### Simulation Setup
This interface allows you to create, save and load a simulation configuration. The configuration defines your simulation setup and includes:
- Global Simulation Parameters: e.g. simulation width, height, pixel_size and wavelength
- Simulation Options: e.g. logging directory, simulation name
- Beam Settings: the beam configuration
- Stage Settings: a combination of a lens configuration and output settings, e.g. output medium, propaagation distances and step sizes. 

This interface also allows you to setup parameter sweeps for most numerical parameters in the simulation. This allows you to run a large number of simulation combinations easily.

<figure>
  <img
  src="/img/tutorial_sim_setup_3.png"
  alt="Simulation Setup">
  <figcaption style="text-align:center">Simulation Setup</figcaption>
</figure>


### Run Simulation
This interface allows you to load a simulation config and run the simulations. A progress bar will display the overall progress of the simulations.

<figure>
  <img
  src="/img/tutorial_sim_run.png"
  alt="Run Simulation UI">
  <figcaption style="text-align:center">Run Simulation UI</figcaption>
</figure>


### Visualise Results
This interface allows you to load multiple simulations, filter the results, and visualise the propagation. 

<figure>
  <img
  src="/img/tutorial_sim_visualisation_3.png"
  alt="Simulation Visualisation UI">
  <figcaption style="text-align:center">Simulation Visualisation UI</figcaption>
</figure>

The interface also provides functionality to load the full simulation propagation in napari for interactive exploration. For simulations of the same shape, you can explore these all at once by pressing the view all button. 

<figure>
  <img
  src="/img/tutorial_sim_vis_napari.png"
  alt="Simulation Visualisation Napari">
  <figcaption style="text-align:center">Simulation Visualisation Napari</figcaption>
</figure>


## Additional Notes

### "Double Sided Lens"
If you define a simulation stage that contains a lens and output with the same medium, the simulation will assume you want to create a double sided lens. This will try to invert the lens profile and medium at the inlet, by constructing the inverted profile from the previous output's medium instead. 

<figure>
  <img
  src="/img/invert_lens.png"
  alt="Double Sided Lens">
  <figcaption style="text-align:center">Double Sided Lens</figcaption>
</figure>
