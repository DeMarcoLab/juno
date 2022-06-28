# Configuration

The available simulation configuration parameters are as follows:

## Simulation Parameters:
```yaml
sim_parameters:
  A: 10000
  pixel_size: 1.e-6 
  sim_width: 500.e-6
  sim_height: 500.e-6
  sim_wavelength: 488.e-9

```
The configurable global simulation parameters are:

- A: The initial amplitude of the wavefront. 
- Pixel Size: The pixel size of the simulation (metres).
- Sim Width: The width of the simulation (metres).
- Sim Height: The height of the simulation (metres).
- Sim Wavelength: The wavelength of light used in the simulation (metres).

## Simulation Options

```yaml
options:
  log_dir: log
  save_plot: true
  debug: false

```

The configurable simulation options are:
- Log Dir: The directory to save the simulation results and plots.
- Save Plot: Flag to save the simulation intermediate and result plots.
- Debug: Flag to debug the simulation, which provides additional information (bool).

## Beam Parameters

```yaml
beam:
  distance_mode: "direct"   # "Direct", "Width", "Focal"
  spread: "plane"           # "Plane", "Converging", "Diverging"
  shape: "rectangular"        # "Circular", "Rectangular"
  width: 300.e-6 
  height: 300.e-6 
  position_x: 100.e-6 
  position_y:  0.e-6
  theta: 45.0 
  numerical_aperture: null 
  tilt_x: 0.0 
  tilt_y: 0.0 
  source_distance: 0.1e-3
  final_diameter: 10.0e-6 
  focal_multiple: null 
  n_steps: 2
  step_size: 0.001

```


The configurable Beam parameters are:
- Width: The initial width of the beam (metres).
- Height: The initial height of the beam (metres).
- DistanceMode: the method for calculating the beam propagation distance [Direct, Diameter, Focal].
  - Direct: Propagate for a fixed distance
  - Diameter: Propagate based on the final diameter of the beam.
  - Focal: Propagate based on the focal distance of the equivalent lens.
- Spread: The spread of the beam propagation [Plane, Converging, Diverging]
  - Plane: A planar beam
  - Converging: A converging beam
  - Diverging: A diverging beam.
- Shape: The shape of the beam propagation [Rectangular, Circular]
  - Rectangular: Rectangular beam shape [Spread.Plane]
  - Circular: Circular beam shape [Spread.Plane, Spread.Converging, Spread.Diverging]
- Position X: The position of the beam from the centre of the simulation in the x-axis (metres).
- Position Y: The position of the beam from the centre of the simulation in the y-axis (metres). 
- Theta: The convergence angle of the beam (degrees). [Spread.Converging, Spread.Diverging] 
- Numerical Aperture: The numerical aperture of the beam (dimensionless). [Spread.Converging, Spread.Diverging] 
- Tilt X: The tilt of the beam propagation along the x-axis (degrees).
- Tilt Y: The tilt of the beam propgation along the y-axis (degrees).
- Source Distance: The total propgation distance of the beam [DistanceMode.Direct].
- Final Diameter: The final diameter of the beam after propagation [DistanceMode.Diameter]. 
- Focal Multiple: The multiple of focal length to propagate the beam [DistanceMode.Focal].
- Num Steps: The number of propagation steps.
- Step Size: The distance between propagation steps. (metres)

Some paramters are mutually exclusive, and others will only be used based on other parameters. For example; theta and numerical aperture are mutually exclusive (theta will be used if both are defined), and are only used for Converging/Diverging beam spreads.

## Lens Parameters

```yaml
lenses:
- name: lens_1
  medium: 2.348 
  diameter: 300.e-6 
  height: 10.e-6 
  exponent: 2.0 
  lens_type:  "Cylindrical"     # Cylindrical, Spherical 
  length: 30.e-9
  escape_path: 0.1
  inverted: false 
  grating: 
    width: 10.e-6 
    distance: 20.e-6 
    depth: 0.15e-6  
    x: true
    y: false
    centred: true
  truncation: 
    height: 3.e-6  
    radius: 50.e-6  
    type: radial # radial, value
    aperture: false
  aperture:
    inner: 50.e-6  
    outer: 100.e-6  
    type: radial   # radial, square
    invert: false
```


The configurable lens parameters are:

- Name: The name of the lens (this must match the stage config).
- Medium: The refractive index of the medium the lens is made from.
- Diameter: The diameter of the lens (metres).
- Height: The height of the lens (metres).
- Exponent: The exponent of the lens profile. 
- LensType: The type of lens to create [LensType.Spherical, LensType.Cylindrical].
- Length: The length of the cylindrical lens (metres) [LensType.Cylindrical].
- Escape Path: The fraction of the lens diameter to add as an escape path (e.g. 0.1 = 10% of diameter).
- Inverted: Flag to invert the lens profile (bool)
- Grating: Apply diffraction gratings to the lens
    - Width: The width of the grating (metres)
    - Distance: The distance between gratings (metres)
    - Depth: The depth of the gratings (metres)
    - X: Flag to apply gratings along the x-axis (bool)
    - Y: Flag to apply gratings along the y-axis (bool)
    - Centred: Flag to centre the gratings (bool)
- Truncation: Truncate the height of the lens
    - Type: The truncation method to apply [value, radial]
    - Height: The height to truncate the lens [Type.value]
    - Radius: The radius to truncation the lens [Type.radial]
    - Aperture: Flag to apply an aperture on the truncated area (bool).
- Aperture: Aperture an area of the lens
    - Type: The aperture method to apply [radial, square]
    - Inner: The inner dimension of the aperture (metres).
    - Outer: The outer dimension of the aperture (metres).
    - Invert: Flag to invert the apertured area (i.e. non-apertured areas become apertured) (bool).


**Experimental Parameters**

The lens config also provides two experimental parameters that can be used without any guarantees:

```yaml
custom: /path/to/custom/profile.npy
custom_config: /path/to/custom/config.yaml
```
These parameters work in the following way
- Custom: Loads a numpy array as the lens profile, allowing the use of any profile not supported by package tools. (.npy) If a .npy file is provided, the profile is loaded. If a directory is provided, the Simulation Runner will sweep through each custom profile in the directory. 
- Custom Config: Override the current lens config with the config provided (.yaml). This is useful for passing through metadata regarding custom loaded profiles, and running larger sweeps on these profiles. 

N.B. These features aren't officially supported and are highly experimental and subject to change.

## Stage Parameters

```yaml
stages:
  - lens: lens_1
    output: 1.0 
    n_steps: 100 
    start_distance: 0
    finish_distance: 2.0e-3 
  - lens: lens_2
    output: 1.33 
    step_size: 0.1e-3 
    use_equivalent_focal_distance: true
    focal_distance_start_multiple: 0.0
    focal_distance_multiple: 2.0 

```

The configurable simulation stage parameters are:

- Lens: The name of the lens to use. This must match that found in the lens section.
- Output: The refractive index of the medium to propagate into. 
- Num Steps: The number of steps to propagate the wave.
- Step Size: The distance between steps of the wave propagation (metres).
- Start Distance: The distance to start simulating the wave propagation (metres).
- Finish Distance: The distance to finish simulating the wave propagation (metres).
- Use Equivalent Focal Distance: Flag to use the equivalent focal distance to calculate the distances. (bool)
- Focal Distance Start Multiple: The multiple of focal distance to use for the start distance. 
- Focal Distance Multiple:  The multiple of the focal distance to use for the finish distance. 
