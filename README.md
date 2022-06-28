# [PACKAGE_NAME]
 [PACKAGE_NAME] is python package for designing lens by performing full wave simulations.  

<figure>
  <img
  src="doc/img/sim.png"
  alt="Simulation Image">
  <figcaption style="text-align:center">Simulation Image</figcaption>
</figure>

## Getting Started


### Installation

The best way to install is by creating an [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment. First download and install anaconda. 

On Windows - Open Anaconda Powershell Prompt:

On Linux/Mac - Open Terminal:

``` bash
$ git clone https://github.com/DeMarcoLab/lens_simulation.git
$ cd lens_simulation
$ conda create --name lens_sim python=3.8 pip
$ conda activate lens_sim
$ pip install -r requirements.txt
$ pip install -e .

```

### User Interface

To run the user interface:
```bash
$ cd src/lens_simulation
$ python ui/main.py
```


Create Lens

Create Beam

Setup Simulation

Run Simulation

Visualise Results


### Examples
The example folder contains a few simulation configurations using common lens types and setups:
 - Focusing Lens (1D and 2D)
 - Axicon Lens (2D)
 - Telescope (1D and 2D)



## Configuration


In the config.yaml file:

### Define your simulation parameters

```yaml
sim_parameters:
  A: 10000
  pixel_size: 1.e-6 
  sim_width: 4500.e-6
  sim_wavelength: 488.e-9

```
### Define your lenses
Define the medium and profile for each lens.
```yaml
lenses:
  - name: lens_1
    height: 70.e-6
    exponent: 0.0
    medium: 2.348
```
### Define your simulation stages
Define the stages for the simulation to run. Simulation will be run in the order these stages are defined in.

```yaml
stages:
  - lens: lens_1
    output: 1.0 
    n_steps: 100 
    start_distance: 0
    finish_distance: 10.0e-3 
    options: 
      use_equivalent_focal_distance: False
      focal_distance_multiple: 2.0
  - lens: lens_2
    output: 1.33 
    n_steps: 1000 
    start_distance: 0
    finish_distance: 10.0e-3 
    options: 
      use_equivalent_focal_distance: True
      focal_distance_multiple: 2.0 
```
The name of the lens must match that found in the lenses section.



### Command Line

Simulations can be run from the commmand line.
```bash
$ python run_simulation.py config.yaml
```
This is useful for running large parameter sweeps on HPC setups.



### Technical Details


ZARR








## Citation 
TODO