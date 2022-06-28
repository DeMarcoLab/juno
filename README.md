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


### Define your mediums
Define the refractive index of the Lens and Output mediums.
```yaml
mediums:
  - name: medium_1
    refractive_index: 2.348
```
### Define your lenses
Define the medium and profile for each lens.
```yaml
lenses:
  - name: lens_1
    height: 70.e-6
    exponent: 0.0
    medium: medium_1
```
The name of the medium needs to be the same as defined in mediums.

### Define your simulation stages
Define the stages for the simulation to run. Simulation will be run in the order these stages are defined in.

```yaml
stages:
  - lens: lens_1
    output: medium_1 
    n_steps: 100 
    start_distance: 0
    finish_distance: 10.0e-3 
    options: 
      use_equivalent_focal_distance: False
      focal_distance_multiple: 2.0
  - lens: lens_2
    output: medium_2 
    n_steps: 1000 
    start_distance: 0
    finish_distance: 10.0e-3 
    options: 
      use_equivalent_focal_distance: True
      focal_distance_multiple: 2.0 
```

## Simulating
To run the simulation:

Open a terminal
```bash
$ python run_simulation config.yaml
```

Simulations results will be saved into the log/ directory. 



### Citation 
TODO