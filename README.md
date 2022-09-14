<figure>
  <img
  src="juno/ui/logo.png"
  alt="Juno Simulation"
  width="750">
</figure>

# Juno
juno is python package for designing lenses by performing full wave simulations.

## Getting Started

### Installation

The best way to install is by creating an [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment. First download and install anaconda.

On Windows - Open Anaconda Powershell Prompt:

On Linux/Mac - Open Terminal:

``` bash
$ git clone https://github.com/DeMarcoLab/juno.git
$ cd juno
$ conda env create -f environment.yml
$ conda activate juno
$ pip install -e .

```

### Tutorial
For a tutorial walkthrough for using the package please see [Tutorial.md](TUTORIAL.md)


### User Interface

To run the user interface:
```bash
$ juno_ui
```

### Examples
The example folder contains a few simulation configurations using common lens types and setups:
 - Focusing Lens (1D and 2D)
 - Axicon Lens (2D)
 - Telescope (1D and 2D)

For more information about all the available configuration parameters and options, please see [Configuration.md](Configuration.md).


### Command Line

Simulations can be run from the commmand line.
```bash
$ python run_simulation.py config.yaml
```
This is useful for running large parameter sweeps on HPC setups.


### Documentation
For more detailed documentation please see [Documentation.md](Documentation.md)


## Citation
TODO: details

## Tests
[![Python package](https://github.com/DeMarcoLab/juno/actions/workflows/python-package.yml/badge.svg)](https://github.com/DeMarcoLab/juno/actions/workflows/python-package.yml)