# JAX Animal Behavior System (JABS)

![JABS Screen Shot](img/jabs_screenshot.png)

## ReadTheDocs Tutorial and User Guide

https://jabs-tutorial.readthedocs.io/en/latest/index.html

[User Guide (Markdown)](https://github.com/KumarLabJax/JABS-behavior-classifier/blob/main/src/jabs/resources/docs/user_guide/user_guide.md)

## Copyright

Copyright 2023 The Jackson Laboratory -- All rights reserved.

## Contact

email us at jabs@jax.org

## License

JABS is licensed under a non-commercial use license, see LICENSE for more information. Contact us for information about
licensing for commercial use.

## Pose Files

JABS requires pose files generated from the Kumar Lab's mouse pose estimation neural networks. Single mouse pose files 
are generated from [this repository](https://github.com/KumarLabJax/deep-hrnet-mouse). Multi-mouse is still under development. Contact us for more information.

## Requirements

JABS was initially developed and tested on Python 3.10. See the `pyproject.toml` for a list of required Python 
packages. These packages are available from the Python Package Index (PyPI).

Currently, JABS supports Python 3.10, 3.11, 3.12, and 3.13.

## Python Env Setup

We recommend creating a Python Virtualenv for JABS:

```
python -m venv jabs.venv

# Linux and MacOS
source jabs.venv/bin/activate

# Windows
jabs.venv\Scripts\activate.bat
```

### JABS Installation

Developers should follow the Developer Setup section below. This section describes how to install JABS into a Python 
environment for a non-developer user.

#### PyPI

JABS is not available on PyPI at this time, but we hope to begin publishing it there soon.

#### Pip install from Github

With the jabs.venv virtualenv activated, run the following command to install JABS from our git repository. This will 
install the latest commit from the main branch: 
`pip install git+https://github.com/KumarLabJax/JABS-behavior-classifier.git`

you can also specify a branch or tag:

`pip install git+https://github.com/KumarLabJax/JABS-behavior-classifier.git@branch-name`

or a specific commit:

`pip install git+https://github.com/KumarLabJax/JABS-behavior-classifier.git@commit-hash`

#### Pip install from local source

If you've cloned the JABS repository, you can install by running the following command in the project root directory:

`pip install .`

If you want to install the package in "editable" mode, use the `-e` flag. This links to the python files instead of 
copying them during installation so that they can be edited.

`pip install -e .`

#### Windows .bat scripts

There are two scripts that Windows users can use to simplify installing and running JABS. These can be executed by 
double-clicking on them in Windows Explorer.

* setup_windows.bat: this will create a Python virtualenv called jabs.venv in the project root and then install JABS as 
a Python package.
* launch_jabs.bat: this script will activate the jabs.venv environment and then launch the JABS GUI.

### Running JABS

After installing JABS, five commands will be added to the bin directory of your Python virtualenv:

* jabs: launch the JABS GUI
* jabs-init: initialize a new JABS project directory from the command line
* jabs-classify: run a trained classifier from the command line
* jabs-stats: print accuracy statistics for the given classifier
* jabs-convert-parquet: convert parquet pose file to JABS pose file format

You can run the <command> --help to get usage information for each of the commands.

**NOTE: On some platforms, the first time you run the JABS GUI it might take several minutes to launch. Subsequent 
startup times should be significantly reduced.**

### Developer Setup

The following instructions are for Linux or macOS Developers. Commands for JABS developers using Windows might be 
slightly different.

This project uses Poetry for packaging and dependency management. JABS developers will need to install Poetry by 
following the instructions on [Poetry's official website](https://python-poetry.org/docs/#installation).

You can use Poetry to manage your virtualenv, or manage your virtualenv externally to Poetry and use Poetry only for 
installing dependencies. The following instructions assume that you've already created and activated a Python 
environment for JABS using whichever method you prefer.

Clone the JABS git repository, and with your JABS virtualenv activated, run the following command in the project root:

```commandline
poetry install
```

This will install all dependencies and JABS will be installed in "editable" mode, meaning that the JABS Python modules 
installed in the virtualenv will be links to the files in the cloned git repository. JABS code changes will be 
reflected immediately in the Python environment.

Note to Developers: JABS uses package metadata to determine the version number. If you change the version number in the 
pyproject.toml file, you will need to run `poetry install` to update the version number in the installed package so 
that the GUI will display the correct version.

#### Code Style

JABS uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Developers should run `ruff check` and `ruff format` before 
committing code. A pre-commit hook is provided to run these commands automatically.

To install pre-commit hooks for linting and formatting run:

```commandline
pre-commit install
```

You can also run [ruff](https://docs.astral.sh/ruff/) directly from command line:

```commandline
ruff check src/packagepath/modulename.py
ruff format src/packagepath/modulename.py
```

#### Building Python Packages

Developers can build a Python package using the `poetry build` command. This will produce both a .tar.gz and a Python 
Wheel file (.whl) in the dist directory. The wheel file can be installed with pip:

```pip install jabs_behavior_classifier-<version>-py3-none-any.whl```

Since the Wheel does not contain any compiled code it is platform independent.

### Enabling XGBoost Classifier

The XGBoost Classifier has a dependency on the OpenMP library. This does not ship with macOS. XGBoost should work "out 
of the box" on other platforms. On macOS, you can install libomp with Homebrew (preferred) with the following 
command `brew install libomp`. You can also install libomp from source if you can't use Homebrew, but this is beyond 
the scope of this Readme.

### Singularity/Linux

We supply a tested pair of singularity definition files. The [first vm](vm/behavior-classifier-vm.def) is intended for command-line use on 
compute clusters when scaling inferences. The [second vm](vm/behavior-classifier-gui-vm.def) is designed for interacting with the GUI in a portable 
environment. Please inspect the definition files for related linux packages to run the software.

