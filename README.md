# JAX Animal Behavior System (JABS)

![JABS Screen Shot](img/jabs_screenshot.png)

## ReadTheDocs Tutorial and User Guide

https://jabs-tutorial.readthedocs.io/en/latest/index.html

[User Guide (Markdown)](docs/user-guide.md)

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

JABS was initially developed on Python 3.10. See the `pyproject.toml` for a list of required Python 
packages. These packages are available from the Python Package Index (PyPI).

Currently, JABS supports Python 3.10 through 3.14.

## Installation

This section describes how to install JABS as an end user. Developers should see the
[JABS Development](#jabs-development) section below for instructions on setting up a development environment.

### Create a Virtual Environment

We recommend installing JABS within a dedicated Python virtual environment to avoid conflicts with other packages.
You can create and activate a virtual environment using the following commands:

```bash
python -m venv jabs.venv

# Linux and macOS
source jabs.venv/bin/activate

# Windows (cmd)
jabs.venv\Scripts\activate.bat
```


### Install from PyPI (Recommended)

JABS can be installed directly from the Python Package Index:

```bash
pip install jabs-behavior-classifier
```

This will install JABS and all required dependencies automatically.

### Install from Source

If you want the latest development version or need to install a specific branch/commit:

#### From GitHub

```bash
pip install git+https://github.com/KumarLabJax/JABS-behavior-classifier.git
```

Specify a branch or commit if needed:

```bash
pip install git+https://github.com/KumarLabJax/JABS-behavior-classifier.git@branch-name
pip install git+https://github.com/KumarLabJax/JABS-behavior-classifier.git@commit-hash
```

#### From Local Clone

If you’ve cloned the JABS repository:

```bash
pip install .
```

#### Windows Setup Helpers

Two batch scripts are included for Windows users working with a local clone:

- **`setup_windows.bat`** — Creates a `jabs.venv` virtual environment and installs JABS.
- **`launch_jabs.bat`** — Activates the environment and launches the JABS GUI.

Double-click these scripts in Windows Explorer to run them.

### Enabling XGBoost Classifier

The XGBoost Classifier has a dependency on the OpenMP library. This does not ship with macOS. XGBoost should work "out 
of the box" on other platforms. On macOS, you can install libomp with Homebrew (preferred) with the following 
command `brew install libomp`. You can also install libomp from source if you can't use Homebrew, but this is beyond 
the scope of this Readme.

---

## Running JABS

After installation, the following commands are available in your environment:

- `jabs` — launch the JABS GUI  
- `jabs-init` — initialize a new JABS project directory or recompute features in an existing project 
- `jabs-classify` — run a trained classifier  
- `jabs-stats` — print accuracy statistics for a classifier  
- `jabs-export-training` — export training data from an existing JABS project
- `jabs-cli` - collection of smaller command line utilities

You can view usage information for any command with:

```bash
<jabs-command> --help
```

## Sample Data

We provide sample data for testing and demonstration purposes. You can download the sample data from
https://doi.org/10.5281/zenodo.16697331

---

## Singularity/Linux

We supply a tested pair of singularity definition files. The [first vm](vm/headless.def) is intended for command-line use on 
compute clusters when scaling inferences. The [second vm](vm/gui.def) is designed for interacting with the GUI in a 
portable environment. Please inspect the definition files for related linux packages to run the software.

## JABS Project Portability

We have 4 version numbers in our software:

* JABS Python package version. This gets bumped every release.
* Feature version. This gets bumped every time we change feature values or the format used to store 
calculated features.
* Classifier version. This gets bumped every time we change characteristics of classifiers.
* Prediction version. This gets bumped every time we change how predictions are stored.

### Long Term Support of JABS-based Classifiers

There are multiple JABS Classifier artifacts that have different compatibility and portability characteristics.

* Project folders. These are the most compatible for upgrades. The vast majority of our upgrades to JABS will allow
transparent upgrades (e.g. re-generation of features) within the project folder without user interaction. We will
provide instructions for changes that are not.
* Exported training data. These are compatible across computers, but should generally not be considered compatible
across JABS package versions. Once we add the appropriate version checks, the error message should be a bit more
clear when and why these aren't compatible across versions.
* Classifier pickle files. These are only compatible within a specific install of the package (e.g. mac will not
be compatible with windows). These are the serialized trained classifiers, so load really fast, but should not be 
considered portable beyond the computer and specific JABS install that created them.

Project folders are big, but are almost always compatible across JABS versions.

Exported classifiers are smaller and easier to move around, but might require the same JABS package version to run. These 
are good for sharing or archiving specific versions (e.g. a version we use in a paper). A comon use case is to export
training data from a project folder, transfer it to our HPC cluster, and then train a and run classifier using the 
`jabs-classify` command from same version of JABS that was used to export the training file.

Pickle files are tiny and efficient, but are not transferable across computers. We use these for large-scale 
predictions in pipelines (for example, using exported training data to train a classifier saved as a .pickle file, 
which can then be used to classify many videos as part of a pipeline).


## JABS Development

If you're interested in contributing to JABS or setting up a development environment:

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute, copyright information, and submission guidelines
- **[Development Guide](docs/DEVELOPMENT.md)** - Detailed technical documentation including:
  - Setting up a development environment with uv
  - Code style and standards
  - Feature extraction architecture
  - Testing guidelines
  - Building and distribution
  - CI/CD and release management

