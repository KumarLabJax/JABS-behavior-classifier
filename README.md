# JAX Animal Behavior System (JABS)

## ReadTheDocs Tutorial
https://jabs-tutorial.readthedocs.io/en/latest/index.html

## Copyright

Copyright 2021 The Jackson Laboratory -- All rights reserved.

## Contact

email us at jabs@jax.org

## License

JABS is licensed under a non-commercial use license, see LICENSE for more 
information. Contact us for information about licensing for commercial use.

## Notice

This is beta software. Changes, including incompatible changes to exported
training data and prediction output, are forthcoming.

## Pose Files

JABS requires pose files generated from the Kumar Lab's mouse pose 
estimation neural network. Contact us for more information.

## Requirements
Developed and tested on Python 3.7, 3.8, and 3.9. See the `requirements.txt` 
for a list of required Python packages. These packages are available from the 
Python Package Index (PyPI)

### Python Virtual Environment

The following instructions are for Linux or MacOS Users. Windows users can 
follow the instructions in the "Windows" section below.

#### Creating the Virtual Environment

You will need to create the virtual environment before you can run the labeler 
for the first time. The following commands will create a new Python3 virtual 
environment, activate it, and install the required packages. Note, your python 
executable may be named `python` or `python3` depending on your installation.

```commandline
python -m venv jabs.venv
source jabs.venv/bin/activate
pip install -r requirements.txt
```

#### Activating 

The virtual environment must be activated before you can run the labeling 
interface. To activate, run the following command:

```commandline
source jabs.venv/bin/activate
```

#### Deactivating

The virtual environment can be deactivated if you no longer need it:

```commandline
deactivate
```

### Installing on Apple M1/M2 Silicone

To install the app on Apple's newer M1/M2 macbooks, the user needs to install via [anaconda](https://www.anaconda.com/download#macos)
using the following instructions:

```commandline
conda env create -n jabs -f environment_jabs.yml
conda activate jabs 
python app.py 
```

#### Enabling XGBoost Classifier

The XGBoost Classifier has a dependency on the OpenMP library. This does
not ship with MacOS. XGBoost should work "out of the box" on other platforms. 
On MacOS, you can install libomp with Homebrew (preferred) with the following 
command `brew install libomp`. You can also install libomp from source if you 
can't use Homebrew, but this is beyond the scope of this Readme.  

### Windows

Make sure that a compatible version of Python is installed (3.7, 3.8, or 3.9).

#### Windows Scripts

There are two convenience scripts included with JABS, `setup_windows.bat` and 
`jabs.bat`, that allow a user to set up the Python environment and launch 
JABS without using the command prompt.

The `setup_windows.bat` script will create a Python virtual 
environment in the JABS directory called `jabs.venv` and then install all 
the required packages from PyPi. This script can be executed by double-clicking 
on it in the Windows Explorer. This script only needs to be executed once.

The `jabs.bat` script will activate the jabs.venv virtual environment and 
launch the JABS application. This can be executed by double-clicking on it in 
the Windows Explorer.

#### Manual Configuration

You can also set up the Python virtual environment and execute JABS from the 
Windows Command Prompt (cmd.exe). 

To configure the Python virtual environment manually, Open a Command Prompt in 
the JABS directory and run the following commands:
```commandline
python -m venv jabs.venv
jabs.venv\Scripts\activate.bat
pip install -r requirements.txt
```

To launch JABS from the command prompt, open a command prompt in the JABS 
directory and run the following commands:
```commandline
jabs.venv\Scripts\activate.bat
python app.py
```
