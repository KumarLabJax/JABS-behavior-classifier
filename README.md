# Behavior Labeler

## Requirements
Developed and tested on Python 3.7 and 3.8. See the `requirements.txt` for a
list of required Python packages. These packages are available from the Python 
Package Index (PyPI)

## Python Virtual Environment

### Creating the Virtual Environment

You will need to create the virtual environment before you can run the labeler 
for the first time. The following commands will create a new Python3 virtual 
environment, activate it, and install the required packages.

```commandline
python3 -m venv rotta.venv
source rotta.venv/bin/activate
pip install -r requirements.txt
```

### Activating 

The virtual environment must be activated before you can run the labeling 
interface. To activate, run the following command:

```commandline
source rotta.venv/bin/activate
```

### Deactivating

The virtual environment can be deactivated if you no longer need it:

```commandline
deactivate
```

### Enabling XGBoost Classifier

Using the XGBoost Classifier has a dependency on the OpenMP library. If you're 
running on Linux, this is likely included with gcc and XGBoost should work. If
you're using another platform you might have to install libomp yourself. On 
Mac OS, you can install libomp with Homebrew, or alternatively use the
"intel-openmp" package available on PyPi.

#### using intel-openmp packge from PyPi on Mac OS

The following instructions assume that you've crated a virtual environment
called `rotta.venv` and it has been activated.

First, install the package from PyPi:

`pip install intel-openmp`

The `intel-openmp` package  may not work out of the box with XGBoost. 

XGBoost tries to load a `libomp` dynamic library, however the `intel-openmp` 
package installs as `libomp5`. To remedy this, setup a symlink with the
following command (assuming you are in the directory that contains rotta.venv):

`ln -s libiomp5.dylib rotta.venv/lib/libomp.dylib`

You need to tell the dynamic linker how to find the library. On Mac, this can be 
done through the `DYLD_LIBRARY_PATH` environment variable. Use the following 
command to launch Rotta:

`DYLD_LIBRARY_PATH=rotta.venv/lib python app.py`
 