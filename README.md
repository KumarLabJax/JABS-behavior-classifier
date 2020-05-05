# Behavior Labeler

## Requirements
Developed at tested on Python 3.7 and 3.8. See the `requirements.txt` for a list
of required Python packages. These packages are available from the Python 
Package Index (PyPI)

## Python Virtual Environment

### Creating the Virtual Environment

You will need to create the virtual environment before you can run the labeler 
for the first time. The following commands will create a new Python3 virtual 
environment, activate it, and install the required packages.

```commandline
python3 -m venv venv.labeler
source venv.labeler/bin/activate
pip install -r requirements.txt
```


### Activating 

The virtual enviornment must be activated before you can run the classifier. 
To activate, run the following command:

```commandline
source venv.labeler/bin/activate
```

### Deactivating

The virtual environment can be deactivated if you no longer need it:

```commandline
deactivate
```