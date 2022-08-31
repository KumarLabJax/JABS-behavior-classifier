Usage
=====

.. _installation:

Installation
------------

Creating the Virtual Environment
#####

You will need to create the virtual environment before you can run the labeler 
for the first time. The following commands will create a new Python3 virtual 
environment, activate it, and install the required packages. Note, your python 
executable may be named ``python` or `python3`` depending on your installation.

.. code-block:: console

   python -m venv jabs.venv
   source jabs.venv/bin/activate
   (.venv) $ pip install -r requirements.txt

Activating
#####

The virtual environment must be activated before you can run the labeling 
interface. To activate, run the following command:

.. code-block:: console
   
   source jabs.venv/bin/activate

Deactivating
#####

The virtual environment can be deactivated if you no longer need it:

.. code-block:: console
   
   deactivate

Enabling XGBoost Classifier
#####

The XGBoost Classifier has a dependency on the OpenMP library. This does
not ship with MacOS. XGBoost should work "out of the box" on other platforms. 
On MacOS, you can install libomp with Homebrew (preferred) with the following 
command ``brew install libomp``.


  
Launching JABS GUI
------------------

To launch JABS from the command prompt, open a command prompt in the JABS 
directory and run the following commands:

.. code-block:: console

   jabs.venv\Scripts\activate.bat
   python app.py



