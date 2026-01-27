---
hide:
  - navigation
---

This section describes how to install JABS as an end user. 

!!! note
    The first time you run JABS, it may take a few minutes to initialize. Startup
    time for subsequent runs will be substantially faster.

!!! warning ""
    Developers should see the [JABS Development](../development/development.md) page 
    for instructions on setting up a development environment.

### Install with pipx or uv 

!!! tip "Recommended"

The easiest way to install JABS is using [pipx](https://pipx.pypa.io/)
or [uv](https://docs.astral.sh/uv/), which install Python applications in isolated
environments:

=== "uv"

    ```bash
    uv tool install jabs-behavior-classifier
    ```

=== "pipx"

    ```bash
    pipx install jabs-behavior-classifier
    ```

Both commands automatically create a virtual environment and make the JABS commands
available system-wide. After installation, you can run JABS from any terminal.

```bash
# launch the JABS GUI
jabs

# view help for jabs-init command
jabs-init --help
```

### Run with uvx

!!! tip "No Installation Required"

Alternatively, you can use `uvx` to run JABS without permanently installing it:

```bash
uvx --from jabs-behavior-classifier jabs
```

This runs JABS in an isolated environment without permanently installing it. You can
also use `uvx` to run other JABS commands:

```bash
uvx --from jabs-behavior-classifier jabs-init
uvx --from jabs-behavior-classifier jabs-classify
```

### Create a Virtual Environment

If not using `pipx` or `uvx`, we recommend installing JABS within a dedicated Python
virtual environment to avoid conflicts with other packages. You can create and activate
a virtual environment using the following commands:

```bash
python -m venv jabs.venv

# Linux and macOS
source jabs.venv/bin/activate

# Windows (cmd)
jabs.venv\Scripts\activate.bat
```

!!! note ""
    JABS supports Python 3.10 through 3.14. Make sure to use a compatible Python version
    when creating the virtual environment.

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

The XGBoost Classifier has a dependency on the OpenMP library. This does not ship with macOS. XGBoost should work "out of the box" on other platforms. On macOS, you can install libomp with Homebrew (preferred) with the following command `brew install libomp`. You can also install libomp from source if you can't use Homebrew, but this is beyond the scope of this Readme.

Because libomp is dynamically loaded by XGBoost it can be installed before or after installing jabs-behavior-classifier.

---

## Running JABS

After installation, the following commands are available in your environment:

- `jabs` — launch the JABS GUI  
- `jabs-init` — initialize a new JABS project directory or recompute features in an existing project 
- `jabs-classify` — run a trained classifier  
- `jabs-export-training` — export training data from an existing JABS project
- `jabs-cli` - collection of smaller command line utilities

You can view usage information for any command with:

```bash
<jabs-command> --help
```

## Sample Data

We provide sample data for testing and demonstration purposes. You can download the sample data from
https://doi.org/10.5281/zenodo.16697331

If everything runs smoothly, you should see a JABS startup window like the following:

![JABS Startup](../tutorials/images/JABS_startup.png)

## Preparing the JABS Project

Once the JABS environment is activated, prepare your project folder. The folder should
contain the videos for labeling and the corresponding pose file for each video.
Once prepared, you may either proceed to open the JABS GUI or initialize the project
folder prior to working using jabs-init.

```console
jabs-init <project_dir>
```

This will generate the JABS features for the project for the default window size of 5.
The argument ‘-w’ can be used to set the initial window size for feature generation.

### Starting up

You can open the JABS GUI with the command:

```console
jabs
```