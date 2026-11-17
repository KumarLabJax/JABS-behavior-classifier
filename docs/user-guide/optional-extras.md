# Optional Extras

Some JABS features depend on packages that are not installed by default, either
because only some users need them or because they are not available on every
platform. These features are grouped into *extras* that you can choose to install.

## Available Extras

| Extra               | Adds                                                                                     | Installs                                                      |
|---------------------|------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| `extra-classifiers` | The CatBoost and LightGBM classifier types. See [Choosing a Classifier](classifier-types.md). | `catboost`, `lightgbm`                                        |
| `nwb`               | Exporting pose data to NWB with `jabs-cli convert-to-nwb`. See [NWB Export](nwb-export.md). | `pynwb`, `ndx-pose`, `ndx-jabs`, `ndx-multisubjects`          |
| `mlflow`            | Logging `jabs-cli cross-validation` runs to an MLflow tracking server. See [Command Line Tools](cli-tools.md#mlflow-logging). | `mlflow`                                                      |
| `yaml`              | YAML config files for `jabs-cli postprocess` (JSON works without it). See [Command Line Tools](cli-tools.md#jabs-cli-postprocess). | `pyyaml`                                                      |

Without an extra, JABS still runs: the features that need it are unavailable. For
example, classifier types whose packages are not installed are left out of the
classifier choices, and `jabs-cli convert-to-nwb` stops with an error saying the NWB
packages are not installed.

## Installing Extras

Extras go in square brackets after the package name. To install more than one,
separate them with commas and no spaces:

```
jabs-behavior-classifier[extra-classifiers]
jabs-behavior-classifier[extra-classifiers,nwb]
```

Put the package name in quotes, as in the commands below. Some shells, including
zsh (the default shell on macOS), otherwise treat the square brackets as a file
pattern and fail with an error such as `no matches found`.

Use the commands for the method you used to install JABS.

### uv

If you installed JABS with `uv tool install`, run it again with the extras you want:

```bash
uv tool install "jabs-behavior-classifier[extra-classifiers,nwb]"
```

This rebuilds the JABS environment with exactly the extras you list, removing any
that you leave out. **List every extra you want each time**, including ones you
installed before. For example, if you installed the `nwb` extra earlier and now want
the extra classifiers too, install `[extra-classifiers,nwb]`, not just
`[extra-classifiers]`.

### uvx

If you run JABS with `uvx` without installing it, add the extras to the `--from`
argument each time you run it:

```bash
uvx --from "jabs-behavior-classifier[extra-classifiers]" jabs
```

### pipx

If you installed JABS with `pipx`, reinstall it with the extras you want. `--force` is
needed because JABS is already installed:

```bash
pipx install --force "jabs-behavior-classifier[extra-classifiers,nwb]"
```

As with uv, **list every extra you want each time**: pipx remembers only the most
recent list when it later upgrades or reinstalls JABS.

### pip

If you installed JABS with `pip`, activate the virtual environment you installed it
into, then install the extras:

```bash
pip install "jabs-behavior-classifier[extra-classifiers]"
```

pip adds the extra's packages to your existing installation, so you only need to
list the extras you are adding. Extras you installed before are kept.

If you installed JABS from a local clone of the repository, install the extras from
the clone's directory instead:

```bash
pip install ".[extra-classifiers]"
```

## Checking an Extra Is Installed

In the JABS GUI, open **About JABS** from the JABS application menu. The *Optional
Extras* box lists each extra as installed, not installed, or partially installed (with
the missing packages named). It checks the installed packages each time it opens, but
JABS only picks up a newly installed extra when it starts, so restart JABS after
installing one.

After installing `extra-classifiers`, restart JABS: CatBoost and LightGBM should
appear in the classifier choices in the GUI, and as `--catboost` and `--lightgbm` in
`jabs-classify classify --help`.

On macOS, LightGBM also needs the OpenMP library (libomp), as XGBoost does. If
LightGBM does not appear after installing the extra, install libomp with
`brew install libomp`.
