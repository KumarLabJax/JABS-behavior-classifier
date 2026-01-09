# JABS Development Guide

Welcome to the JABS (JAX Animal Behavior System) development guide. This document provides detailed technical information about the software architecture, development setup, code organization, and implementation details.

**📝 Looking to contribute?** See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines, copyright information, and how to submit pull requests.

**👤 End user?** See the [User Guide](user-guide.md) for instructions on using JABS.

## Quick Start (5 Minutes)

**New to JABS development? Start here:**

```bash
# 1. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# OR: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Clone the repository
git clone https://github.com/KumarLabJax/JABS-behavior-classifier.git
cd JABS-behavior-classifier

# 3. Set up development environment (creates venv, installs dependencies, sets up pre-commit)
uv sync
source .venv/bin/activate  # macOS/Linux (or .venv\Scripts\activate on Windows)
pre-commit install

# 4. Verify installation
jabs --version            # Should print JABS version number
jabs                      # should launch the JABS GUI

# 5. Run tests to ensure everything works
pytest
```

**Note:** The first time you run JABS, it may take a few moments to initialize the GUI.

✅ **If all tests pass, you're ready to develop!**

💡 **Recommended Python version**: 3.13 or 3.14 (though 3.10-3.14 are all supported)

📖 For detailed explanations of each step, continue reading the full guide below.

---

## Table of Contents

1. [Quick Start](#quick-start-5-minutes)
2. [Common Development Tasks](#common-development-tasks) (Quick Reference)
3. [Development Environment Setup](#development-environment-setup)
4. [Project Structure](#project-structure)
5. [Software Architecture](#software-architecture)
6. [Development Workflow](#development-workflow)
7. [Testing](#testing)
8. [Code Style and Standards](#code-style-and-standards)
9. [Building and Distribution](#building-and-distribution)
10. [CI/CD and Release Management](#cicd-and-release-management)
11. [Tips for Developers](#tips-for-developers)
12. [Getting Help](#getting-help)

**Note:** For contribution guidelines, copyright information, and how to submit pull requests, see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## Common Development Tasks

**Quick reference guide for frequent operations:**

### "How do I...?"

**Add a new behavioral feature**
→ See [Adding a New Individual Feature](#adding-a-new-individual-feature)

**Fix a bug or make changes**
→ See [Development Workflow](#development-workflow)

**Run tests**
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest tests/specific/    # Run specific tests
```

**Fix code style issues**
```bash
ruff check --fix .        # Auto-fix linting issues
ruff format .             # Format code
# OR run both:
ruff check --fix . && ruff format .
```

**My commit was rejected by pre-commit**
1. Run `ruff check --fix . && ruff format .` to fix style issues
2. Run `git add -u` to stage the fixes
3. Run `pytest` to ensure tests still pass
4. Try committing again

**Create a pull request**
1. Create branch: `git checkout -b feature/my-feature`
2. Make changes and commit
3. Push: `git push origin feature/my-feature`
4. Open PR on GitHub to `main` branch

**Bump FEATURE_VERSION after adding/modifying features**
- Edit `FEATURE_VERSION` in `src/jabs/feature_extraction/features.py`
- See [Feature Version Management](#feature-version-management-and-cache-invalidation) for details

**Create a new release**
→ See [Creating a New Release](#creating-a-new-release)

### Common Errors & Solutions

**Error: `uv sync` fails with "lock file out of date"**
```bash
uv lock          # Regenerate lock file
uv sync          # Try again
```

**Error: `ImportError: No module named 'jabs'`**
```bash
uv pip install -e ".[dev]"   # Reinstall in editable mode
```

**Error: Pre-commit hooks fail**
```bash
ruff check --fix .     # Fix issues manually
ruff format .
git add -u             # Stage fixes
git commit             # Try again
```

**Error: Tests fail with "cannot find pose file"**
- Ensure you're running tests from the project root directory
- Check that test fixtures in `tests/` are intact

---

## Development Environment Setup

### Prerequisites

- Python 3.10 or later (currently supports 3.10-3.14)
- [uv](https://docs.astral.sh/uv/) - Modern Python package and project manager (recommended)
- Git

### Installing uv

Install uv using the official installer:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via pip:

```bash
pip install uv
```

### Setting Up the Development Environment

1. **Clone the repository:**

```bash
git clone https://github.com/KumarLabJax/JABS-behavior-classifier.git
cd JABS-behavior-classifier
```

2. **Create and activate a virtual environment with uv:**

```bash
# Create a virtual environment
uv venv

# Activate the environment
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

3. **Install JABS in development mode with all dependencies:**

```bash
# Install using lockfile (recommended - ensures exact versions)
uv sync

# Alternative: Install from pyproject.toml (gets latest compatible versions)
uv pip install -e ".[dev]"
```

`uv sync` is recommended because:
- Uses `uv.lock` to install exact dependency versions
- Ensures all developers have identical environments
- Faster than resolving dependencies from scratch
- Automatically installs the project in editable mode

This installs:
- Core dependencies (required for running JABS)
- Test dependencies (pytest)
- Lint dependencies (ruff)
- Development utilities (matplotlib for script to update keypoint legend for user guide)
- Pre-commit hooks package

4. **Install pre-commit hooks:**

```bash
pre-commit install
```

This activates the hooks to run automatically before each commit.

**What the pre-commit hook does:**

The pre-commit hook automatically runs two checks before allowing a commit:

1. **`ruff check`** - Lints your code for errors and style violations
2. **`ruff format --check`** - Verifies code is properly formatted

**Important:** The hook will **block your commit** if:
- Ruff finds any linting errors
- Code doesn't conform to Ruff's formatting standards

If your commit is blocked, you'll need to fix the issues:
```bash
ruff check --fix .    # Auto-fix linting issues
ruff format .         # Format code
git add -u            # Stage the fixes
git commit            # Try again
```

This ensures all committed code meets JABS quality standards.

### Alternative: Using Standard pip

If you prefer not to use uv:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install
```

## Project Structure

JABS follows a modern Python project structure with source code in the `src/` directory (not all subdirectories shown):

```
JABS-behavior-classifier/
├── src/jabs/               # Main source code
│   ├── behavior_search/    # Behavior search and query functionality
│   ├── classifier/         # Machine learning classifier implementations
│   ├── feature_extraction/ # Feature extraction from pose data
│   │   ├── base_features/      # Core feature classes
│   │   ├── landmark_features/  # Body landmark-based features
│   │   ├── segmentation_features/  # Spatial segmentation features
│   │   ├── social_features/    # Multi-animal social features
│   │   └── window_operations/  # Temporal window operations
│   ├── pose_estimation/    # Pose file handling and validation
│   ├── project/            # Project management and data organization
│   ├── resources/          # Static resources (icons, docs, etc.)
│   ├── schema/             # JSON schemas for validation
│   ├── scripts/            # Command-line interface scripts
│   ├── types/              # Type definitions and data models
│   ├── ui/                 # PySide6-based GUI components
│   ├── utils/              # Utility functions and helpers
│   ├── version/            # Version information
│   └── video_reader/       # Video file reading and processing
├── tests/                  # Test suite
├── dev/                    # Development utilities and scripts
├── build/                  # Build artifacts (generated)
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Lock file for exact dependency versions
├── ruff.toml               # Ruff linter/formatter configuration
└── .pre-commit-config.yaml # Pre-commit hooks configuration
```

### Key Directories Explained

- **`src/jabs/`**: All production code lives here. This is a "src layout" which keeps source separate from tests and build artifacts.

- **`feature_extraction/`**: The heart of JABS - extracts behavioral features from pose estimation data. Features are modular and extensible.

- **`project/`**: Manages JABS projects, including:
  - `Project`: Main project class
  - `ProjectPaths`: Directory structure management
  - `VideoManager`, `FeatureManager`, `PredictionManager`: Resource management
  - `TimelineAnnotations`, `TrackLabels`, `VideoLabels`: Annotation data structures

- **`ui/`**: PySide6-based GUI components.

- **`classifier/`**: Implements machine learning classifiers (currently supports scikit-learn RandomForest as well as XGBoost).

- **`scripts/`**: Entry points for command-line tools (installed as console scripts).

- **`dev/`**: Development utilities. These are not installed as part of the package, but might be useful for developers.

## Software Architecture

### Core Design Principles

1. **Separation of Concerns**: GUI, business logic, and data layers are separated.
2. **Modularity**: Features and classifiers are plugin-based and extensible.
3. **Data Validation**: JSON schemas validate project files and configurations.
4. **Type Safety**: Type hints are used throughout for better IDE support and error checking.

### Key Components

#### Project Management

The `Project` class is the central coordinator:

```python
from jabs.project import Project

# Create/open a project (initialization automatically loads videos, poses, annotations, etc.)
project = Project(project_path="/path/to/project")

# The project is ready to use immediately after instantiation
videos = project.videos
```

Projects are organized with this structure:
```
project_directory/
├── video1.mp4
├── video1_pose_est_v6.h5
├── video2.mp4
├── video2_pose_est_v6.h5
└── jabs/                    # JABS metadata directory
    ├── project.json         # Project configuration
    ├── annotations/         # User annotations
    ├── features/            # Extracted features
    ├── predictions/         # Classifier predictions
    ├── classifiers/         # Trained classifier models
    ├── archive/             # Archived data
    ├── session/             # Session logging
    └── cache/               # Performance cache
```

#### Feature Extraction

Features are modular and inherit from base classes:

- `Feature`: Base class for individual feature modules (e.g., angles, distances, velocities)
- `FeatureGroup`: Base class for organizing related features into groups

##### Feature System Architecture

The feature extraction system has a hierarchical structure:

```
IdentityFeatures (per identity/video)
    └── FeatureGroup instances (e.g., BaseFeatureGroup, SocialFeatureGroup)
        └── Feature instances (e.g., Angles, PointSpeeds, CentroidVelocityMag)
```

**How it works:**

1. **`IdentityFeatures`** (in `features.py`) is the top-level class that manages all features for a single identity in a video.
   - Instantiated with pose data, identity, and project directory
   - Automatically loads/creates all relevant `FeatureGroup` instances based on pose version
   - Handles caching of computed features to HDF5 files
   - Provides methods like `get_per_frame()` and `get_window_features()`

2. **`FeatureGroup`** subclasses organize related features:
   - `BaseFeatureGroup`: Core features (angles, distances, velocities) - always available
   - `SocialFeatureGroup`: Multi-animal social features (requires pose v3+)
   - `SegmentationFeatureGroup`: Spatial segmentation features (requires pose v6+)
   - `LandmarkFeatureGroup`: Static object/landmark features (extended features)

3. **`Feature`** subclasses compute actual feature values:
   - Each implements `per_frame(identity)` to compute per-frame values
   - Inherits `window()` method for temporal window operations
   - Can specify required pose version and dependencies

##### Feature Registration

Feature groups are registered in `src/jabs/feature_extraction/features.py`:

```python
# Core feature groups (always loaded if pose version supports them)
_FEATURE_MODULES = [BaseFeatureGroup, SocialFeatureGroup, SegmentationFeatureGroup]

# Extended feature groups (optional, loaded separately)
_EXTENDED_FEATURE_MODULES = [LandmarkFeatureGroup]
```

When `IdentityFeatures` is instantiated, it automatically creates instances of all applicable feature groups based on the pose file version.

##### Adding a New Individual Feature

To add a new feature to an existing group (e.g., a new base feature):

1. **Create the feature class** in the appropriate subdirectory (e.g., `base_features/`)

   Look at existing examples like `src/jabs/feature_extraction/base_features/centroid_velocity.py` for reference.
   
   Key requirements:
   - Set `_name` class attribute (unique identifier)
   - Set `_min_pose` (minimum pose version required)
   - Implement `per_frame(identity)` returning `dict[str, np.ndarray]`
   - Optionally set `_use_circular = True` for angular features

2. **Register the feature in its group** (e.g., `base_features/base_group.py`)

   Add your feature class to the `_features` dictionary:
   ```python
   _features: typing.ClassVar[dict[str, type[Feature]]] = {
       # ...existing features...
       MyNewFeature.name(): MyNewFeature,
   }
   ```

3. **The feature is now available!** It will automatically:
   - Be computed when features are generated
   - Have window operations applied (mean, std, etc.)
   - Be saved/loaded from cache
   - Be available for classifier training

##### Adding a New Feature Group

To create an entirely new feature group:

1. **Create the group directory** under `src/jabs/feature_extraction/`:

   ```bash
   mkdir src/jabs/feature_extraction/my_feature_group
   ```

2. **Create feature classes** in the new directory (see existing examples in `base_features/`, `social_features/`, etc.)

3. **Create the group class** following the pattern in `base_features/base_group.py`:
   - Inherit from `FeatureGroup`
   - Define `_name` and `_features` class attributes
   - Implement `_init_feature_mods(identity)` to instantiate feature classes

4. **Register the group in `features.py`**:
   - Add to `_FEATURE_MODULES` (core features, always loaded)
   - Or add to `_EXTENDED_FEATURE_MODULES` (optional features)

##### Special Feature Types

**Circular/Angular Features:**

For features that represent angles, set `_use_circular = True`. See `src/jabs/feature_extraction/base_features/angles.py` for a complete example.

Key points:
- Override `_circular_window_operations` if your angles are in a different range than the default `[-180, 180)`
- Provide sine/cosine transformations for non-circular window operations
- Example: The `Angles` class uses `[0, 360)` for joint angles

**Note:** The base `Feature` class provides defaults for the range `[-180, 180)`, which is appropriate for bearings and most JABS directional features. You only need to override if your angular feature uses a different range.

**Features with Dependencies:**

Features can specify minimum pose versions and required static objects via class attributes:
- `_min_pose`: Minimum pose version (e.g., `_min_pose = 6`)
- `_static_objects`: List of required static objects (e.g., `["water_bottle", "food_hopper"]`)

See `src/jabs/feature_extraction/landmark_features/` for examples.

##### Detailed Initialization and Execution Flow

Understanding when and how features are initialized is key to extending the feature extraction system.

**Key Concepts:**

1. **`IdentityFeatures.__init__()` creates `FeatureGroup` instances**
   - Each `FeatureGroup` stores *class references* to `Feature` classes (not instances)
   - See `base_features/base_group.py` for the `_features` dictionary structure
   - Per-frame features are **eagerly loaded or computed** during initialization

2. **Two paths for per-frame features:**
   - **Cache hit**: `__load_from_file()` reads numpy arrays from HDF5 - no Feature instances created
   - **Cache miss**: `__initialize_from_pose_estimation()` creates Feature instances, computes values, then discards instances

3. **Feature instances are transient:**
   - Created only when needed (during computation or window operations)
   - `FeatureGroup._init_feature_mods()` instantiates Feature classes from the `_features` dict
   - Immediately discarded after extracting values
   - Only computed numpy arrays persist in `self._per_frame`

4. **Window features are computed on-demand:**
   - By loading on demand, new window sizes can be added at any time
   - `get_window_features()` creates fresh Feature instances each time
   - Applies statistical operations (mean, std, etc.) to cached per-frame values
   - Can also be loaded from cache if available

**Data Structure:**

`self._per_frame` is a nested dict: `dict[str, dict[str, np.ndarray]]`
- Outer keys: feature group names 
- Inner keys: feature names 
- Values: numpy arrays of per-frame feature values

**Complete Flow Diagram:**

```
Project opened
    └── When training/classifying:
        └── IdentityFeatures.__init__(pose_est, identity=0)
            ├── Create FeatureGroup instances
            │   ├── BaseFeatureGroup(pose_est, pixel_scale)
            │   ├── SocialFeatureGroup(pose_est, pixel_scale)  [if pose v3+]
            │   └── SegmentationFeatureGroup(pose_est, pixel_scale)  [if pose v6+]
            │
            └── Load or compute per-frame features (EAGERLY, during __init__)
                ├── PATH 1: Cache hit → __load_from_file()
                │   ├── Read numpy arrays from HDF5
                │   ├── Populate self._per_frame with loaded arrays
                │   └── NO Feature instances created!
                │
                └── PATH 2: Cache miss → __initialize_from_pose_estimation()
                    └── For each FeatureGroup:
                        └── group.per_frame(identity)
                            ├── _init_feature_mods() → Create Feature instances
                            ├── Call feature.per_frame(identity) on each
                            │   └── Returns dict: {"feature_name": numpy_array}
                            └── Feature instances discarded after use
                    ├── self._per_frame now populated with computed arrays
                    └── Save to cache (if directory provided)
            
            → __init__ completes with self._per_frame populated (either way)
        
        └── Later: features.get_per_frame()
            └── Simply returns self._per_frame (already loaded/computed!)
        
        └── Later: features.get_window_features(window_size=5)
            ├── Try to load window features from cache
            └── If cache miss: __compute_window_features()
                └── For each FeatureGroup:
                    └── group.window(identity, window_size, self._per_frame)
                        ├── _init_feature_mods() → Create Feature instances (fresh)
                        ├── Call feature.window(...) on each
                        │   └── Applies statistical ops to per-frame values
                        └── Feature instances discarded after use
```


**Why this design?**

- **Eager per-frame loading**: Per-frame features are loaded/computed during `__init__` so they're immediately available
- **Transient Feature instances**: Feature instances are only created during computation (cache miss), then immediately discarded after values are extracted
- **Cache bypasses Feature creation**: When loading from cache, Feature instances are never created - just numpy arrays loaded from HDF5
- **Memory efficiency**: Feature instances don't persist; only the computed numpy arrays are kept in memory
- **Caching**: Expensive computations are cached to disk (HDF5 files) and loaded once
- **Flexibility**: Feature groups can be enabled/disabled based on pose version
- **Stateless features**: Each time Features are instantiated, they compute fresh values without carrying state

##### Feature Version Management and Cache Invalidation

The `FEATURE_VERSION` constant in `src/jabs/feature_extraction/features.py` is critical for managing cached feature files:

```python
# In features.py
FEATURE_VERSION = 16  # Current version

class IdentityFeatures:
    _version = FEATURE_VERSION
```

**What it does:**

When features are computed, they're cached to HDF5 files in the project's feature directory:
```
project/jabs/features/
    video1/
        0/  # identity 0
            features.h5  # Contains FEATURE_VERSION in metadata
        1/  # identity 1
            features.h5
```

Each cached `features.h5` file stores the `FEATURE_VERSION` that was used to compute it:

```python
# When saving features
with h5py.File(file_path, "w") as features_h5:
    features_h5.attrs["version"] = self._version  # Store FEATURE_VERSION
    features_h5.attrs["pose_hash"] = self._pose_hash
    # ... save feature data

# When loading features
with h5py.File(path, "r") as features_h5:
    if features_h5.attrs["version"] != FEATURE_VERSION:
        raise FeatureVersionException  # Forces recomputation!
```

**When to bump FEATURE_VERSION:**

You must increment `FEATURE_VERSION` whenever you make changes that would make cached features incompatible or incorrect. This includes:

1. **Adding a new base feature** to `BaseFeatureGroup`, `SocialFeatureGroup`, etc.
   - Example: Adding a new distance metric or angle calculation
   - Why: Old cache files won't have the new feature

2. **Modifying an existing feature's computation**
   - Example: Changing how angles are calculated, fixing a bug in distance computation
   - Why: Cached values would be wrong/outdated

3. **Changing feature names or structure**
   - Example: Renaming features, changing dict structure
   - Why: Loading code expects new structure

4. **Modifying window operations or signal processing**
   - Example: Adding new statistical operations, changing FFT bands
   - Why: Window features would be incomplete

5. **Changing how features are stored in HDF5**
   - Example: Modifying the file format, adding/removing metadata
   - Why: Loading/saving logic wouldn't match

**How to bump:**

Simply increment the version number in `features.py`:

```python
# Before
FEATURE_VERSION = 16

# After
FEATURE_VERSION = 17
```

**What happens after bumping:**

1. Next time `IdentityFeatures` is instantiated, it tries to load cached features
2. Sees version mismatch: `cached_version (16) != FEATURE_VERSION (17)`
3. Raises `FeatureVersionException`
4. Exception is caught, triggers recomputation:
   ```python
   try:
       self.__load_from_file()
   except (OSError, FeatureVersionException, ...):
       # Recompute features with new version
       self.__initialize_from_pose_estimation(pose_est)
   ```
5. New features are computed and cached with version 17
6. Users see one-time delay as all features are recomputed

**Additional cache invalidation mechanisms:**

Beyond version checking, the cache is also invalidated when:

- **Pose file changes**: `pose_hash` mismatch triggers recomputation
  ```python
  if features_h5.attrs["pose_hash"] != self._pose_hash:
      raise PoseHashException
  ```

- **Distance scale changes**: Switching between pixel/cm units
  ```python
  if self._distance_scale_factor != features_h5.attrs.get("distance_scale_factor", None):
      raise DistanceScaleException
  ```

**Best practices:**

1. **Increment conservatively**: When in doubt, bump the version
2. **Document in commits**: Note version bumps in commit messages
3. **Communicate to users**: Feature version bumps may cause long recomputation times
4. **Test thoroughly**: Ensure new features work correctly
5. **Consider backwards compatibility**: If possible, design features to be additive rather than breaking

**Summary:**

This versioning system ensures that users never accidentally use stale or incorrect cached features, maintaining data integrity throughout the analysis pipeline.

##### Testing New Features

Add tests in `tests/feature_tests/`. See existing feature tests for examples of how to:
- Test per-frame feature computation
- Verify feature output shapes and data types
- Test with sample pose estimation data
- Handle edge cases (missing keypoints, invalid frames, etc.)

#### Classifiers

Classifiers wrap scikit-learn or XGBoost models:

- Support for multiple classifier types (LogisticRegression, RandomForest, XGBoost, etc.)
- K-fold cross-validation
- Model persistence and versioning

#### GUI Architecture

- **Main Window** (`main_window.py`): Application shell
- **Central Widget** (`central_widget.py`): Main workspace
- **Player Widget**: Video playback controls
- **Timeline Widget**: Behavior timeline visualization
- **Video List Widget**: Project video management

The GUI uses Qt's signal/slot mechanism for communication between components.

### Data Flow

1. **Input**: Video files + Pose estimation HDF5 files
2. **Feature Extraction**: Compute behavioral features from pose keypoints
3. **Annotation**: User labels behaviors in the GUI
4. **Training**: Train classifiers on annotated data
5. **Prediction**: Apply trained classifiers to new data
6. **Visualization**: Display predictions in timeline

## Development Workflow

### Making Changes

1. **Create a feature branch:**

```bash
git checkout -b feature/my-new-feature
```

2. **Make your changes**

3. **Run tests:**

```bash
pytest
```

4. **Check code style:**

```bash
ruff check .
ruff format .
```

Pre-commit hooks will automatically run these checks, but you can run them manually.

5. **Commit your changes:**

```bash
git add .
git commit -m "Add new feature: description"
```

Pre-commit hooks will run automatically. Fix any issues before committing.

6. **Push and create a pull request**

### Running JABS During Development

After installing in development mode, you can run JABS commands:

```bash
# Launch GUI
jabs

# Initialize a JABS project
jabs-init /path/to/project

# Generate features for an HDF5 file without requring a JABS project (for batch processing)
jabs-features /path/to/project

# Run batch classification
jabs-classify --classifier my_classifier.pkl /path/to/pose_file.h5

# View statistics
jabs-stats /path/to/project

# Access unified CLI
jabs-cli --help
```

### About `jabs-cli`: The Unified Command-Line Interface

**What is `jabs-cli`?**

`jabs-cli` is a unified command-line interface that consolidates smaller JABS utilities under a single entry point. Instead of having many separate standalone scripts, `jabs-cli` uses Click's group/command pattern to organize related functionality.

**Current commands:**
- `jabs-cli export-training` - Export training data from a project
- `jabs-cli rename-behavior` - Rename a behavior across a project
- `jabs-cli prune` - Remove videos from a project based on criteria

View all available commands:
```bash
jabs-cli --help
jabs-cli <command> --help  # Get help for specific command
```

**Why use `jabs-cli` for new utilities?**

When adding new command-line functionality, prefer adding it to `jabs-cli` rather than creating a new standalone script because:

1. **Better organization**: Related commands are grouped under a single namespace
2. **Discoverability**: Users can find all utilities with `jabs-cli --help`
3. **Consistency**: Shared options (like `--verbose`) work across all commands
4. **Easier maintenance**: One codebase to maintain instead of many separate scripts
5. **Modern CLI patterns**: Uses Click for robust argument parsing and help text

**How to add a new command to `jabs-cli`:**

1. Open `src/jabs/scripts/cli.py`
2. Add a new command function decorated with `@cli.command()`:
   ```python
   @cli.command(name="my-command")
   @click.argument("project_dir", type=click.Path(...))
   @click.option("--my-option", help="Description")
   @click.pass_context
   def my_command(ctx, project_dir, my_option):
       """Brief description of what this command does."""
       # Implementation here
       pass
   ```
3. The command is now available as `jabs-cli my-command`

See existing commands in `cli.py` for complete examples.

**Future direction:**

Some standalone scripts may be deprecated in favor of expanding `jabs-cli`. Candidates (non-exhaustive) for consolidation include:
- `jabs-init` → `jabs-cli init-project`
- `jabs-project-merge` → `jabs-cli merge`

This consolidation would provide a more cohesive user experience and easier maintenance. However, backward compatibility will be maintained during any transition period.

**When to create a standalone script:**

Create a standalone script (like `jabs` or `jabs-classify`) only if:
- It's a primary, frequently-used command that warrants top-level access
- It needs to be extremely simple to invoke (e.g., `jabs` to launch GUI)
- It has significantly different requirements or dependencies

For most utilities, adding to `jabs-cli` is the better choice.

### Debugging

- Use PyCharm, VS Code, or your preferred Python debugger
- Set breakpoints in source code
- Run scripts with debugger attached
- GUI debugging: Use Qt Creator's tools for widget inspection

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_pose_file.py

# Run specific test
pytest tests/test_pose_file.py::test_function_name

# Run with coverage
pytest --cov=src/jabs --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory, mirroring the structure of `src/jabs/`
- Use pytest fixtures for common setup
- Use `tmp_path` fixture for temporary file operations
- See existing tests for patterns and examples

## Code Style and Standards

### Linting and Formatting: Ruff

JABS uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. Ruff is extremely fast and replaces multiple tools (flake8, isort, black, etc.).

#### Configuration

See `ruff.toml` for the complete configuration. Key settings:

- **Python version target**: 3.10
- **Line length**: 99 characters
- **Enabled rules**:
  - `E`: pycodestyle errors
  - `F`: pyflakes
  - `D`: pydocstyle (Google-style docstrings)
  - `I`: isort (import sorting)
  - `UP`: pyupgrade (modern Python syntax)
  - `B`: flake8-bugbear (common bugs)
  - `C4`: flake8-comprehensions
  - `SIM`: flake8-simplify
  - `RUF`: Ruff-specific rules

#### Running Ruff

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Check and format in one go
ruff check --fix . && ruff format .
```

#### Pre-commit Hooks

JABS uses pre-commit hooks to enforce code quality automatically. After running `pre-commit install`, the hooks will run before every commit.

**What happens during a commit:**

1. `ruff check` runs to detect linting errors
2. `ruff format --check` runs to verify formatting

**If the hook fails:**
- Your commit will be **blocked**
- You'll see error messages indicating what needs to be fixed
- Fix the issues and try committing again

**Common workflow when hook fails:**
```bash
# Attempt commit (fails due to hook)
git commit -m "My changes"

# Fix issues automatically
ruff check --fix .
ruff format .

# Stage the fixes
git add -u

# Commit again (should succeed now)
git commit -m "My changes"
```

**Bypassing the hook (not recommended):**
```bash
git commit --no-verify -m "Skip pre-commit checks"
```

Note: Bypassing hooks is discouraged as it may cause CI/CD failures.

### Documentation Standards

- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Type hints**: Always use type hints for function parameters and return values
- **Comments**: Explain "why", not "what" (code should be self-documenting)

See existing code in `src/jabs/` for examples of well-documented functions and classes.

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `VideoManager`, `FeatureExtractor`)
- **Functions/Methods**: `snake_case` (e.g., `load_project`, `compute_features`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_WINDOW_SIZE`, `DEFAULT_FPS`)
- **Private members**: Prefix with `_` (e.g., `_internal_state`, `_helper_function`)
- **File names**: `snake_case` (e.g., `video_manager.py`, `feature_extraction.py`)

### Import Organization

Ruff's isort integration automatically organizes imports into three groups:

1. Standard library imports
2. Third-party imports
3. Local application imports

See `.ruff.toml` for configuration details.


## Building and Distribution

### Version Management

Version is managed in `pyproject.toml`:

```toml
[project]
version = "0.38.1"
```

The version module (`src/jabs/version/`) reads this at runtime.

**Note:** For complete release instructions, see the [CI/CD and Release Management](#cicd-and-release-management) section below.

## CI/CD and Release Management

JABS uses GitHub Actions for continuous integration and automated releases to PyPI. The CI/CD pipeline is defined in `.github/workflows/` and automatically manages package building, testing, and publishing.

### Pull Request Checks

Pull requests to the `main` branch trigger automated checks to ensure code quality and functionality:

1. **Code Formatting and Linting**: Ensures code adheres to style guidelines using ruff
2. **Test Execution**: Runs the full test suite to verify functionality

### Automated Release Process

The release process is triggered automatically when the version number in `pyproject.toml` is changed on the `main` branch:

1. **Version Detection**: The workflow monitors changes to `pyproject.toml` and extracts the version number
2. **Pre-release Detection**: Versions containing letters (e.g., `1.0.0a1`, `2.1.0rc1`) are automatically marked as pre-releases
3. **Build Pipeline**: If version changed, the system runs:
   - Code formatting and linting checks
   - Test execution
   - Package building with `uv build`
4. **PyPI Publishing**: Successfully built packages are automatically published to PyPI
5. **GitHub Release**: A corresponding GitHub release is created with build artifacts

### Release Workflow Files

- **`.github/workflows/release.yml`**: Main release workflow that orchestrates the entire process
- **`.github/workflows/_format-lint-action.yml`**: Reusable workflow for code quality checks
- **`.github/workflows/_run-tests-action.yml`**: Reusable workflow for test execution
- **`.github/workflows/pull-request.yml`**: CI checks for pull requests

### Creating a New Release

To create a new release:

1. Update the version number in `pyproject.toml`:
   ```toml
   version = "X.Y.Z"  # for stable releases
   version = "X.Y.Za1" # for alpha pre-releases
   version = "X.Y.Zrc1" # for release candidates
   ```

2. Re-lock the uv lock file:
   ```bash
   uv lock
   ```

3. Commit and push the change:
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Bump version to X.Y.Z"
   ```

4. Merge your changes into the `main` branch via a pull request

5. The CI/CD pipeline will automatically:
   - Detect the version change
   - Run all quality checks and tests
   - Build and publish the package to PyPI
   - Create a GitHub release with generated release notes

### Environment Requirements

The release workflow requires:
- **PyPI API Token**: Stored as `PYPI_API_TOKEN` in GitHub repository secrets

## Tips for Developers

### Adding a New Feature Type

See the detailed [Feature Extraction](#feature-extraction) section in Software Architecture for a complete guide.

**Quick steps:**

1. Create a new class inheriting from `Feature` in `src/jabs/feature_extraction/`
2. Implement `per_frame(identity)` method to compute feature values
3. Register the feature in its `FeatureGroup` (e.g., `BaseFeatureGroup`)
4. Add tests in `tests/feature_tests/`
5. Bump `FEATURE_VERSION` in `src/jabs/feature_extraction/features.py`

**Example locations:**
- Individual features: `src/jabs/feature_extraction/base_features/angles.py`
- Feature groups: `src/jabs/feature_extraction/base_features/base_group.py`
- Registration: `src/jabs/feature_extraction/features.py`

### Adding a New Classifier

1. Add classifier type to `src/jabs/classifier/classifier.py`
2. Ensure scikit-learn compatibility
3. Add tests
4. Update GUI to expose new classifier in UI

### Extending the GUI

1. Create new widget classes in `src/jabs/ui/`
2. Use Qt Designer for complex layouts (optional, all existing widgets are hand-coded)
3. Connect signals and slots for communication
4. Follow existing widget patterns
5. Test with various data scenarios

### Adding New CLI Utilities

**Prefer adding to `jabs-cli` over creating standalone scripts.**

See the detailed [`jabs-cli` section](#about-jabs-cli-the-unified-command-line-interface) in Development Workflow for:
- Why `jabs-cli` is the preferred approach
- How to add a new command
- When standalone scripts are appropriate

**Quick example:**
1. Edit `src/jabs/scripts/cli.py`
2. Add a new `@cli.command()` decorated function
3. Use Click decorators for arguments and options
4. Command is immediately available as `jabs-cli <your-command>`

## Getting Help

### Resources

- **Main Repository**: https://github.com/KumarLabJax/JABS-behavior-classifier
- **Issue Tracker**: https://github.com/KumarLabJax/JABS-behavior-classifier/issues
- **User Guide**: [User Guide (Markdown)](docs/user-guide.md)
- **Contact**: jabs@jax.org

### How to Get Support

- Check existing issues on GitHub
- Review the user guide and documentation
- Contact the development team at jabs@jax.org
- Review code comments and docstrings for implementation details

---

**Last Updated**: December 30, 2024  
**JABS Version**: 0.38.1


