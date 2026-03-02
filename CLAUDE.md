# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

JABS (JAX Animal Behavior System) is a Python platform for mouse behavioral phenotyping.
It provides an interactive PySide6 GUI for behavior annotation and classifier training,
plus CLI tools for batch processing. Input data is video + HDF5 pose estimation files;
the system extracts behavioral features, trains classifiers (XGBoost, RandomForest,
CatBoost), and generates predictions. Supports Python 3.10-3.14.

For comprehensive developer documentation, see `docs/development/development.md`
(architecture deep-dives, detailed feature extraction guide, GUI architecture, release
process) and `docs/development/contributing.md` (contribution guidelines, copyright
assignment, PR process).

## Build & Development Commands

```bash
# Setup
uv sync                                    # Install all deps from lockfile (creates .venv)
uv run pre-commit install                  # Install git hooks (ruff check + ruff format)

# Testing
uv run pytest                              # Run root tests
uv run pytest tests/specific_dir/          # Run specific test directory
uv run pytest tests/test_file.py::test_name  # Run single test
uv run pytest --cov=src/jabs --cov-report=term-missing  # Run with coverage
# Package tests (CI runs these separately):
uv run pytest packages/jabs-core/tests
uv run pytest packages/jabs-io/tests
uv run pytest packages/jabs-behavior/tests
uv run pytest packages/jabs-vision/tests

# Linting & Formatting
uv run ruff check .                        # Lint
uv run ruff check --fix .                  # Auto-fix lint issues
uv run ruff format .                       # Format code
uv run ruff check --fix . && uv run ruff format .  # Fix all (use after failed pre-commit)

# Dependency Management (always use uv, never pip)
uv add <package>                           # Add a dependency to root
uv add --package jabs-core <pkg>           # Add a dependency to a sub-package
uv remove <package>                        # Remove a dependency
uv lock                                    # Regenerate lockfile after manual pyproject.toml edits

# Run the app
uv run jabs                                # Launch GUI
uv run jabs-cli --help                     # Unified CLI utilities

# Standalone CLI scripts
uv run jabs-init /path/to/project          # Initialize a JABS project
uv run jabs-features /path/to/project      # Generate features (batch, no GUI)
uv run jabs-classify --classifier model.pkl /path/to/pose.h5  # Batch classification
uv run jabs-stats /path/to/project         # View project statistics

# jabs-cli subcommands (preferred for new utilities)
uv run jabs-cli export-training            # Export training data from a project
uv run jabs-cli rename-behavior            # Rename a behavior across a project
uv run jabs-cli prune                      # Remove videos from a project
```

## Architecture

### Monorepo Structure

The project is a **uv workspace** transitioning from monolithic (`src/jabs/`) to a
monorepo with reusable library packages. The root `pyproject.toml` defines the
workspace; each package builds as a `jabs.*` namespace package.

**Dependency graph:**

```
jabs-behavior-classifier (root: src/jabs/)
├── jabs-core     (packages/jabs-core)     — lightweight shared types, enums, abstract base classes
├── jabs-io       (packages/jabs-io)       — file I/O (HDF5, JSON, Parquet, NWB) — depends on jabs-core
├── jabs-behavior (packages/jabs-behavior) — event processing & postprocessing filters — independent
└── jabs-vision   (packages/jabs-vision)   — DL inference (PyTorch, HRNet, timm) — depends on jabs-core, jabs-io
```

**New reusable logic should go in the appropriate sub-package, not in `src/jabs/`.** The
root `src/jabs/` contains the GUI application, project management, feature extraction,
classifiers, and CLI scripts.

### Key Subsystems in `src/jabs/`

- **`feature_extraction/`** — The heart of JABS. Hierarchical: `IdentityFeatures` →
  `FeatureGroup` → `Feature`. Feature instances are transient (created for computation,
  then discarded); only numpy arrays persist. Per-frame features cached in HDF5 with
  version/hash validation.
- **`project/`** — Central `Project` class coordinates videos, poses, features,
  annotations, predictions. A JABS project is a directory with this on-disk layout:
  ```
  project_directory/
  ├── video1.mp4
  ├── video1_pose_est_v6.h5
  └── jabs/                    # JABS metadata directory
      ├── project.json         # Project configuration
      ├── annotations/         # User annotations
      ├── features/            # Cached extracted features
      ├── predictions/         # Classifier predictions
      ├── classifiers/         # Trained classifier models
      ├── archive/             # Archived data
      ├── session/             # Session logging
      └── cache/               # Performance cache
  ```
- **`classifier/`** — Wraps scikit-learn/XGBoost/CatBoost models with cross-validation
  and persistence.
- **`ui/`** — PySide6 GUI using Qt signals/slots. Use `MessageDialog` (not
  `QMessageBox`) for error/warning/info dialogs.
- **`scripts/`** — CLI entrypoints. New CLI utilities should be added as Click
  subcommands of `jabs-cli` in `scripts/cli/`. Only create standalone scripts for
  primary, frequently-used commands (like `jabs` for the GUI). See
  `docs/development/development.md` for details on adding `jabs-cli` commands.
- **`pose_estimation/`** — Version-specific pose file readers (v2-v8).
- **`behavior_search/`** — Behavior search and query functionality.
- **`video_reader/`** — Video file reading and processing.

### Feature Extraction Details

Feature groups are registered in `src/jabs/feature_extraction/features.py`:

- `_FEATURE_MODULES`: core groups (BaseFeatureGroup, SocialFeatureGroup,
  SegmentationFeatureGroup)
- `_EXTENDED_FEATURE_MODULES`: optional groups (LandmarkFeatureGroup)

**Adding a new feature:** Create a `Feature` subclass with `_name`, `_min_pose`, and
`per_frame(identity)` method. Register it in the group's `_features` dict. The feature
auto-integrates with caching, window operations, and classifier training. For angular
features, set `_use_circular = True` (see `base_features/angles.py`). Features can also
declare `_static_objects` for landmark dependencies.

**Adding a new feature group:** Create a new directory under `feature_extraction/`,
create a `FeatureGroup` subclass with `_name`, `_features`, and `_init_feature_mods()`,
then register it in `_FEATURE_MODULES` or `_EXTENDED_FEATURE_MODULES` in `features.py`.

**`FEATURE_VERSION`** in `features.py` must be bumped when adding/modifying features,
changing window operations, or altering HDF5 storage format. This triggers cache
invalidation and recomputation. See `docs/development/development.md` for the full
cache invalidation logic (version mismatch, pose hash, distance scale).

## Code Style

- **Ruff** for linting and formatting (config in `ruff.toml`)
- Line length: 99, target: Python 3.10
- Google-style docstrings (pydocstyle convention)
- Rules: pycodestyle, pyflakes, pydocstyle, isort, pyupgrade, flake8-bugbear,
  flake8-comprehensions, flake8-simplify, Ruff-specific
- `__init__.py` files: unused imports (F401) are allowed
- **Naming**: `PascalCase` classes, `snake_case` functions/methods/files,
  `UPPER_SNAKE_CASE` constants, `_` prefix for private members

## Coding Standards

### Type Hints

Use comprehensive type hints on **all** new code. This is non-negotiable.

- Annotate all function parameters, return types, and class/instance attributes.
- Use modern syntax: `list[str]`, `dict[str, int]`, `str | None` (not `List`, `Dict`,
  `Optional`).
- Use `npt.NDArray[np.float64]` (not bare `np.ndarray`) for numpy arrays.
- Use `pathlib.Path` over `str` for filesystem paths in signatures.
- Import from `collections.abc` for abstract types (`Sequence`, `Mapping`, `Iterator`,
  `Callable`).
- Define `TypeAlias` for complex domain types (e.g.,
  `FeatureMap: TypeAlias = dict[str, npt.NDArray[np.float64]]`).
- Use `TYPE_CHECKING` guards for import-only annotations to avoid circular imports:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jabs.core.types import PoseArray
    from jabs.project import Project
```

### Docstrings

Every module, class, and public function/method requires a Google-style docstring.
Private helpers (`_`-prefixed) need at least a one-liner.

```python
"""Module for computing pairwise distance features between identities."""

class PairwiseDistance:
    """Computes pairwise Euclidean distance between identity centroids.

    Attributes:
        arena_diameter: Diameter of the arena in pixels, used for normalization.
    """

    def compute(
        self,
        poses_a: npt.NDArray[np.float64],
        poses_b: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute frame-by-frame pairwise distances.

        Args:
            poses_a: Pose array for identity A, shape (n_frames, n_keypoints, 2).
            poses_b: Pose array for identity B, shape (n_frames, n_keypoints, 2).

        Returns:
            Array of normalized distances with shape (n_frames,).

        Raises:
            ValueError: If input arrays have mismatched frame counts.
        """
        ...
```

Include `Args`, `Returns`, and `Raises` sections as applicable; omit empty sections.

### Logging

Use Python's standard `logging` module. **Never use `print()` for operational output.**

- Create `logger = logging.getLogger(__name__)` in every module.
- Use lazy formatting (`logger.info("x=%s", x)`) — never f-strings in log calls.
- Include `exc_info=True` on error-level messages when an exception is in scope.
- Log at module boundaries: entry/exit of significant public methods, I/O operations.
- Levels: **DEBUG** = internal state, cache hits; **INFO** = operations
  starting/completing; **WARNING** = deprecated usage, fallbacks; **ERROR** = caught
  failures; **EXCEPTION** = unexpected errors (auto-includes traceback).

### Module Design

Every module should have a **clear, single responsibility**. If a module docstring needs
"and" to describe its purpose, consider splitting it.

- Use `__init__.py` to define the public API; re-export key symbols.
- Avoid circular imports — shared types and protocols belong in `jabs-core`.
- Prefer composition over inheritance; use Protocols and ABCs from `jabs-core`.

### General Principles

1. **Keep functions small and focused** — each function does one thing well.
2. **Avoid side effects in core logic** — push I/O and state mutation to the boundaries.
3. **Test extensively** with clear assertions (see Testing below).
4. **Log important operations and errors.**
5. **Document public interfaces thoroughly** — if it's importable, it needs a docstring.

## Dependency Management

**Always use `uv`** — never `pip install`, `conda`, or manual virtualenv creation.

- All versions pinned via `uv.lock`. Never edit the lockfile manually.
- After adding/removing dependencies, verify with `uv lock`.
- **Always use `uv run`** to execute commands — never `source .venv/bin/activate`.

## Testing

### Framework & Tools

- **pytest** (plain functions, no unittest-style classes), **pytest-cov**, **pytest-mock**
- Run coverage: `uv run pytest --cov=src/jabs --cov-report=term-missing`

### Test Organization

Mirror the source structure. Every source module gets a corresponding test module:

```
src/jabs/feature_extraction/social.py  →  tests/feature_extraction/test_social.py
packages/jabs-io/src/jabs/io/hdf5.py  →  packages/jabs-io/tests/io/test_hdf5.py
```

Place `conftest.py` at each test directory level for shared fixtures.

### Writing Tests

```python
# Test pure functions directly
def test_normalize_distance_basic() -> None:
    result = normalize_distance(distance=50.0, arena_diameter=100.0)
    assert result == pytest.approx(0.5)

# Mock external dependencies — never hit filesystem, network, or GPU in unit tests
def test_classifier_predict(mocker: MockerFixture) -> None:
    mock_model = mocker.Mock()
    mock_model.predict.return_value = np.array([0, 1, 0])
    classifier = BehaviorClassifier(model=mock_model)
    predictions = classifier.predict(features)
    mock_model.predict.assert_called_once()
    assert len(predictions) == 3

# Use fixtures extensively for shared setup (in conftest.py)
@pytest.fixture
def sample_pose_array() -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((100, 12, 2))

# Parametrize for multiple scenarios
@pytest.mark.parametrize(
    ("distance", "diameter", "expected"),
    [(50.0, 100.0, 0.5), (0.0, 100.0, 0.0), (100.0, 100.0, 1.0)],
    ids=["half", "zero", "equal"],
)
def test_normalize_distance(distance: float, diameter: float, expected: float) -> None:
    assert normalize_distance(distance, diameter) == pytest.approx(expected)

# Test edge cases and error conditions
def test_normalize_distance_zero_diameter() -> None:
    with pytest.raises(ValueError, match="diameter must be positive"):
        normalize_distance(distance=50.0, arena_diameter=0.0)
```

### Coverage

- **Aim for >80%** but prioritize meaningful tests over chasing numbers.
- **Focus on core business logic**: feature extraction, classifiers, pose processing,
  I/O adapters. Lower coverage is acceptable for GUI and CLI glue code.
- Every new module must include tests. PRs adding untested core code won't pass review.
- Use `# pragma: no cover` sparingly (platform-specific fallbacks,
  `if __name__ == "__main__"` blocks).


### Creating a Release

1. Update `version` in the relevant `pyproject.toml`
2. Run `uv lock` to regenerate the lockfile
3. Commit both `pyproject.toml` and `uv.lock`
4. Merge to `main` via PR — CI automatically builds, publishes to PyPI, and creates a
   GitHub release
- Pre-release versions (e.g., `1.0.0a1`, `2.1.0rc1`) are auto-detected and marked
  accordingly