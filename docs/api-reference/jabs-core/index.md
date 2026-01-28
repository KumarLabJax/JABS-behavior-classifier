# JABS Core (`jabs-core`)

The infrastructure and shared utility layer for the JABS.

## Overview

`jabs-core` provides low-level, domain-agnostic utilities used across all JABS packages.
It is designed to be lightweight and free of heavy scientific dependencies (like
`scikit-learn` or `pandas`), making it safe to import at any level of the hierarchy.

## Responsibilities

- **Shared Constants**: Global constants used for file compression and configuration.
- **Exceptions**: Centralized exception hierarchy (`JabsError`, `PoseHashException`,
  etc.).
- **Infrastructure**: Base classes for registries and plugin discovery systems.
- **Abstract Bases**: High-level interface definitions (e.g., the `PoseEstimation`
  abstract base).
- **Utility Functions**: Generic helpers for file hashing, logging configuration, and
  basic string/path manipulation.

## Package Structure

- `jabs.core.constants`: Global constants.
- `jabs.core.exceptions`: Shared exception classes.
- `jabs.core.abstract`: Abstract base classes for the system.
- `jabs.core.utils`: Generic utility functions.
- `jabs.core.enums`: Shared enumerations (e.g., `ClassifierType`).
