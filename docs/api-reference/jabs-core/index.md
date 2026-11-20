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
- **Pose interfaces**: The `PoseLike` protocol (`jabs.core.pose_protocol`) that pose
  readers implement, and the skeleton constants (`jabs.core.pose_skeleton`). The
  former `PoseEstimation` abstract base is deprecated: `jabs.core.abstract.PoseEstimation`
  only carries the skeleton constants and warns when used.
- **Utility Functions**: Generic helpers for file hashing, logging configuration, and
  basic string/path manipulation.

## Package Structure

- `jabs.core.constants`: Global constants.
- `jabs.core.exceptions`: Shared exception classes.
- `jabs.core.pose_protocol`: The `PoseLike` protocol and pose metadata helpers.
- `jabs.core.pose_skeleton`: Keypoint indexes and connected-segment constants.
- `jabs.core.abstract`: Deprecated; holds only the `PoseEstimation` name for one release.
- `jabs.core.utils`: Generic utility functions.
- `jabs.core.enums`: Shared enumerations (e.g., `ClassifierType`).
