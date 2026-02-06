# JABS IO (`jabs-io`)

This package defines the canonical data models and handles all 
serialization/deserialization logic.

## Overview

`jabs-io` decouples the JABS data representation from specific file formats or legacy
versions. It provides a unified way to interact with pose estimation data, behavioral
features, and annotations.

## Development

Data models are defined as dataclasses in `jabs.io.types`. Some backends will be able to
implicitly handle most dataclasses, but if the dataclass is complicated, or if there is
special handling required of the type for a backend, then a type specific adapter 
should be defined in `jabs.io.backends.$BACKEND.adaptets`.
