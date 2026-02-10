# JABS IO (`jabs-io`)

This package handles all serialization/deserialization logic.

## Overview

`jabs-io` decouples the JABS data representation from specific file formats or legacy
versions. It provides a unified way to interact with pose estimation data, behavioral
features, and annotations.

### API

The package provides a high level API that's intended to cover most use-cases.
```python
from jabs.io import save, load
from jabs.core.types.keypoints import FrameKeypoints

data_instance = load('frames.json', FrameKeypoints)
save(data_instance, 'frames.parquet')
```

## Development

Data models are defined as dataclasses in `jabs-core`: `jabs.core.types`. Some backends
will be able to implicitly handle most dataclasses, but if the dataclass is 
complicated, or if there is special handling required of the type for a backend, then a 
type specific adapter should be defined.

All adapters must inherit from `jabs.io.base.Adapter`.

For convenience, backend specific subclasses are provided that handle shared 
functionality for specific file backends.
```python
from jabs.io.base import (
    JSONAdapter,
    ParquetAdapter,
    # TODO: HDF5Adapter,
)
```

To have register you adapter for use, use the `register` adapter decorator.
```python
from jabs.core.enums import StorageFormat
from jabs.io.base import JSONAdapter
from jabs.io.registry import register_adapter

@register_adapter(StorageFormat.JSON)
class DataclassJSONAdapter(JSONAdapter):
    ...
```
