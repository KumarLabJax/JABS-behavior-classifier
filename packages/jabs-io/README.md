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

## Installation extras

`jabs-io` ships optional extras for format-specific backends:

| Extra     | Installs            | Enables                   |
|-----------|---------------------|---------------------------|
| `nwb`     | `pynwb`, `ndx-pose` | NWB pose file read/write  |
| `parquet` | `pyarrow`           | Parquet feature cache I/O |

`jabs-io` depends on `jabs-core`, which requires `h5py` directly, so HDF5 support is
always available with no extra needed. `pyarrow` is also a direct dependency of
`jabs-behavior-classifier`, so the `parquet` backend is available automatically when
`jabs-io` is installed as part of the full JABS application. The `nwb` extra must
always be installed explicitly:

```bash
pip install "jabs-behavior-classifier[nwb]"
```

When using `jabs-io` as a standalone library:

```bash
pip install "jabs-io[nwb,parquet]"
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

To register your adapter for use, use the `register` adapter decorator.
```python
from jabs.core.enums import StorageFormat
from jabs.io.base import JSONAdapter
from jabs.io.registry import register_adapter

@register_adapter(StorageFormat.JSON)
class DataclassJSONAdapter(JSONAdapter):
    ...
```

## Future Work

- **NWB: segmentation data roundtrip** — `PoseData.segmentation_data` is not currently stored by the NWB adapter. If possible, we might want to consider writing it as an additional `TimeSeries` in the behavior processing module (similar to `jabs_identity_mask`) so it survives a write/read roundtrip.
