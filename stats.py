#!/usr/bin/env python
import pathlib
import sys

src_path = str(pathlib.Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from jabs.scripts.stats import main  # noqa: E402

if __name__ == "__main__":
    main()
