#!/usr/bin/env python

"""
initialize a JABS project directory

computes features if they do not exist
optional regenerate and overwrite existing feature h5 files
"""
import pathlib
import sys

src_path = str(pathlib.Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from jabs.scripts.initialize_project import main  # noqa: E402

if __name__ == "__main__":
    main()