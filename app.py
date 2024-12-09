#!/usr/bin/env python
"""
main program for JABS video labeling and classifier
takes one optional positional argument: path to project directory
"""

import pathlib
import sys

src_path = str(pathlib.Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from jabs.__main__ import main


if __name__ == '__main__':
    main()
