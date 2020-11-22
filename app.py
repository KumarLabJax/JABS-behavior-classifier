#!/usr/bin/env python
"""
main program for Rotta video labeler and classifier
takes one positional argument: path to video file
"""

import argparse
import sys

from PyQt5 import QtWidgets

from src.ui import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow(app_name="Rotta")

    parser = argparse.ArgumentParser()

    parser.add_argument('project_dir', nargs='?')
    args = parser.parse_args()

    if args.project_dir is not None:
        try:
            main_window.open_project(args.project_dir)
        except Exception as e:
            sys.exit(f"Error opening project:  {e}")

    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
