#!/usr/bin/env python
"""
main program for JABS video labeling and classifier
takes one optional positional argument: path to project directory
"""
import argparse
import sys

from PySide2 import QtWidgets

from src import APP_NAME, APP_NAME_LONG
from src.ui import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow(app_name=APP_NAME, app_name_long=APP_NAME_LONG)

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
