import argparse
import sys

from PySide6 import QtWidgets

from jabs.ui import MainWindow
from jabs.constants import APP_NAME, APP_NAME_LONG


def main():
    """
    main entrypoint for JABS video labeling and classifier
    takes one optional positional argument: path to project directory
    """

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
    if main_window.show_license_dialog() == QtWidgets.QDialog.Accepted:
        # user accepted license terms, run the main application loop
        sys.exit(app.exec())

    # user rejected license terms
    sys.exit(1)