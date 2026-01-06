"""Main window module for JABS UI.

This module provides the MainWindow class and related components.
The MainWindow is refactored into separate files for better organization:
- constants.py: Configuration constants and settings keys
- menu_builder.py: Menu creation and configuration logic
- main_window.py: Core MainWindow class
"""

# Re-export MainWindow from the main_window.py file for backward compatibility
# This allows existing code to continue using:
#   from jabs.ui.main_window import MainWindow
# without needing to know about the internal module structure

from .main_window import MainWindow

__all__ = ["MainWindow"]
