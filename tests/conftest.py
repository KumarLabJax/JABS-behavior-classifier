"""Shared test configuration for the application test suite."""

import os

# Render Qt widgets offscreen so UI tests don't open windows and steal focus. Set before
# any test module creates a QApplication; override with e.g. QT_QPA_PLATFORM=cocoa to
# watch the widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
