"""Package providing widgets for label and prediction overview in the user interface.

This package includes:
- LabelOverviewWidget: Widget for displaying labels.
- PredictionOverviewWidget: Widget for displaying prediction data.

These widgets are used to visualize labels and predictions for different identities within the application. While the
user is labeling, the LabelOverviewWidget is used to indicate which frames are currently selected for labeling. The
PredictionOverviewWidget is used to display the predictions made by the model.
"""

from .label_overview_widget import LabelOverviewWidget
from .prediction_overview_widget import PredictionOverviewWidget

__all__ = [
    "LabelOverviewWidget",
    "PredictionOverviewWidget",
]
