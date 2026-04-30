"""BehaviorTimelineWidget — multi-identity behavior annotation timeline panel.

This package includes:
- BehaviorTimelineWidget: Displays label and prediction tracks for all identities in a
  video, with support for identity switching, selection modes, and binary/multi-class
  view modes.
"""

from .behavior_timeline_widget import BehaviorTimelineWidget
from .track_widgets.timeline_util import (
    binary_predictions_to_lut_indices,
    track_labels_to_lut_indices,
)
