"""Package providing the StackedTimelineWidget for visualizing label and prediction timelines.

This package includes:
- StackedTimelineWidget: A widget that displays label and prediction overviews for multiple identities,
  allowing users to view labels and predictions in a stacked timeline format. Also used for indicating
  which frames are currently selected for labeling.

The widget supports switching between different identities, selection modes, and view modes for efficient
annotation and review.
"""

from .stacked_timeline_widget import StackedTimelineWidget
