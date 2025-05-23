from PySide6.QtCore import Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..label_overview_widget import LabelOverviewWidget


class StackedLabelOverviewWidget(QWidget):
    """A widget that manages and displays multiple LabelOverviewWidgets, one for each identity.

    This widget allows toggling between showing all identities or only the active one,
    manages selection transfer between identities, and forwards label and frame updates
    to its child widgets. It is designed for use in multi-identity labeling interfaces,
    such as behavioral video annotation tools.

    Properties:
        num_identities (int): Number of identities to display.
        num_frames (int): Number of frames in the video.
        framerate (int): Framerate of the video.
        active_identity_index (int): Index of the currently active identity.
        show_only_active_identity (bool): Whether to display only the active identity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._active_identity_index = None
        self._selection_starting_frame = None
        self._show_only_active_identity = True
        self._num_identities = 0
        self._num_frames = 0
        self._framerate = 0
        self._widgets = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.setLayout(self._layout)

    def _overview_widget_factory(self, parent) -> LabelOverviewWidget:
        """Factory method to create an overview widget."""
        widget = LabelOverviewWidget(parent)
        widget.num_frames = self.num_frames
        widget.framerate = self.framerate
        return widget

    @property
    def num_identities(self):
        """Get the number of identities."""
        return self._num_identities

    @num_identities.setter
    def num_identities(self, value: int):
        """Set the number of identities and reset the layout."""
        if value != self._num_identities:
            self._num_identities = value
            self._reset_layout()

    def _reset_layout(self):
        # Remove old widgets from layout and delete them
        for widget in self._widgets:
            self._layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._widgets = []

        # Create new widgets
        for _ in range(self._num_identities):
            widget = self._overview_widget_factory(self)
            widget.framerate = self.framerate
            widget.num_frames = self.num_frames
            self._widgets.append(widget)
            self._layout.addWidget(widget)

    @property
    def num_frames(self):
        """Get the number of frames."""
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value: int):
        """Set the number of frames."""
        self._num_frames = value

    @property
    def framerate(self):
        """Get the framerate."""
        return self._framerate

    @framerate.setter
    def framerate(self, value: int):
        """Set the framerate."""
        self._framerate = value

    @property
    def active_identity_index(self):
        """Get the index of the active identity."""
        return self._active_identity_index

    @active_identity_index.setter
    def active_identity_index(self, value: int):
        """Set the index of the active identity."""
        if value != self._active_identity_index:
            old_index = self._active_identity_index
            selection_frame = self._selection_starting_frame

            # Clear selection on old active widget if selection is active
            if old_index is not None and selection_frame is not None:
                self._widgets[old_index].clear_selection()

            # Deactivate old, activate new
            if old_index is not None:
                self._widgets[old_index].active = False
            self._active_identity_index = value
            if not self._show_only_active_identity:
                self._widgets[self._active_identity_index].active = True
            self._update_widget_visibility()

            # Transfer selection to new active widget if selection is active
            if selection_frame is not None:
                self._widgets[self._active_identity_index].start_selection(
                    selection_frame
                )

    @property
    def show_only_active_identity(self):
        """Whether to display only the active identity."""
        return self._show_only_active_identity

    @show_only_active_identity.setter
    def show_only_active_identity(self, value: bool):
        if self._show_only_active_identity != value:
            self._show_only_active_identity = value
            self._update_widget_visibility()

    def _update_widget_visibility(self):
        if self._show_only_active_identity and self._active_identity_index is not None:
            for i, widget in enumerate(self._widgets):
                widget.setVisible(i == self._active_identity_index)
        else:
            for widget in self._widgets:
                widget.setVisible(True)

    @Slot(int)
    def set_current_frame(self, current_frame: int):
        """Forward current frame to all LabelOverviewWidgets."""
        for widget in self._widgets:
            widget.set_current_frame(current_frame)

    def set_labels(self, labels_list, masks_list=None):
        """
        Set labels for all LabelOverviewWidgets.

        Args:
            labels_list: List of TrackLabels, one per identity.
            masks_list: Optional list of masks, one per identity.
        """
        if len(labels_list) != self._num_identities:
            raise ValueError(
                f"Number of labels ({len(labels_list)}) does not match number of identities ({self._num_identities})."
            )

        for i, widget in enumerate(self._widgets):
            labels = labels_list[i]
            mask = masks_list[i] if masks_list else None

            # need to set the number of frames and framerate on the child widgets because they were zero when they
            # were created. Now that data has been loaded, they can be set to the correct values.
            widget.num_frames = self.num_frames
            widget.framerate = self.framerate

            widget.set_labels(labels, mask)
            widget.update_labels()

    def start_selection(self, starting_frame: int):
        """Start selection on the active identity's widget and record the starting frame."""
        if self._active_identity_index is not None:
            self._selection_starting_frame = starting_frame
            self._widgets[self._active_identity_index].start_selection(starting_frame)
            self._widgets[self._active_identity_index].update()

    def clear_selection(self):
        """Clear selection on the active identity's widget and reset the starting frame."""
        if self._active_identity_index is not None:
            self._widgets[self._active_identity_index].clear_selection()
            self._selection_starting_frame = None
            self.update_labels()

    def update_labels(self):
        """Call update_labels on all child LabelOverviewWidgets."""
        for widget in self._widgets:
            widget.update_labels()

    def reset(self):
        """reset state of all child widgets"""
        for widget in self._widgets:
            widget.reset()
