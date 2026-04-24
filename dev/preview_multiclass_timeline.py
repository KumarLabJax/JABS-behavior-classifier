"""Preview script: StackedTimelineWidget with fake multi-class data.

Run with:
    uv run python preview_multiclass_timeline.py

Displays 3 identities, 3 behaviors, ~2 000 frames.  A slider scrubs through
frames; radio buttons toggle binary / multi-class mode and single / all-identity
view so you can compare layouts.
"""

import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from jabs.core.enums import ClassifierMode
from jabs.ui.stacked_timeline_widget import StackedTimelineWidget

# ---------------------------------------------------------------------------
# Fake pose shim (duck-types the attributes/method that _reset_layout needs)
# ---------------------------------------------------------------------------


class _FakePose:
    def __init__(self, num_identities: int, num_frames: int) -> None:
        self.num_identities = num_identities
        self.num_frames = num_frames

    def identity_index_to_display(self, index: int) -> str:
        return f"Mouse {index + 1}"


# ---------------------------------------------------------------------------
# Fake data generation
# ---------------------------------------------------------------------------

NUM_IDENTITIES = 3
NUM_FRAMES = 2000
FRAMERATE = 30
BEHAVIORS = ["grooming", "rearing", "locomotion"]

rng = np.random.default_rng(seed=42)


def _make_bouts(n_frames: int, avg_gap: int = 120, avg_dur: int = 60) -> np.ndarray:
    """Return a boolean mask with random on/off bouts."""
    mask = np.zeros(n_frames, dtype=bool)
    frame = 0
    while frame < n_frames:
        frame += max(10, int(rng.exponential(avg_gap)))
        dur = max(5, int(rng.exponential(avg_dur)))
        mask[frame : frame + dur] = True
        frame += dur
    return mask


def _make_multiclass_labels(n_frames: int) -> np.ndarray:
    """Build a combined class-index label array for one identity.

    Index layout (matches build_multiclass_color_lut):
      0 = unlabeled, 1 = "None", 2 = grooming, 3 = rearing, 4 = locomotion
    """
    labels = np.zeros(n_frames, dtype=np.int16)
    for behavior_idx in range(len(BEHAVIORS)):
        bouts = _make_bouts(n_frames, avg_gap=200, avg_dur=50)
        free = labels == 0
        labels[free & bouts] = behavior_idx + 2
    none_bouts = _make_bouts(n_frames, avg_gap=400, avg_dur=20)
    labels[(labels == 0) & none_bouts] = 1
    return labels


def _make_binary_labels(n_frames: int) -> np.ndarray:
    """Binary LUT-index labels: 1 = not-behavior, 2 = behavior."""
    arr = np.ones(n_frames, dtype=np.int16)
    arr[_make_bouts(n_frames, avg_gap=200, avg_dur=60)] = 2
    return arr


def _make_per_class_predictions(
    n_frames: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-class binary predictions + probabilities for one identity.

    Returns one array per class: [None/background, behA, behB, behC, ...].
    Each prediction array uses the 3-entry per-class LUT:
      0 = no pose, 1 = not this class, 2 = this class predicted.
    """
    behavior_bouts = [_make_bouts(n_frames, avg_gap=200, avg_dur=55) for _ in BEHAVIORS]
    any_behavior = np.zeros(n_frames, dtype=bool)
    for b in behavior_bouts:
        any_behavior |= b

    preds, probs = [], []

    # None/background class: predicted wherever no behavior is active
    none_pred = np.ones(n_frames, dtype=np.int16)
    none_pred[~any_behavior] = 2
    none_prob = np.where(
        ~any_behavior,
        rng.uniform(0.6, 1.0, n_frames),
        rng.uniform(0.0, 0.3, n_frames),
    ).astype(np.float32)
    preds.append(none_pred)
    probs.append(none_prob)

    for bouts in behavior_bouts:
        pred = np.ones(n_frames, dtype=np.int16)
        pred[bouts] = 2
        prob = np.where(
            bouts,
            rng.uniform(0.6, 1.0, n_frames),
            rng.uniform(0.0, 0.3, n_frames),
        ).astype(np.float32)
        preds.append(pred)
        probs.append(prob)

    return preds, probs


# ---------------------------------------------------------------------------
# Pre-generate data for all identities
# ---------------------------------------------------------------------------

MULTICLASS_LABELS = [_make_multiclass_labels(NUM_FRAMES) for _ in range(NUM_IDENTITIES)]
BINARY_LABELS = [_make_binary_labels(NUM_FRAMES) for _ in range(NUM_IDENTITIES)]
MASKS = [np.ones(NUM_FRAMES, dtype=np.int8) for _ in range(NUM_IDENTITIES)]

_mc_data = [_make_per_class_predictions(NUM_FRAMES) for _ in range(NUM_IDENTITIES)]
MULTICLASS_PREDS_LIST = [preds for preds, _ in _mc_data]
MULTICLASS_PROBS_LIST = [probs for _, probs in _mc_data]

BINARY_PREDS_LIST = [[_make_binary_labels(NUM_FRAMES)] for _ in range(NUM_IDENTITIES)]
BINARY_PROBS_LIST = [
    [rng.uniform(0.0, 1.0, NUM_FRAMES).astype(np.float32)] for _ in range(NUM_IDENTITIES)
]

FAKE_POSE = _FakePose(NUM_IDENTITIES, NUM_FRAMES)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class PreviewWindow(QMainWindow):
    """Simple preview window for StackedTimelineWidget multi-class layout."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StackedTimelineWidget — multi-class preview")
        self.resize(1200, 500)

        self._mode = ClassifierMode.MULTICLASS

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ---- Controls row --------------------------------------------------
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(4, 4, 4, 4)

        # Classifier mode toggle
        mode_box = QGroupBox("Classifier mode")
        mode_layout = QHBoxLayout(mode_box)
        self._rb_multiclass = QRadioButton("Multi-class")
        self._rb_binary = QRadioButton("Binary")
        self._rb_multiclass.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self._rb_multiclass)
        mode_group.addButton(self._rb_binary)
        mode_layout.addWidget(self._rb_multiclass)
        mode_layout.addWidget(self._rb_binary)
        self._rb_multiclass.toggled.connect(self._on_mode_toggled)

        # Identity view toggle
        identity_box = QGroupBox("Identity view")
        identity_layout = QHBoxLayout(identity_box)
        self._rb_active = QRadioButton("Active only")
        self._rb_all = QRadioButton("All animals")
        self._rb_active.setChecked(True)
        identity_group = QButtonGroup(self)
        identity_group.addButton(self._rb_active)
        identity_group.addButton(self._rb_all)
        identity_layout.addWidget(self._rb_active)
        identity_layout.addWidget(self._rb_all)
        self._rb_all.toggled.connect(self._on_identity_mode_toggled)

        # Collapse inactive checkboxes (only relevant in all-animals mode)
        collapse_box = QGroupBox("Collapse inactive")
        collapse_layout = QHBoxLayout(collapse_box)
        self._chk_collapse_label = QCheckBox("Label bar")
        self._chk_collapse_label.setChecked(False)
        self._chk_collapse_label.setEnabled(False)
        self._chk_collapse_combined = QCheckBox("Combined bar")
        self._chk_collapse_combined.setChecked(False)
        self._chk_collapse_combined.setEnabled(False)
        self._chk_collapse_per_class = QCheckBox("Per-class bars")
        self._chk_collapse_per_class.setChecked(True)
        self._chk_collapse_per_class.setEnabled(False)
        collapse_layout.addWidget(self._chk_collapse_label)
        collapse_layout.addWidget(self._chk_collapse_combined)
        collapse_layout.addWidget(self._chk_collapse_per_class)
        self._chk_collapse_label.toggled.connect(
            lambda v: setattr(self._timeline, "collapse_inactive_label_bar", v)
        )
        self._chk_collapse_combined.toggled.connect(
            lambda v: setattr(self._timeline, "collapse_inactive_combined_bar", v)
        )
        self._chk_collapse_per_class.toggled.connect(
            lambda v: setattr(self._timeline, "collapse_inactive_per_class_bars", v)
        )

        # Frame slider
        slider_box = QGroupBox(f"Frame  (0 - {NUM_FRAMES - 1})")
        slider_layout = QHBoxLayout(slider_box)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, NUM_FRAMES - 1)
        self._slider.setValue(0)
        self._frame_label = QLabel("0")
        self._frame_label.setFixedWidth(50)
        slider_layout.addWidget(self._slider)
        slider_layout.addWidget(self._frame_label)
        self._slider.valueChanged.connect(self._on_frame_changed)

        # Active identity selector
        identity_sel_box = QGroupBox("Active identity  ([ / ]  or  Tab)")
        identity_sel_layout = QHBoxLayout(identity_sel_box)
        self._btn_prev_id = QPushButton("<")
        self._btn_prev_id.setFixedWidth(28)
        self._btn_next_id = QPushButton(">")
        self._btn_next_id.setFixedWidth(28)
        self._active_id_label = QLabel(self._identity_display(0))
        self._active_id_label.setFixedWidth(70)
        identity_sel_layout.addWidget(self._btn_prev_id)
        identity_sel_layout.addWidget(self._active_id_label)
        identity_sel_layout.addWidget(self._btn_next_id)
        self._btn_prev_id.clicked.connect(self._prev_identity)
        self._btn_next_id.clicked.connect(self._next_identity)

        controls_layout.addWidget(mode_box)
        controls_layout.addWidget(identity_box)
        controls_layout.addWidget(collapse_box)
        controls_layout.addWidget(identity_sel_box)
        controls_layout.addWidget(slider_box, stretch=1)

        root_layout.addWidget(controls)

        # ---- Keyboard shortcuts -------------------------------------------
        QShortcut(QKeySequence(Qt.Key.Key_Tab), self).activated.connect(self._next_identity)
        QShortcut(QKeySequence("["), self).activated.connect(self._prev_identity)
        QShortcut(QKeySequence("]"), self).activated.connect(self._next_identity)

        # ---- Scrollable timeline area -------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._timeline = StackedTimelineWidget()
        scroll.setWidget(self._timeline)
        root_layout.addWidget(scroll, stretch=1)

        # ---- Load initial data -------------------------------------------
        self._timeline.pose = FAKE_POSE
        self._timeline.framerate = FRAMERATE
        self._apply_mode()

    # ------------------------------------------------------------------

    @staticmethod
    def _identity_display(index: int) -> str:
        return FAKE_POSE.identity_index_to_display(index)

    def _set_active_identity(self, index: int) -> None:
        self._timeline.active_identity_index = index
        self._active_id_label.setText(
            f"{self._identity_display(index)}  ({index + 1}/{NUM_IDENTITIES})"
        )

    def _prev_identity(self) -> None:
        current = self._timeline.active_identity_index or 0
        self._set_active_identity((current - 1) % NUM_IDENTITIES)

    def _next_identity(self) -> None:
        current = self._timeline.active_identity_index or 0
        self._set_active_identity((current + 1) % NUM_IDENTITIES)

    def _apply_mode(self) -> None:
        """Push the current mode + fake data into the timeline widget."""
        if self._mode == ClassifierMode.MULTICLASS:
            self._timeline.set_classifier_mode(ClassifierMode.MULTICLASS, BEHAVIORS)
            self._timeline.set_labels(MULTICLASS_LABELS, MASKS)
            self._timeline.set_predictions(MULTICLASS_PREDS_LIST, MULTICLASS_PROBS_LIST)
        else:
            self._timeline.set_classifier_mode(ClassifierMode.BINARY, [])
            self._timeline.set_labels(BINARY_LABELS, MASKS)
            self._timeline.set_predictions(BINARY_PREDS_LIST, BINARY_PROBS_LIST)

    def _on_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._mode = ClassifierMode.MULTICLASS
        else:
            self._mode = ClassifierMode.BINARY
        self._apply_mode()

    def _on_identity_mode_toggled(self, _checked: bool) -> None:
        is_all = self._rb_all.isChecked()
        self._timeline.identity_mode = (
            StackedTimelineWidget.IdentityMode.ALL
            if is_all
            else StackedTimelineWidget.IdentityMode.ACTIVE
        )
        self._chk_collapse_label.setEnabled(is_all)
        self._chk_collapse_combined.setEnabled(is_all)
        self._chk_collapse_per_class.setEnabled(is_all)

    def _on_frame_changed(self, frame: int) -> None:
        self._frame_label.setText(str(frame))
        self._timeline.set_current_frame(frame)


def main() -> None:
    """Entry point."""
    app = QApplication(sys.argv)
    win = PreviewWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
