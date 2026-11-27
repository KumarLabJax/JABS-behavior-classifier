"""Interactive non-modal BALE - Bout & Label Explorer Dialog (`Ctrl+Shift+B`).

Simplified 4-Mode Triage & Screening workflow
(`1. Audit Labels` | `2. Review Predictions` | `3. Sub-Types` | `4. Videos`)
with `< 25 ms` in-place relabeling, whole-video 50k-style outlier & zero-behavior screening,
user-controlled bout/video selection & one-way Picker -> Video linking,
1-click video clip playback (`Space`), frame-probability / whole-video timeline click scrubbing,
JABS Set1/Binary color parity, and macOS ARM64 PySide6 safety.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import threading
import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from jabs.bale import (
    BaleReport,
    BaleUndoStack,
    BoutRecord,
    BoutSpatialResult,
    EditAction,
    ExaminedBoutStore,
    FeatureLens,
    ProjectionMethod,
    RepresentationSpace,
    SubtypeScope,
    VideoMetricSource,
    VideoScreenerReport,
    VideoScreeningPreset,
    VideoSummaryRecord,
    apply_bout_edit,
    classify_video_outliers,
    draw_arena_static_landmarks,
    evaluate_cv_group_impact,
    export_bale_summary_figure,
    extract_bout_spatial_locations,
    filter_video_records,
    load_bout_pose_trajectory,
    recompute_embedding_only,
    run_bale,
    select_feature_lens_mask,
    simulate_track_count_delta,
    summarize_cv_track_counts,
    summarize_project_videos,
)
from jabs.bale.spatial_localization import clear_spatial_cache
from jabs.bale.trajectory_and_export import clear_pose_cache
from jabs.behavior_search import SearchHit
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode
from jabs.project import TrackLabels
from jabs.ui.colors import BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR, make_behavior_color_map

from .message_dialog import MessageDialog
from .user_guide_dialog import UserGuideDialog

if TYPE_CHECKING:
    from jabs.project import Project, VideoLabels
    from jabs.ui.main_window.central_widget import CentralWidget

logger = logging.getLogger(__name__)


class _SortableTableItem(QtWidgets.QTableWidgetItem):
    """QTableWidgetItem that sorts by an explicit Python key without Qt QVariant marshaling."""

    def __init__(self, text: str = "", sort_key: Any = None) -> None:
        super().__init__(text)
        self._sort_key: Any = sort_key

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        a = getattr(self, "_sort_key", None)
        b = getattr(other, "_sort_key", None)
        if a is not None and b is not None:
            try:
                return bool(a < b)
            except TypeError:
                pass
        return self.text().lower() < other.text().lower()


ExplorerWorkflowMode = Literal["audit", "predict", "subtypes", "videos"]
QueueFilterMode = Literal["needs_review", "label_edges", "all", "reviewed"]
VideoQuickPillMode = Literal["all", "high_frames", "high_bouts", "zero_all", "any_outlier"]
ScatterDragMode = Literal["box", "lasso", "zoom"]
InspectorViewMode = Literal["prob", "feat", "pose", "clusters"]

# Muted journal-grade styling for 50k whole-video screening buckets: (bg_hex, fg_hex, dot_hex)
_VIDEO_BUCKET_STYLE: dict[str, tuple[str, str, str]] = {
    "E1 High Frames & Bouts": ("#DBEAFE", "#1E3A8A", "#1E40AF"),
    "E1 High Frames": ("#DBEAFE", "#1E40AF", "#2563EB"),
    "E1 High Bouts": ("#E0F2FE", "#075985", "#0284C7"),
    "E1 Chiral Asymmetry": ("#EDE9FE", "#5B21B6", "#6D28D9"),
    "E2 Long Bout": ("#E0E7FF", "#3730A3", "#4338CA"),
    "E3a+E3b Uncertain": ("#FEE2E2", "#991B1B", "#B91C1C"),
    "E3a Low-Conf Beh": ("#FEE2E2", "#991B1B", "#DC2626"),
    "E3b Near-Miss None": ("#FEF3C7", "#92400E", "#C2410C"),
    "E4 Fragile / Disagree": ("#FCE7F3", "#9D174D", "#BE185D"),
    "E5 Zero (+ Near-Miss)": ("#FEF3C7", "#92400E", "#D97706"),
    "E5 Zero (High-Conf)": ("#CCFBF1", "#115E59", "#0F766E"),
    "E5 Zero Behavior": ("#F1F5F9", "#334155", "#64748B"),
    "Normal": ("#F8FAFC", "#475569", "#94A3B8"),
}

_FEATURE_LABEL_REPLACEMENTS = (
    ("centroid_velocity_mag", "centroid speed"),
    ("angular_velocity", "ang vel"),
    ("huddle_features", "huddle"),
    ("pairwise_distances", "dist"),
    ("shape_descriptor", "shape"),
    ("point_speeds", "speed"),
    ("hu_moments", "hu"),
    ("connected group size", "grp sz"),
    ("angles", "angle"),
)


def _short_feature_label(name: str, max_len: int = 20) -> str:
    """Shorten verbose JABS per-frame feature statistic names for compact plots."""
    txt = str(name)
    for stat_suf, short_suf in (
        ("_mean", " mean"),
        ("_q10", " p10"),
        ("_q25", " p25"),
        ("_q50", " p50"),
        ("_q75", " p75"),
        ("_q90", " p90"),
        ("_mad", " mad"),
        ("_iqr", " iqr"),
    ):
        if txt.endswith(stat_suf):
            txt = txt[: -len(stat_suf)] + short_suf
            break
    for src, dst in _FEATURE_LABEL_REPLACEMENTS:
        txt = txt.replace(src, dst)
    txt = " ".join(txt.split())
    if len(txt) <= max_len:
        return txt
    return f"{txt[: max(4, max_len - 1)].rstrip()}…"


# High-contrast, colorblind-aware cluster palette (restrained journal tones)
_CLUSTER_PALETTE: tuple[str, ...] = (
    "#1E40AF",
    "#0F766E",
    "#B91C1C",
    "#7C3AED",
    "#C2410C",
    "#0369A1",
    "#4D7C0F",
    "#BE185D",
)

_CAGE_DATE_RE = re.compile(r"(cage_\d+).*?(\d{4}-(\d{2}-\d{2}))[._](\d{2})[._](\d{2})")


def _short_video_name(video: str) -> str:
    """Format long JABS filenames into scannable `cage • MM-DD HH:MM` labels."""
    m = _CAGE_DATE_RE.search(video)
    if m:
        return f"{m.group(1)} • {m.group(3)} {m.group(4)}:{m.group(5)}"
    stem = Path(video).stem
    if stem.endswith(".top"):
        stem = stem[:-4]
    return stem if len(stem) <= 28 else f"{stem[:14]}…{stem[-11:]}"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Return natural singular/plural text (e.g. `1 bout` vs. `4 bouts`)."""
    form = singular if int(count) == 1 else (plural or f"{singular}s")
    return f"{int(count)} {form}"


_PLAY_CLIP_PRIMARY_CSS = (
    "QPushButton { background-color: #0F172A; color: #FFFFFF; font-weight: 700; "
    "padding: 6px 13px; border-radius: 5px; border: none; }"
    "QPushButton:hover { background-color: #1E293B; }"
    "QPushButton:disabled { background-color: #CBD5E1; color: #64748B; }"
)
_PLAY_CLIP_SECONDARY_CSS = (
    "QPushButton { background-color: #FFFFFF; color: #0F172A; font-weight: 700; "
    "padding: 5px 11px; border-radius: 5px; border: 1.5px solid #334155; }"
    "QPushButton:hover { background-color: #F1F5F9; }"
    "QPushButton:disabled { background-color: #F8FAFC; color: #94A3B8; border-color: #CBD5E1; }"
)


def _qcolor_to_hex(color: QtGui.QColor, darken_light: bool = True) -> str:
    """Convert a `QColor` to hex, slightly darkening bright yellow fills for white plots."""
    r, g, b = color.red(), color.green(), color.blue()
    if darken_light and (0.2126 * r + 0.7152 * g + 0.0722 * b) > 210:
        return "#CA8A04"  # readable warm gold for Set1 Yellow on white background
    return f"#{r:02X}{g:02X}{b:02X}"


class _BaleWorkerSignals(QtCore.QObject):
    """Signals emitted by the background `QRunnable` explorer worker."""

    finished = QtCore.Signal(object, bool)  # BaleReport, included_preds
    error = QtCore.Signal(str)


class _VideoScreenerSignals(QtCore.QObject):
    """Signals emitted by the background Mode 4 video screener worker."""

    finished = QtCore.Signal(object, object, int)  # VideoScreenerReport, key, req_id
    cancelled = QtCore.Signal(int)  # req_id
    error = QtCore.Signal(str, int)  # err_msg, req_id


class _ExplorerComputeTask(QtCore.QRunnable):
    """Background worker task for `run_bale` so the Qt GUI thread never blocks."""

    def __init__(
        self,
        project: Project,
        behavior: str | None,
        *,
        include_positive: bool,
        include_negative: bool,
        include_all_multiclass: bool,
        include_predictions: bool,
        max_predicted_bouts: int,
        space: RepresentationSpace,
        subtype_scope: SubtypeScope = "all_labeled",
        feature_lens: FeatureLens = "all",
        target_behavior: str | None = None,
        forced_k: int | None = None,
        compute_tsne: bool = False,
        compute_mds: bool = False,
        compute_spatial: bool = False,
    ) -> None:
        super().__init__()
        self.project = project
        self.behavior = behavior
        self.include_positive = include_positive
        self.include_negative = include_negative
        self.include_all_multiclass = include_all_multiclass
        self.include_predictions = include_predictions
        self.max_predicted_bouts = max_predicted_bouts
        self.space = space
        self.subtype_scope = subtype_scope
        self.feature_lens = feature_lens
        self.target_behavior = target_behavior
        self.forced_k = forced_k
        self.compute_tsne = compute_tsne
        self.compute_mds = compute_mds
        self.compute_spatial = compute_spatial
        self.cancel_event = threading.Event()
        self.signals = _BaleWorkerSignals()

    def run(self) -> None:
        if self.cancel_event.is_set():
            return
        try:
            report = run_bale(
                self.project,
                self.behavior,
                include_positive=self.include_positive,
                include_negative=self.include_negative,
                include_all_multiclass=self.include_all_multiclass,
                include_predictions=self.include_predictions,
                max_predicted_bouts=self.max_predicted_bouts,
                space=self.space,
                subtype_scope=self.subtype_scope,
                feature_lens=self.feature_lens,
                target_behavior=self.target_behavior,
                forced_k=self.forced_k,
                compute_tsne=self.compute_tsne,
                compute_mds=self.compute_mds,
                compute_spatial=self.compute_spatial,
                cancel_event=self.cancel_event,
            )
            if self.cancel_event.is_set():
                return
            with contextlib.suppress(RuntimeError):
                self.signals.finished.emit(report, self.include_predictions)
        except Exception as exc:
            if self.cancel_event.is_set():
                return
            logger.exception("BALE background task failed")
            with contextlib.suppress(RuntimeError):
                self.signals.error.emit(str(exc))


class _SpatialWorkerSignals(QtCore.QObject):
    """Signals emitted by the lazy arena spatial-localization worker."""

    finished = QtCore.Signal(int, object)  # req_id, BoutSpatialResult
    cancelled = QtCore.Signal(int)  # req_id
    error = QtCore.Signal(int, str)


class _SpatialComputeTask(QtCore.QRunnable):
    """Background worker computing arena coordinates off the Qt GUI thread."""

    def __init__(
        self,
        req_id: int,
        project: Project,
        records: list[BoutRecord],
    ) -> None:
        super().__init__()
        self.req_id = int(req_id)
        self.project = project
        self.records = list(records)
        self.cancel_event = threading.Event()
        self.signals = _SpatialWorkerSignals()

    def run(self) -> None:
        if self.cancel_event.is_set():
            with contextlib.suppress(RuntimeError):
                self.signals.cancelled.emit(self.req_id)
            return
        try:
            res = extract_bout_spatial_locations(
                self.project, self.records, cancel_event=self.cancel_event
            )
            if self.cancel_event.is_set():
                with contextlib.suppress(RuntimeError):
                    self.signals.cancelled.emit(self.req_id)
                return
            with contextlib.suppress(RuntimeError):
                self.signals.finished.emit(self.req_id, res)
        except Exception as exc:
            if self.cancel_event.is_set():
                with contextlib.suppress(RuntimeError):
                    self.signals.cancelled.emit(self.req_id)
                return
            logger.exception("BALE spatial localization failed")
            with contextlib.suppress(RuntimeError):
                self.signals.error.emit(self.req_id, str(exc))


class _VideoScreenerTask(QtCore.QRunnable):
    """Background worker task for `summarize_project_videos` so the GUI never blocks."""

    def __init__(
        self,
        project: Project,
        behavior: str | None,
        *,
        include_all_multiclass: bool,
        use_postprocessed: bool = True,
        source: VideoMetricSource = "predictions",
        key: tuple[str, bool] = ("", False),
        req_id: int = 0,
    ) -> None:
        super().__init__()
        self.project = project
        self.behavior = behavior
        self.include_all_multiclass = include_all_multiclass
        self.use_postprocessed = use_postprocessed
        self.source = source
        self.key = key
        self.req_id = int(req_id)
        self.cancel_event = threading.Event()
        self.signals = _VideoScreenerSignals()

    def run(self) -> None:
        if self.cancel_event.is_set():
            with contextlib.suppress(RuntimeError):
                self.signals.cancelled.emit(self.req_id)
            return
        try:
            report = summarize_project_videos(
                self.project,
                self.behavior,
                include_all_multiclass=self.include_all_multiclass,
                use_postprocessed=self.use_postprocessed,
                source=self.source,
                cancel_event=self.cancel_event,
            )
            if self.cancel_event.is_set():
                with contextlib.suppress(RuntimeError):
                    self.signals.cancelled.emit(self.req_id)
                return
            with contextlib.suppress(RuntimeError):
                self.signals.finished.emit(report, self.key, self.req_id)
        except Exception as exc:
            if self.cancel_event.is_set():
                with contextlib.suppress(RuntimeError):
                    self.signals.cancelled.emit(self.req_id)
                return
            logger.exception("BALE video screener task failed")
            with contextlib.suppress(RuntimeError):
                self.signals.error.emit(str(exc), self.req_id)


class _ElidedLabel(QtWidgets.QLabel):
    """Single-line label that elides gracefully on narrow windows without locking dialog width."""

    def __init__(
        self,
        text: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        elide_mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight,
    ) -> None:
        super().__init__(parent)
        self._full_text: str = str(text)
        self._explicit_tooltip: str = ""
        self._elide_mode = elide_mode
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.setMinimumWidth(0)
        self._sync_tooltip()

    def text(self) -> str:
        """Return full un-elided text so callers and tests always read complete content."""
        return self._full_text

    def setText(self, text: str) -> None:
        """Store full text, refresh tooltip, and repaint elided to current width."""
        self._full_text = str(text)
        self._sync_tooltip()
        self.updateGeometry()
        self.update()

    def setToolTip(self, tip: str) -> None:
        """Override automatic hover tooltip with an explicit caller tooltip."""
        self._explicit_tooltip = str(tip)
        super().setToolTip(self._explicit_tooltip or self._full_text)

    def _sync_tooltip(self) -> None:
        if not self._explicit_tooltip:
            super().setToolTip(self._full_text)

    def minimumSizeHint(self) -> QtCore.QSize:
        """Allow shrinking down to 0 px width so long strings never lock window resize."""
        h = max(16, self.fontMetrics().height() + 4)
        return QtCore.QSize(0, h)

    def sizeHint(self) -> QtCore.QSize:
        """Provide a compact preferred width hint while keeping natural text height."""
        fm = self.fontMetrics()
        w = min(260, fm.horizontalAdvance(self._full_text) + 6)
        h = max(18, fm.height() + 4)
        return QtCore.QSize(w, h)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        """Render elided single-line text using the active Qt style and stylesheet palette."""
        painter = QtGui.QPainter(self)
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, painter, self)
        cr = self.contentsRect()
        if cr.width() <= 0 or cr.height() <= 0 or not self._full_text:
            return
        elided = self.fontMetrics().elidedText(self._full_text, self._elide_mode, cr.width())
        align = int(self.alignment() | Qt.AlignmentFlag.AlignVCenter)
        self.style().drawItemText(
            painter,
            cr,
            align,
            self.palette(),
            self.isEnabled(),
            elided,
            self.foregroundRole(),
        )


class _FlowLayout(QtWidgets.QLayout):
    """Cross-platform responsive wrapping layout so toolbars wrap cleanly on narrow windows."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        h_spacing: int = 6,
        v_spacing: int = 4,
    ) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._h_spacing = int(h_spacing)
        self._v_spacing = int(v_spacing)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        """Append a layout item to the wrapping flow."""
        self._items.append(item)

    def count(self) -> int:
        """Return total number of tracked items."""
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        """Return item at `index` or `None` if out of bounds."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        """Remove and return item at `index`."""
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        """Flow layouts grow vertically as needed rather than forcing horizontal expansion."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        """Enable height-for-width recalculation as the parent window resizes horizontally."""
        return True

    def heightForWidth(self, width: int) -> int:
        """Compute total wrapped height for `width` pixels."""
        return self._do_layout(QtCore.QRect(0, 0, max(1, int(width)), 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        """Lay out all visible children inside `rect`."""
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:
        """Return compact preferred size without locking dialog minimum width."""
        margins = self.contentsMargins()
        visible = self._visible_items()
        if not visible:
            return QtCore.QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        total_w = 0
        max_h = 0
        for i, item in enumerate(visible):
            sz = self._item_effective_size(item)
            total_w += sz.width() + (self._h_spacing if i > 0 else 0)
            max_h = max(max_h, sz.height())
        return QtCore.QSize(
            min(total_w, 640) + margins.left() + margins.right(),
            max_h + margins.top() + margins.bottom(),
        )

    def minimumSize(self) -> QtCore.QSize:
        """Return only the widest single visible child so horizontal resizing is never blocked."""
        margins = self.contentsMargins()
        max_w = 0
        max_h = 0
        for item in self._visible_items():
            w_obj = item.widget()
            if isinstance(w_obj, _ElidedLabel):
                sz = item.minimumSize()
            else:
                sz = item.minimumSize()
                if sz.width() <= 0 or sz.height() <= 0:
                    sh = item.sizeHint()
                    sz = QtCore.QSize(max(sz.width(), sh.width()), max(sz.height(), sh.height()))
            max_w = max(max_w, min(sz.width(), 220))
            max_h = max(max_h, sz.height())
        return QtCore.QSize(
            max_w + margins.left() + margins.right(),
            max_h + margins.top() + margins.bottom(),
        )

    def _visible_items(self) -> list[QtWidgets.QLayoutItem]:
        out: list[QtWidgets.QLayoutItem] = []
        for item in self._items:
            w_obj = item.widget()
            if w_obj is not None and w_obj.isHidden():
                continue
            out.append(item)
        return out

    @staticmethod
    def _item_effective_size(item: QtWidgets.QLayoutItem) -> QtCore.QSize:
        sh = item.sizeHint()
        w_obj = item.widget()
        if w_obj is not None:
            min_w = w_obj.minimumWidth()
            max_w = w_obj.maximumWidth()
            w = max(sh.width(), min_w)
            if 0 < max_w < 16777215:
                w = min(w, max_w)
            h = max(sh.height(), w_obj.minimumHeight())
            return QtCore.QSize(w, h)
        return sh

    def _do_layout(self, rect: QtCore.QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        eff = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        avail_w = max(1, eff.width())
        visible = self._visible_items()
        if not visible:
            return margins.top() + margins.bottom()

        rows: list[list[tuple[QtWidgets.QLayoutItem, int, int]]] = []
        curr_row: list[tuple[QtWidgets.QLayoutItem, int, int]] = []
        curr_x = 0

        for item in visible:
            w_obj = item.widget()
            sz = self._item_effective_size(item)
            iw = sz.width()
            ih = max(1, sz.height())
            if isinstance(w_obj, _ElidedLabel):
                rem = avail_w - curr_x - (self._h_spacing if curr_row else 0)
                if curr_row and rem < 140:
                    rows.append(curr_row)
                    curr_row = []
                    curr_x = 0
                    rem = avail_w
                iw = max(60, rem)
            else:
                iw = min(iw, avail_w)
                next_x = curr_x + (self._h_spacing if curr_row else 0) + iw
                if curr_row and next_x > avail_w:
                    rows.append(curr_row)
                    curr_row = []
                    curr_x = 0
            curr_row.append((item, iw, ih))
            curr_x += (self._h_spacing if len(curr_row) > 1 else 0) + iw

        if curr_row:
            rows.append(curr_row)

        y = eff.y()
        for r_idx, row in enumerate(rows):
            if r_idx > 0:
                y += self._v_spacing
            row_h = max(ih for _, _, ih in row)
            if not test_only:
                x = eff.x()
                for item, iw, ih in row:
                    item_y = y + (row_h - ih) // 2
                    item.setGeometry(QtCore.QRect(x, item_y, iw, ih))
                    x += iw + self._h_spacing
            y += row_h

        return (y - eff.y()) + margins.top() + margins.bottom()


class _InteractiveAggCanvas(QtWidgets.QWidget):
    """Pure-Qt interactive Matplotlib canvas rendered offscreen via `FigureCanvasAgg`.

    Eliminates `FigureCanvasQTAgg` / `NavigationToolbar2QT` / `RectangleSelector`
    cursor mutations inside `paintEvent` (`QWidget::setCursor` -> `QImage::toCGImage()`
    pointer authentication crash on macOS ARM64 with PySide6 6.11).
    Supports click-to-inspect, box/lasso selection (`Shift` additive), wheel zoom,
    zoom-box, smooth right/middle/Alt+left pan, double-click zoom reset, and cohort
    context menus with zero `QWidget` cursor mutations.
    """

    point_clicked = QtCore.Signal(int)
    box_selected = QtCore.Signal(float, float, float, float, bool)
    lasso_selected = QtCore.Signal(object, bool)
    zoom_box_selected = QtCore.Signal(float, float, float, float)
    wheel_zoomed = QtCore.Signal(float, float, float)
    pan_dragged = QtCore.Signal(float, float, float, float)
    reset_zoom_requested = QtCore.Signal()
    context_menu_requested = QtCore.Signal(QtCore.QPoint, int)
    x_value_clicked = QtCore.Signal(float)
    y_value_clicked = QtCore.Signal(float)

    def __init__(
        self,
        figure: Figure,
        ax: Any,
        *,
        interactive_scatter: bool = False,
        interactive_x_scrub: bool = False,
        interactive_y_pick: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.figure = figure
        self.ax = ax
        self._agg = FigureCanvasAgg(self.figure)
        self._interactive_scatter = interactive_scatter
        self._interactive_x_scrub = interactive_x_scrub
        self._interactive_y_pick = interactive_y_pick
        self._drag_mode: ScatterDragMode = "box"
        self._pixmap: QtGui.QPixmap | None = None
        self._pick_coords: np.ndarray | None = None
        self._pick_global_indices: np.ndarray | None = None
        self._hover_labels: list[str] = []
        self._hover_idx: int | None = None
        self._hover_pos: QtCore.QPointF | None = None
        self._press_pos: QtCore.QPointF | None = None
        self._drag_pos: QtCore.QPointF | None = None
        self._lasso_screen_pts: list[QtCore.QPointF] = []
        self._panning = False
        self._pan_button: Qt.MouseButton | None = None
        self._pan_press_pos: QtCore.QPointF | None = None
        self._pan_drag_exceeded = False
        self._pan_start_mpl: tuple[float, float] = (0.0, 0.0)
        self._pan_start_xlim: tuple[float, float] = (0.0, 1.0)
        self._pan_start_ylim: tuple[float, float] = (0.0, 1.0)
        self._pan_bbox_wh: tuple[float, float] = (1.0, 1.0)
        self._scrubbing_x = False
        self._render_pending = False
        self._layout_dirty = True
        self._last_layout_key: tuple[int, int, float] | None = None
        self._rendered_xlim: tuple[float, float] | None = None
        self._rendered_ylim: tuple[float, float] | None = None
        self._rendered_ax_rect: QtCore.QRectF | None = None
        self._is_disposed = False
        self._idle_render_timer = QtCore.QTimer(self)
        self._idle_render_timer.setSingleShot(True)
        self._idle_render_timer.setInterval(0)
        self._idle_render_timer.timeout.connect(self._render_now)
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(50)
        self._resize_timer.timeout.connect(self._render_now)
        self._viewport_timer = QtCore.QTimer(self)
        self._viewport_timer.setSingleShot(True)
        self._viewport_timer.setInterval(60)
        self._viewport_timer.timeout.connect(self._render_now)
        self.setMouseTracking(interactive_scatter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    @property
    def drag_mode(self) -> ScatterDragMode:
        return self._drag_mode

    def set_drag_mode(self, mode: ScatterDragMode) -> None:
        """Switch left-drag behavior between `'box'`, `'lasso'`, and `'zoom'`."""
        self._drag_mode = mode
        self._press_pos = None
        self._drag_pos = None
        self._lasso_screen_pts.clear()
        self._hover_idx = None
        self._hover_pos = None
        self.update()

    def mark_layout_dirty(self) -> None:
        """Request `tight_layout` recalculation at the exact widget geometry on next render."""
        self._layout_dirty = True

    def set_pick_points(
        self,
        coords_2d: np.ndarray | None,
        global_indices: np.ndarray | None = None,
        hover_labels: list[str] | None = None,
    ) -> None:
        """Store visible `(M, 2)` coordinates and their global bout indices for picking & tooltips."""
        self._pick_coords = coords_2d
        if coords_2d is not None and global_indices is None:
            self._pick_global_indices = np.arange(len(coords_2d), dtype=np.int32)
        else:
            self._pick_global_indices = global_indices
        self._hover_labels = list(hover_labels) if hover_labels is not None else []
        self._hover_idx = None
        self._hover_pos = None

    def dispose(self) -> None:
        """Stop all queued and coalesced render timers before widget deletion."""
        self._is_disposed = True
        self._render_pending = False
        self._idle_render_timer.stop()
        self._resize_timer.stop()
        self._viewport_timer.stop()

    def draw_idle(self) -> None:
        """Schedule a coalesced offscreen Agg render on the Qt event loop."""
        if self._is_disposed:
            return
        self._resize_timer.stop()
        self._viewport_timer.stop()
        if not self._render_pending:
            self._render_pending = True
            self._idle_render_timer.start()

    def draw_viewport_preview(self) -> None:
        """Blit cached plot pixels immediately during pan/wheel zoom (`60+ FPS`), settling cleanly after 60 ms."""
        if self._is_disposed:
            return
        if (
            self._pixmap is None
            or self._pixmap.isNull()
            or not self.isVisible()
            or self._rendered_ax_rect is None
            or self._rendered_xlim is None
            or self._rendered_ylim is None
        ):
            self.draw_idle()
            return
        self._viewport_timer.start()
        self.update()

    def _render_now(self) -> None:
        self._render_pending = False
        if self._is_disposed:
            return
        self._resize_timer.stop()
        self._viewport_timer.stop()
        w_log = max(80, self.width())
        h_log = max(80, self.height())
        dpr = max(1.0, float(self.devicePixelRatioF()))
        base_dpi = 100.0
        self.figure.set_dpi(base_dpi * dpr)
        self.figure.set_size_inches(w_log / base_dpi, h_log / base_dpi, forward=False)
        layout_key = (w_log, h_log, round(dpr, 3))
        if self._layout_dirty or layout_key != self._last_layout_key:
            with contextlib.suppress(Exception), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.figure.tight_layout(pad=0.65)
            self._last_layout_key = layout_key
            self._layout_dirty = False
        try:
            self._agg.draw()
            buf = self._agg.buffer_rgba()
            w_px, h_px = self._agg.get_width_height()
            qimg = QtGui.QImage(
                buf,
                w_px,
                h_px,
                4 * w_px,
                QtGui.QImage.Format.Format_RGBA8888,
            ).copy()
            qimg.setDevicePixelRatio(dpr)
            self._pixmap = QtGui.QPixmap.fromImage(qimg)
            bbox = self.ax.bbox
            self._rendered_xlim = tuple(float(v) for v in self.ax.get_xlim())
            self._rendered_ylim = tuple(float(v) for v in self.ax.get_ylim())
            self._rendered_ax_rect = QtCore.QRectF(
                float(bbox.x0) / dpr,
                float(h_log) - float(bbox.y1) / dpr,
                max(1.0, float(bbox.width) / dpr),
                max(1.0, float(bbox.height) / dpr),
            )
        except Exception:
            logger.exception("Failed to rasterize Matplotlib figure via FigureCanvasAgg")
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._is_disposed:
            return
        if self._pixmap is None or self._pixmap.isNull() or not self.isVisible():
            self.draw_idle()
        else:
            self._viewport_timer.stop()
            self._resize_timer.start()
            self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Clear in-canvas scatter hover callout cleanly on cursor exit (zero NSWindow tooltip moves)."""
        if self._hover_idx is not None or self._hover_pos is not None:
            self._hover_idx = None
            self._hover_pos = None
            self.update()
        super().leaveEvent(event)

    def _draw_hover_card(self, painter: QtGui.QPainter) -> None:
        """Paint an in-canvas callout pill near the hovered bout without spawning/moving a native tooltip window."""
        if self._hover_idx is None or not (0 <= self._hover_idx < len(self._hover_labels)):
            return
        anchor = self._hover_pos
        if self._pick_coords is not None and 0 <= self._hover_idx < len(self._pick_coords):
            try:
                px, py = self.ax.transData.transform(self._pick_coords[self._hover_idx])
                dpr = max(1.0, float(self.devicePixelRatioF()))
                anchor = QtCore.QPointF(float(px) / dpr, float(self.height()) - float(py) / dpr)
            except Exception:
                anchor = self._hover_pos
        if anchor is None:
            return
        text = self._hover_labels[self._hover_idx]
        lines = text.splitlines() or [text]
        font = QtGui.QFont("Arial", 10)
        fm = QtGui.QFontMetricsF(font)
        line_h = fm.height() + 1.5
        max_w = max((fm.horizontalAdvance(line) for line in lines), default=0.0)
        pad_x, pad_y = 8.0, 5.0
        box_w = max_w + 2.0 * pad_x
        box_h = len(lines) * line_h + 2.0 * pad_y

        bounds = QtCore.QRectF(self.rect()).adjusted(6.0, 6.0, -6.0, -6.0)
        bx = anchor.x() + 12.0
        if bx + box_w > bounds.right():
            bx = anchor.x() - 12.0 - box_w
        by = anchor.y() - 0.5 * box_h
        bx = max(bounds.left(), min(bx, max(bounds.left(), bounds.right() - box_w)))
        by = max(bounds.top(), min(by, max(bounds.top(), bounds.bottom() - box_h)))
        box_rect = QtCore.QRectF(bx, by, box_w, box_h)

        painter.save()
        painter.setFont(font)
        painter.setBrush(QtGui.QColor(255, 255, 255, 245))
        painter.setPen(QtGui.QPen(QtGui.QColor("#94A3B8"), 1.1))
        painter.drawRoundedRect(box_rect, 5.0, 5.0)
        painter.setPen(QtGui.QColor("#1E293B"))
        for i, line in enumerate(lines):
            y_text = by + pad_y + i * line_h + fm.ascent()
            painter.drawText(QtCore.QPointF(bx + pad_x, y_text), line)
        painter.restore()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#FFFFFF"))
        if self._pixmap is not None and not self._pixmap.isNull():
            dpr_pix = max(1.0, float(self._pixmap.devicePixelRatio()))
            pix_w = float(self._pixmap.width()) / dpr_pix
            pix_h = float(self._pixmap.height()) / dpr_pix
            if abs(pix_w - float(self.width())) > 1.0 or abs(pix_h - float(self.height())) > 1.0:
                painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
                painter.drawPixmap(self.rect(), self._pixmap)
            else:
                painter.drawPixmap(0, 0, self._pixmap)
                if (
                    self._viewport_timer.isActive()
                    and self._rendered_ax_rect is not None
                    and self._rendered_xlim is not None
                    and self._rendered_ylim is not None
                ):
                    cx0, cx1 = (float(v) for v in self.ax.get_xlim())
                    cy0, cy1 = (float(v) for v in self.ax.get_ylim())
                    rx0, rx1 = self._rendered_xlim
                    ry0, ry1 = self._rendered_ylim
                    if (
                        abs(cx1 - cx0) > 1e-12
                        and abs(cy1 - cy0) > 1e-12
                        and ((cx0, cx1) != (rx0, rx1) or (cy0, cy1) != (ry0, ry1))
                    ):
                        ax_r = self._rendered_ax_rect
                        fx0 = (rx0 - cx0) / (cx1 - cx0)
                        fx1 = (rx1 - cx0) / (cx1 - cx0)
                        dst_x0 = ax_r.left() + min(fx0, fx1) * ax_r.width()
                        dst_x1 = ax_r.left() + max(fx0, fx1) * ax_r.width()
                        fy_top = (cy1 - ry1) / (cy1 - cy0)
                        fy_bot = (cy1 - ry0) / (cy1 - cy0)
                        dst_y0 = ax_r.top() + min(fy_top, fy_bot) * ax_r.height()
                        dst_y1 = ax_r.top() + max(fy_top, fy_bot) * ax_r.height()
                        dst_r = QtCore.QRectF(
                            dst_x0, dst_y0, max(1.0, dst_x1 - dst_x0), max(1.0, dst_y1 - dst_y0)
                        )
                        src_r = QtCore.QRectF(
                            ax_r.left() * dpr_pix,
                            ax_r.top() * dpr_pix,
                            ax_r.width() * dpr_pix,
                            ax_r.height() * dpr_pix,
                        )
                        painter.save()
                        painter.setClipRect(ax_r)
                        painter.fillRect(ax_r, QtGui.QColor("#FFFFFF"))
                        painter.drawPixmap(dst_r, self._pixmap, src_r)
                        painter.restore()
        if (
            self._interactive_scatter
            and self._press_pos is not None
            and self._drag_pos is not None
        ):
            if self._drag_mode == "lasso" and len(self._lasso_screen_pts) >= 2:
                path = QtGui.QPainterPath(self._lasso_screen_pts[0])
                for pt in self._lasso_screen_pts[1:]:
                    path.lineTo(pt)
                path.closeSubpath()
                painter.fillPath(path, QtGui.QColor(30, 64, 175, 34))
                pen = QtGui.QPen(QtGui.QColor("#1E40AF"), 1.6, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawPath(path)
            else:
                rect = QtCore.QRectF(self._press_pos, self._drag_pos).normalized()
                if rect.width() >= 5.0 and rect.height() >= 5.0:
                    if self._drag_mode == "zoom":
                        painter.fillRect(rect, QtGui.QColor(15, 118, 110, 32))
                        pen = QtGui.QPen(QtGui.QColor("#0F766E"), 1.6, Qt.PenStyle.DashLine)
                    else:
                        painter.fillRect(rect, QtGui.QColor(30, 64, 175, 34))
                        pen = QtGui.QPen(QtGui.QColor("#1E40AF"), 1.5, Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.drawRect(rect)
        if (
            self._interactive_scatter
            and self._hover_idx is not None
            and self._press_pos is None
            and not self._panning
            and 0 <= self._hover_idx < len(self._hover_labels)
        ):
            self._draw_hover_card(painter)
        painter.end()

    def _screen_to_data(self, pos: QtCore.QPointF) -> tuple[float, float] | None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        x_mpl = pos.x() * dpr
        y_mpl = (float(self.height()) - pos.y()) * dpr
        try:
            xd, yd = self.ax.transData.inverted().transform((x_mpl, y_mpl))
            return float(xd), float(yd)
        except Exception:
            return None

    def _emit_x_at_pos(self, pos: QtCore.QPointF) -> None:
        pt = self._screen_to_data(pos)
        if pt is not None:
            self.x_value_clicked.emit(pt[0])

    def _emit_y_at_pos(self, pos: QtCore.QPointF) -> None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        x_mpl = pos.x() * dpr
        y_mpl = (float(self.height()) - pos.y()) * dpr
        if not self.ax.bbox.contains(x_mpl, y_mpl):
            return
        pt = self._screen_to_data(pos)
        if pt is not None:
            self.y_value_clicked.emit(pt[1])

    def _nearest_visible_point(
        self, pos: QtCore.QPointF, *, flush_pending: bool = False
    ) -> int | None:
        if (
            self._pick_coords is None
            or self._pick_global_indices is None
            or len(self._pick_coords) == 0
        ):
            return None
        if self._render_pending or (
            flush_pending and (self._viewport_timer.isActive() or self._resize_timer.isActive())
        ):
            self._render_now()
        dpr = max(1.0, float(self.devicePixelRatioF()))
        x_mpl = pos.x() * dpr
        y_mpl = (float(self.height()) - pos.y()) * dpr
        pick_tol = 14.0 * dpr
        bbox = self.ax.bbox
        if not (
            float(bbox.x0) - pick_tol <= x_mpl <= float(bbox.x1) + pick_tol
            and float(bbox.y0) - pick_tol <= y_mpl <= float(bbox.y1) + pick_tol
        ):
            return None
        x0, x1 = sorted(float(v) for v in self.ax.get_xlim())
        y0, y1 = sorted(float(v) for v in self.ax.get_ylim())
        dx_pad = 0.02 * max(1e-6, x1 - x0)
        dy_pad = 0.02 * max(1e-6, y1 - y0)
        in_view = (
            (self._pick_coords[:, 0] >= x0 - dx_pad)
            & (self._pick_coords[:, 0] <= x1 + dx_pad)
            & (self._pick_coords[:, 1] >= y0 - dy_pad)
            & (self._pick_coords[:, 1] <= y1 + dy_pad)
        )
        in_idx = np.flatnonzero(in_view)
        if in_idx.size == 0:
            return None
        try:
            pts_px = self.ax.transData.transform(self._pick_coords[in_idx])
        except Exception:
            return None
        dists = np.hypot(pts_px[:, 0] - x_mpl, pts_px[:, 1] - y_mpl)
        best_sub = int(np.argmin(dists))
        if float(dists[best_sub]) <= pick_tol:
            return int(in_idx[best_sub])
        return None

    def _begin_pan(self, pos: QtCore.QPointF, button: Qt.MouseButton) -> None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        self._panning = True
        self._pan_button = button
        self._pan_press_pos = pos
        self._pan_drag_exceeded = False
        self._hover_idx = None
        self._hover_pos = None
        self._pan_start_mpl = (pos.x() * dpr, (float(self.height()) - pos.y()) * dpr)
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self._pan_start_xlim = (float(x0), float(x1))
        self._pan_start_ylim = (float(y0), float(y1))
        bbox = self.ax.bbox
        self._pan_bbox_wh = (
            max(1.0, float(bbox.width)),
            max(1.0, float(bbox.height)),
        )

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self._interactive_scatter:
            super().wheelEvent(event)
            return
        dy = float(event.angleDelta().y())
        if abs(dy) < 1e-3:
            dy = float(event.pixelDelta().y()) * 2.0
        if abs(dy) < 1e-3:
            super().wheelEvent(event)
            return
        steps = max(-3.0, min(3.0, dy / 120.0))
        scale = float(0.84**steps)
        pt = self._screen_to_data(event.position())
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        if pt is None:
            xa = 0.5 * (float(x0) + float(x1))
            ya = 0.5 * (float(y0) + float(y1))
        else:
            xmin, xmax = sorted((float(x0), float(x1)))
            ymin, ymax = sorted((float(y0), float(y1)))
            xa = min(max(pt[0], xmin), xmax)
            ya = min(max(pt[1], ymin), ymax)
        self.wheel_zoomed.emit(xa, ya, scale)
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._interactive_scatter
            and event.button() == Qt.MouseButton.LeftButton
            and self._nearest_visible_point(event.position(), flush_pending=True) is None
        ):
            self._press_pos = None
            self._drag_pos = None
            self._lasso_screen_pts.clear()
            self.reset_zoom_requested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        btn = event.button()
        if self._interactive_scatter:
            if btn in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or (
                btn == Qt.MouseButton.LeftButton
                and bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            ):
                self._begin_pan(event.position(), btn)
                event.accept()
                return
            if btn == Qt.MouseButton.LeftButton:
                self._press_pos = event.position()
                self._drag_pos = event.position()
                self._lasso_screen_pts = [event.position()]
                self._hover_idx = None
                self._hover_pos = None
        elif self._interactive_x_scrub and btn == Qt.MouseButton.LeftButton:
            self._scrubbing_x = True
            self._emit_x_at_pos(event.position())
        elif self._interactive_y_pick and btn == Qt.MouseButton.LeftButton:
            self._emit_y_at_pos(event.position())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._interactive_scatter:
            if self._panning:
                if self._pan_button == Qt.MouseButton.RightButton and not self._pan_drag_exceeded:
                    p0 = self._pan_press_pos or event.position()
                    if (
                        abs(event.position().x() - p0.x()) < 5.0
                        and abs(event.position().y() - p0.y()) < 5.0
                    ):
                        event.accept()
                        return
                self._pan_drag_exceeded = True
                dpr = max(1.0, float(self.devicePixelRatioF()))
                x_mpl = event.position().x() * dpr
                y_mpl = (float(self.height()) - event.position().y()) * dpr
                dx_mpl = x_mpl - self._pan_start_mpl[0]
                dy_mpl = y_mpl - self._pan_start_mpl[1]
                x0, x1 = self._pan_start_xlim
                y0, y1 = self._pan_start_ylim
                w_px, h_px = self._pan_bbox_wh
                dx_data = -dx_mpl * (x1 - x0) / w_px
                dy_data = -dy_mpl * (y1 - y0) / h_px
                self.pan_dragged.emit(x0 + dx_data, x1 + dx_data, y0 + dy_data, y1 + dy_data)
                event.accept()
                return
            if self._press_pos is not None:
                cur = event.position()
                self._drag_pos = cur
                if self._drag_mode == "lasso":
                    last_pt = self._lasso_screen_pts[-1]
                    if abs(cur.x() - last_pt.x()) + abs(cur.y() - last_pt.y()) >= 3.0:
                        self._lasso_screen_pts.append(cur)
                self.update()
            elif self._hover_labels:
                local_idx = self._nearest_visible_point(event.position())
                new_idx = (
                    local_idx
                    if (local_idx is not None and local_idx < len(self._hover_labels))
                    else None
                )
                if new_idx != self._hover_idx:
                    self._hover_idx = new_idx
                    self._hover_pos = event.position() if new_idx is not None else None
                    self.update()
        elif self._interactive_x_scrub and self._scrubbing_x:
            self._emit_x_at_pos(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        btn = event.button()
        if self._interactive_scatter and self._panning and btn == self._pan_button:
            end_pos = event.position()
            was_right = btn == Qt.MouseButton.RightButton
            exceeded = self._pan_drag_exceeded
            self._panning = False
            self._pan_button = None
            self._pan_press_pos = None
            self._pan_drag_exceeded = False
            if was_right and not exceeded:
                local_idx = self._nearest_visible_point(end_pos, flush_pending=True)
                hit_gi = (
                    int(self._pick_global_indices[local_idx])
                    if (local_idx is not None and self._pick_global_indices is not None)
                    else -1
                )
                self.context_menu_requested.emit(event.globalPosition().toPoint(), hit_gi)
            elif exceeded and self._viewport_timer.isActive():
                self._render_now()
            event.accept()
            return

        if btn == Qt.MouseButton.LeftButton:
            if self._interactive_x_scrub and self._scrubbing_x:
                self._scrubbing_x = False
            if self._interactive_scatter and self._press_pos is not None:
                end_pos = event.position()
                dx = abs(end_pos.x() - self._press_pos.x())
                dy = abs(end_pos.y() - self._press_pos.y())
                p0 = self._press_pos
                lasso_pts = list(self._lasso_screen_pts)
                self._press_pos = None
                self._drag_pos = None
                self._lasso_screen_pts.clear()
                self.update()

                shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                dpr = max(1.0, float(self.devicePixelRatioF()))
                h_log = float(self.height())
                if dx < 6.0 and dy < 6.0:
                    local_idx = self._nearest_visible_point(end_pos, flush_pending=True)
                    if local_idx is not None and self._pick_global_indices is not None:
                        self.point_clicked.emit(int(self._pick_global_indices[local_idx]))
                elif self._drag_mode == "lasso" and len(lasso_pts) >= 3:
                    if (
                        self._render_pending
                        or self._viewport_timer.isActive()
                        or self._resize_timer.isActive()
                    ):
                        self._render_now()
                    try:
                        inv = self.ax.transData.inverted()
                        mpl_pts = np.array(
                            [(pt.x() * dpr, (h_log - pt.y()) * dpr) for pt in lasso_pts],
                            dtype=np.float64,
                        )
                        poly_xy = inv.transform(mpl_pts)
                        self.lasso_selected.emit(poly_xy, shift_held)
                    except Exception:
                        logger.debug("Could not invert scatter axes transform for lasso select")
                else:
                    if (
                        self._render_pending
                        or self._viewport_timer.isActive()
                        or self._resize_timer.isActive()
                    ):
                        self._render_now()
                    x0_mpl = min(p0.x(), end_pos.x()) * dpr
                    x1_mpl = max(p0.x(), end_pos.x()) * dpr
                    y0_mpl = (h_log - max(p0.y(), end_pos.y())) * dpr
                    y1_mpl = (h_log - min(p0.y(), end_pos.y())) * dpr
                    try:
                        inv = self.ax.transData.inverted()
                        d0 = inv.transform((x0_mpl, y0_mpl))
                        d1 = inv.transform((x1_mpl, y1_mpl))
                        xmin, xmax = sorted([float(d0[0]), float(d1[0])])
                        ymin, ymax = sorted([float(d0[1]), float(d1[1])])
                        if self._drag_mode == "zoom":
                            self.zoom_box_selected.emit(xmin, xmax, ymin, ymax)
                        else:
                            self.box_selected.emit(xmin, xmax, ymin, ymax, shift_held)
                    except Exception:
                        logger.debug("Could not invert scatter axes transform for box/zoom select")
        super().mouseReleaseEvent(event)


class BaleDialog(QtWidgets.QDialog):
    """Simplified, fast non-modal companion window for triaging labels & predictions in JABS."""

    def __init__(
        self,
        project: Project,
        central_widget: CentralWidget | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)
        self.setWindowTitle("BALE - Bout & Label Explorer")
        self.setMinimumSize(640, 480)

        self._project = project
        self._central_widget = central_widget
        self._report: BaleReport | None = None
        self._report_includes_predictions: bool = False

        # Core workflow state
        self._mode: ExplorerWorkflowMode = "audit"
        self._queue_filter: QueueFilterMode = "needs_review"
        self._class_filter: str = "all"
        self._inspector_view: InspectorViewMode = "prob"
        self._subtypes_scope: SubtypeScope = "pos_only"
        self._subtypes_lens_ui: str = "intrinsic"
        self._subtypes_forced_k: int | None = None
        self._selected_bout_idx: int | None = None
        self._last_card_bout_idx: int | None = None
        self._scrubbed_frame: int | None = None
        self._lasso_selected_indices: list[int] = []
        self._custom_selection_label: str = ""
        self._custom_selection_preserve_order: bool = False
        self._scatter_default_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._scatter_view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._map_expanded: bool = False
        self._saved_upper_sizes: list[int] | None = None
        self._saved_main_sizes: list[int] | None = None
        self._repr_matrix_cache: tuple[int, str, np.ndarray] | None = None
        self._identity_label_cache: dict[tuple[str, int], str] = {}
        self._video_ext_id_cache: dict[str, list[str] | None] = {}

        # Fast in-place triage state (< 25 ms per edit without cold 5-fold CV reload)
        self._examined_store = ExaminedBoutStore(Path(project.project_paths.project_dir))
        self._undo_stack = BaleUndoStack()
        self._ui_undo_history: list[dict[str, Any]] = []
        self._reviewed_bout_ids: set[int] = set()
        self._examined_only_bout_ids: set[int] = set()
        self._pinned_needs_review_bout_ids: set[int] = set()
        self._deleted_bout_ids: set[int] = set()
        self._unlabeled_bout_ids: set[int] = set()
        self._saved_notes_by_bout: dict[int, str] = {}
        self._orig_boundary_flags: dict[int, tuple[bool, bool]] = {}
        self._table_sort_col: int | None = None
        self._table_sort_order: Qt.SortOrder = Qt.SortOrder.DescendingOrder
        self._reselect_bout_key_on_finish: tuple[str, int, int, int] | None = None
        self._pending_recalc_count: int = 0
        self._external_labels_dirty: bool = False
        self._ignoring_self_label_signal: bool = False
        self._ignoring_self_seek_signal: bool = False
        self._central_widget_connections: list[tuple[Any, Any]] = []
        self._table_mouse_press_col: int | None = None
        self._table_mouse_press_btn: Qt.MouseButton | None = None
        self._bout_seek_epoch: int = 0
        self._video_seek_epoch: int = 0

        # Advanced overrides (tucked inside Options menu)
        self._proj_method: ProjectionMethod = "pca"
        self._last_feature_proj: Literal["pca", "mds", "tsne"] = "pca"
        self._proj_menu_actions: dict[str, QtGui.QAction] = {}
        self._space_override: Literal["auto", "classifier", "unsupervised", "rf_proximity"] = (
            "auto"
        )
        self._color_override: Literal[
            "auto", "class", "confusion", "cluster", "identity", "video"
        ] = "auto"
        self._max_pred_bouts: int = 500

        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._active_worker: _ExplorerComputeTask | None = None
        self._spatial_worker: _SpatialComputeTask | None = None
        self._spatial_request_id: int = 0
        self._queued_refresh: bool = False
        self._is_closing: bool = False
        self._suppress_auto_seek: bool = False
        self._pending_click_already_sought_row: int | None = None
        self._pending_bring_main_window_to_front: bool = False
        self._standalone_help_dialog: UserGuideDialog | None = None
        self._initial_splitters_applied: bool = False

        # Whole-video screener (`4. Videos`) state — zero Parquet I/O, lazy-cached per scope
        self._video_report: VideoScreenerReport | None = None
        self._video_report_key: tuple[str, bool] | None = None
        self._video_worker: _VideoScreenerTask | None = None
        self._queued_video_refresh: bool = False
        self._video_request_id: int = 0
        self._analysis_needed_on_bout_mode: bool = False
        self._video_source: VideoMetricSource = "predictions"
        self._video_preset: VideoScreeningPreset = "all"
        self._video_unlabeled_only: bool = False
        self._video_lasso_selected: set[str] = set()
        self._selected_video_name: str | None = None
        self._video_scrubbed_frame: int | None = None
        self._video_table_sort_col: int = 2  # Default: Frames (%) descending
        self._video_table_sort_order: Qt.SortOrder = Qt.SortOrder.DescendingOrder
        self._video_scatter_coords: np.ndarray | None = None
        self._video_scatter_names: list[str] = []
        self._video_scatter_sel_artist: Any = None
        self._bout_scatter_sel_artist: Any = None
        self._bout_scatter_vis_coords: np.ndarray | None = None
        self._bout_scatter_vis_map: dict[int, int] = {}
        self._video_table_mouse_press_btn: Qt.MouseButton | None = None
        self._pending_video_click_already_sought_row: int | None = None
        self._pending_select_video_after_refresh: str | None = None

        self._build_ui()
        self._apply_initial_window_geometry()
        self._populate_behaviors()
        self._wire_central_widget_signals()
        self._scatter_redraw_timer = QtCore.QTimer(self)
        self._scatter_redraw_timer.setSingleShot(True)
        self._scatter_redraw_timer.setInterval(45)
        self._scatter_redraw_timer.timeout.connect(self._draw_scatter)
        self._bout_search_timer = QtCore.QTimer(self)
        self._bout_search_timer.setSingleShot(True)
        self._bout_search_timer.setInterval(120)
        self._bout_search_timer.timeout.connect(self._on_bout_search_applied)
        self._video_search_timer = QtCore.QTimer(self)
        self._video_search_timer.setSingleShot(True)
        self._video_search_timer.setInterval(120)
        self._video_search_timer.timeout.connect(self._on_video_search_applied)
        self._init_refresh_timer = QtCore.QTimer(self)
        self._init_refresh_timer.setSingleShot(True)
        self._init_refresh_timer.timeout.connect(self._on_initial_refresh_timeout)
        self._init_refresh_timer.start(60)

    def _apply_default_splitter_sizes(
        self, total_w: int | None = None, total_h: int | None = None
    ) -> None:
        """Apply default 2/3 : 1/3 horizontal (2D Plot vs Inspector) and 60% : 40% vertical splits."""
        avail_w = max(420, int((total_w if total_w is not None else self.width()) - 24))
        left_w = round((2.0 / 3.0) * avail_w)
        right_w = max(260, avail_w - left_w)
        left_w = max(180, avail_w - right_w)
        self._upper_splitter.setSizes([left_w, right_w])

        avail_h = max(360, int((total_h if total_h is not None else self.height()) - 96))
        top_h = round(0.60 * avail_h)
        bot_h = max(140, avail_h - top_h)
        self._main_splitter.setSizes([top_h, bot_h])

    def _apply_initial_window_geometry(self) -> None:
        """Size dialog comfortably within the active screen so all edges stay resizable on every OS."""
        target_w, target_h = 1340, 820
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            if avail.width() > 0 and avail.height() > 0:
                target_w = max(self.minimumWidth(), min(target_w, avail.width() - 40))
                target_h = max(self.minimumHeight(), min(target_h, avail.height() - 60))
        self.resize(target_w, target_h)
        self._apply_default_splitter_sizes(target_w, target_h)

    # ------------------------------------------------------------------
    # UI Construction (Clean, Minimalist, Progressive Disclosure)
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background-color: #FFFFFF; color: #0F172A; }
            QPushButton#modePill {
                background-color: #F1F5F9;
                color: #334155;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 5px 13px;
                font-weight: 600;
            }
            QPushButton#modePill:checked {
                background-color: #1E40AF;
                color: #FFFFFF;
                border-color: #1E40AF;
            }
            QPushButton#subPill {
                background-color: #F8FAFC;
                color: #475569;
                border: 1px solid #CBD5E1;
                border-radius: 5px;
                padding: 3px 10px;
                font-weight: 600;
            }
            QPushButton#subPill:checked {
                background-color: #0F172A;
                color: #FFFFFF;
                border-color: #0F172A;
            }
            """
        )

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(6)

        # 1. Clean Top Bar (Responsive Wrapping FlowLayout: Behavior + 3-Mode Pills + CV + Undo/Recalc + Options)
        top_bar = QtWidgets.QFrame()
        top_bar.setStyleSheet(
            "QFrame { background-color: #F8FAFC; border-bottom: 1px solid #E2E8F0; border-radius: 6px; }"
        )
        top_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        tb = _FlowLayout(top_bar, h_spacing=7, v_spacing=5)
        tb.setContentsMargins(10, 6, 10, 6)

        beh_lbl = QtWidgets.QLabel("Behavior:")
        beh_lbl.setStyleSheet("font-weight: 700; color: #1E293B; border: none;")
        tb.addWidget(beh_lbl)

        self.behavior_combo = QtWidgets.QComboBox()
        self.behavior_combo.setMinimumWidth(110)
        self.behavior_combo.setMaximumWidth(240)
        self.behavior_combo.setMinimumContentsLength(10)
        self.behavior_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.behavior_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.behavior_combo.currentIndexChanged.connect(self._on_behavior_combo_changed)
        tb.addWidget(self.behavior_combo)

        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_audit_btn = QtWidgets.QPushButton("1. Audit Labels")
        self.mode_audit_btn.setObjectName("modePill")
        self.mode_audit_btn.setCheckable(True)
        self.mode_audit_btn.setChecked(True)
        self.mode_audit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_audit_btn.setToolTip(
            "Find human labels that look mislabeled or have sloppy start/end boundaries"
        )

        self.mode_predict_btn = QtWidgets.QPushButton("2. Review Predictions")
        self.mode_predict_btn.setObjectName("modePill")
        self.mode_predict_btn.setCheckable(True)
        self.mode_predict_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_predict_btn.setToolTip(
            "Inspect classifier predictions and turn false-positives or uncertain bouts into training labels"
        )

        self.mode_subtypes_btn = QtWidgets.QPushButton("3. Sub-Types")
        self.mode_subtypes_btn.setObjectName("modePill")
        self.mode_subtypes_btn.setCheckable(True)
        self.mode_subtypes_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_subtypes_btn.setToolTip(
            "Explore unsupervised behavioral sub-clusters and kinematic outliers"
        )

        self.mode_videos_btn = QtWidgets.QPushButton("4. Videos")
        self.mode_videos_btn.setObjectName("modePill")
        self.mode_videos_btn.setCheckable(True)
        self.mode_videos_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_videos_btn.setToolTip(
            "Screen whole videos by total behavior frames, bout counts, zero-behavior videos, "
            "and 50k-style outlier detection criteria"
        )

        self.mode_group.addButton(self.mode_audit_btn, 0)
        self.mode_group.addButton(self.mode_predict_btn, 1)
        self.mode_group.addButton(self.mode_subtypes_btn, 2)
        self.mode_group.addButton(self.mode_videos_btn, 3)
        self.mode_group.idClicked.connect(self._on_mode_pill_clicked)

        tb.addWidget(self.mode_audit_btn)
        tb.addWidget(self.mode_predict_btn)
        tb.addWidget(self.mode_subtypes_btn)
        tb.addWidget(self.mode_videos_btn)

        self.cv_badge = QtWidgets.QLabel("CV Tracks: --")
        self.cv_badge.setStyleSheet(
            "padding: 4px 9px; border-radius: 5px; background-color: #E2E8F0; "
            "color: #1E293B; font-weight: 700;"
        )
        tb.addWidget(self.cv_badge)

        self.undo_btn = QtWidgets.QPushButton("↶ Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.undo_btn.setToolTip("Undo last triage step or label change (⌘Z / Ctrl+Z)")
        self.undo_btn.clicked.connect(self._on_undo_clicked)
        tb.addWidget(self.undo_btn)

        self.recalc_btn = QtWidgets.QPushButton("↻ Recalculate")
        self.recalc_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.recalc_btn.setToolTip(
            "Re-run cross-validation diagnostic scoring & 2D map from current disk labels [R]"
        )
        self.recalc_btn.clicked.connect(self._on_recalc_btn_clicked)
        tb.addWidget(self.recalc_btn)

        self.options_btn = QtWidgets.QToolButton()
        self.options_btn.setText("Options ▾")
        self.options_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.options_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.options_btn.setStyleSheet(
            "QToolButton { padding: 4px 10px; border: 1px solid #CBD5E1; "
            "border-radius: 5px; background-color: #FFFFFF; font-weight: 600; }"
            "QToolButton::menu-indicator { image: none; }"
        )
        self._build_options_menu()
        tb.addWidget(self.options_btn)

        self.help_btn = QtWidgets.QPushButton("? Help")
        self.help_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.help_btn.setStyleSheet(
            "QPushButton { padding: 4px 10px; border: 1px solid #CBD5E1; "
            "border-radius: 5px; background-color: #FFFFFF; font-weight: 600; color: #1E40AF; }"
            "QPushButton:hover { background-color: #EFF6FF; border-color: #93C5FD; }"
        )
        self.help_btn.setToolTip("Open the BALE (Bout & Label Explorer) User Guide")
        self.help_btn.clicked.connect(self._open_help_guide)
        tb.addWidget(self.help_btn)

        root_layout.addWidget(top_bar)

        # 1b. Slim Progressive-Disclosure Sub-Types Bar (Visible ONLY in 3. Sub-Types mode)
        self.subtypes_bar = QtWidgets.QFrame()
        self.subtypes_bar.setStyleSheet(
            "QFrame { background-color: #F1F5F9; border: none; border-radius: 5px; }"
        )
        self.subtypes_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        stb = _FlowLayout(self.subtypes_bar, h_spacing=7, v_spacing=4)
        stb.setContentsMargins(10, 4, 10, 4)

        scope_lbl = QtWidgets.QLabel("Scope:")
        scope_lbl.setStyleSheet("font-weight: 700; color: #1E293B; border: none;")
        stb.addWidget(scope_lbl)

        self.subtypes_scope_combo = QtWidgets.QComboBox()
        self.subtypes_scope_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.subtypes_scope_combo.setMinimumContentsLength(10)
        self.subtypes_scope_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.subtypes_scope_combo.addItem("Positive Only", "pos_only")
        self.subtypes_scope_combo.addItem("All Labeled (Pos + Neg)", "all_labeled")
        self.subtypes_scope_combo.setToolTip(
            "Cluster only positive target bouts (default) or include Not-Behavior bouts"
        )
        self.subtypes_scope_combo.currentIndexChanged.connect(self._on_subtypes_controls_changed)
        stb.addWidget(self.subtypes_scope_combo)

        lens_lbl = QtWidgets.QLabel("Lens:")
        lens_lbl.setStyleSheet("font-weight: 700; color: #1E293B; border: none;")
        stb.addWidget(lens_lbl)

        self.subtypes_lens_combo = QtWidgets.QComboBox()
        self.subtypes_lens_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.subtypes_lens_combo.setMinimumContentsLength(12)
        self.subtypes_lens_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.subtypes_lens_combo.addItem("Intrinsic Kinematics", "intrinsic")
        self.subtypes_lens_combo.addItem("Locomotion & Turn", "locomotion")
        self.subtypes_lens_combo.addItem("Core Body Posture", "posture")
        self.subtypes_lens_combo.addItem("Social & Huddle", "social")
        self.subtypes_lens_combo.addItem("Classifier-Weighted", "classifier_weighted")
        self.subtypes_lens_combo.addItem("RF Leaf Proximity", "rf_proximity")
        self.subtypes_lens_combo.setToolTip(
            "Choose which ethological feature lens defines behavioral sub-types"
        )
        self.subtypes_lens_combo.currentIndexChanged.connect(self._on_subtypes_lens_combo_changed)
        stb.addWidget(self.subtypes_lens_combo)

        k_lbl = QtWidgets.QLabel("Clusters:")
        k_lbl.setStyleSheet("font-weight: 700; color: #1E293B; border: none;")
        stb.addWidget(k_lbl)

        self.subtypes_k_combo = QtWidgets.QComboBox()
        self.subtypes_k_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.subtypes_k_combo.setMinimumContentsLength(7)
        self.subtypes_k_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.subtypes_k_combo.addItem("Auto (2\u20136)", None)
        for k_val in range(2, 7):
            self.subtypes_k_combo.addItem(f"k = {k_val}", k_val)
        self.subtypes_k_combo.setToolTip(
            "Override Ward.D2 sub-type cluster count k or auto-select via Silhouette + Elbow"
        )
        self.subtypes_k_combo.currentIndexChanged.connect(self._on_subtypes_controls_changed)
        stb.addWidget(self.subtypes_k_combo)

        self.subtypes_summary_lbl = _ElidedLabel("")
        self.subtypes_summary_lbl.setStyleSheet("color: #334155; font-size: 11.5px; border: none;")
        stb.addWidget(self.subtypes_summary_lbl)

        self.cage_skew_badge = QtWidgets.QLabel("")
        self.cage_skew_badge.setStyleSheet(
            "background-color: #FEF3C7; color: #92400E; border: none; "
            "border-radius: 4px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
        )
        self.cage_skew_badge.setVisible(False)
        stb.addWidget(self.cage_skew_badge)

        self.subtypes_bar.setVisible(False)
        root_layout.addWidget(self.subtypes_bar)

        # 1c. Slim Progressive-Disclosure Whole-Video Screener Bar (Visible ONLY in 4. Videos mode)
        self.videos_bar = QtWidgets.QFrame()
        self.videos_bar.setStyleSheet(
            "QFrame { background-color: #F1F5F9; border: none; border-radius: 5px; }"
        )
        self.videos_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        vtb = _FlowLayout(self.videos_bar, h_spacing=7, v_spacing=4)
        vtb.setContentsMargins(10, 4, 10, 4)

        vsrc_lbl = QtWidgets.QLabel("Source:")
        vsrc_lbl.setStyleSheet("font-weight: 700; color: #1E293B; border: none;")
        vtb.addWidget(vsrc_lbl)

        self.videos_source_combo = QtWidgets.QComboBox()
        self.videos_source_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.videos_source_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.videos_source_combo.setMaximumWidth(145)
        self.videos_source_combo.addItem("Predictions", "predictions")
        self.videos_source_combo.addItem("Human Labels", "labels")
        self.videos_source_combo.setToolTip(
            "Choose whether whole-video counts & 50k screening use classifier predictions or human labels"
        )
        self.videos_source_combo.currentIndexChanged.connect(self._on_video_source_changed)
        vtb.addWidget(self.videos_source_combo)

        vpre_lbl = QtWidgets.QLabel("Filter:")
        vpre_lbl.setStyleSheet("font-weight: 700; color: #1E293B; border: none;")
        vtb.addWidget(vpre_lbl)

        self.videos_preset_combo = QtWidgets.QComboBox()
        self.videos_preset_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.videos_preset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.videos_preset_combo.setMaximumWidth(210)
        for p_lbl, p_key in (
            ("All Videos", "all"),
            ("50k Outliers (E1\u2013E4)", "any_outlier"),
            ("E1 \u2022 Most Frames (%)", "high_frames"),
            ("E1 \u2022 Most Bouts", "high_bouts"),
            ("E2 \u2022 Longest Bout", "long_bout"),
            ("E3a \u2022 Low-Conf Beh", "low_conf_pos"),
            ("E3b \u2022 Near-Miss None", "low_conf_none"),
            ("E4 \u2022 Fragile / Disagree", "fragile"),
            ("E5 \u2022 Zero Behavior", "zero_all"),
            ("E5 \u2022 Zero (High-Conf)", "zero_high_conf"),
            ("Unlabeled w/ Behavior", "unlabeled_with_behavior"),
            ("Labeled Only", "labeled_only"),
        ):
            self.videos_preset_combo.addItem(p_lbl, p_key)
        self.videos_preset_combo.setToolTip(
            "Select videos using the same 50k screening outlier buckets (E1\u2013E5), "
            "high frames/bouts, or zero-behavior filters"
        )
        self.videos_preset_combo.currentIndexChanged.connect(self._on_video_preset_changed)
        vtb.addWidget(self.videos_preset_combo)

        self.videos_unlabeled_btn = QtWidgets.QPushButton("Unlabeled Only")
        self.videos_unlabeled_btn.setObjectName("subPill")
        self.videos_unlabeled_btn.setCheckable(True)
        self.videos_unlabeled_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.videos_unlabeled_btn.setToolTip(
            "Restrict the video list to videos that have no human annotations yet"
        )
        self.videos_unlabeled_btn.toggled.connect(self._on_video_unlabeled_toggled)
        vtb.addWidget(self.videos_unlabeled_btn)

        self.videos_summary_lbl = _ElidedLabel("")
        self.videos_summary_lbl.setStyleSheet("color: #334155; font-size: 11.5px; border: none;")
        vtb.addWidget(self.videos_summary_lbl)

        self.videos_bar.setVisible(False)
        root_layout.addWidget(self.videos_bar)

        # 2. Main Vertical Splitter: [Upper: 2D Map + Selected Bout Triage Card] / [Lower: Sortable 7-Col Queue]
        main_splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self._main_splitter = main_splitter

        upper_splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self._upper_splitter = upper_splitter

        # Left Pane: Clean 2D Map with Compact Wrapping Selection, Zoom & Expand Strip
        scatter_panel = QtWidgets.QFrame()
        scatter_panel.setMinimumWidth(180)
        scatter_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        scatter_panel.setStyleSheet("QFrame { background-color: #FFFFFF; border: none; }")
        scatter_vbox = QtWidgets.QVBoxLayout(scatter_panel)
        scatter_vbox.setContentsMargins(0, 0, 4, 0)
        scatter_vbox.setSpacing(3)

        map_bar_widget = QtWidgets.QWidget()
        map_bar_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        map_bar = _FlowLayout(map_bar_widget, h_spacing=3, v_spacing=3)
        map_bar.setContentsMargins(2, 0, 2, 0)

        self.arena_map_btn = QtWidgets.QPushButton("🗺 Arena [A]")
        self.arena_map_btn.setCheckable(True)
        self.arena_map_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.arena_map_btn.setStyleSheet(
            "QPushButton { padding: 2px 6px; border: 1px solid #CBD5E1; border-radius: 4px; "
            "background-color: #F8FAFC; color: #0F172A; font-weight: 600; font-size: 11.0px; }"
            "QPushButton:checked { background-color: #CCFBF1; color: #0F766E; border-color: #5EEAD4; font-weight: 700; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
        )
        self.arena_map_btn.setToolTip(
            "Toggle Arena Spatial Map showing where bouts happen inside the cage [A]"
        )
        self.arena_map_btn.clicked.connect(self._toggle_arena_map)
        map_bar.addWidget(self.arena_map_btn)

        self.map_class_combo = QtWidgets.QComboBox()
        self.map_class_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.map_class_combo.setMinimumContentsLength(10)
        self.map_class_combo.setMaximumWidth(185)
        self.map_class_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.map_class_combo.addItem("All Classes", "all")
        self.map_class_combo.setToolTip(
            "Filter 2D plot & bout table by Positive, Negative, or specific behavior class"
        )
        self.map_class_combo.currentIndexChanged.connect(self._on_class_filter_combo_changed)
        map_bar.addWidget(self.map_class_combo)

        self.drag_mode_group = QtWidgets.QButtonGroup(self)
        self.drag_mode_group.setExclusive(True)

        self.drag_box_btn = QtWidgets.QPushButton("⬚ Box")
        self.drag_box_btn.setObjectName("subPill")
        self.drag_box_btn.setCheckable(True)
        self.drag_box_btn.setChecked(True)
        self.drag_box_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.drag_box_btn.setToolTip(
            "Drag a rectangle to select bouts (hold Shift to add another pocket)"
        )

        self.drag_lasso_btn = QtWidgets.QPushButton("✎ Lasso")
        self.drag_lasso_btn.setObjectName("subPill")
        self.drag_lasso_btn.setCheckable(True)
        self.drag_lasso_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.drag_lasso_btn.setToolTip(
            "Freehand draw around curved pockets or clusters (hold Shift to add more)"
        )

        self.drag_zoom_btn = QtWidgets.QPushButton("🔍 Zoom")
        self.drag_zoom_btn.setObjectName("subPill")
        self.drag_zoom_btn.setCheckable(True)
        self.drag_zoom_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.drag_zoom_btn.setToolTip(
            "Drag a rectangle to zoom into dense pockets (or scroll wheel / right-drag pan)"
        )

        self.drag_mode_group.addButton(self.drag_box_btn, 0)
        self.drag_mode_group.addButton(self.drag_lasso_btn, 1)
        self.drag_mode_group.addButton(self.drag_zoom_btn, 2)
        self.drag_mode_group.idClicked.connect(self._on_drag_mode_pill_clicked)

        map_bar.addWidget(self.drag_box_btn)
        map_bar.addWidget(self.drag_lasso_btn)
        map_bar.addWidget(self.drag_zoom_btn)

        _zoom_btn_css = (
            "QPushButton { padding: 2px 7px; border: 1px solid #CBD5E1; border-radius: 4px; "
            "background-color: #FFFFFF; color: #334155; font-weight: 700; font-size: 11.5px; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
            "QPushButton:disabled { color: #CBD5E1; border-color: #E2E8F0; }"
        )
        self.map_zoom_out_btn = QtWidgets.QPushButton("\u2212")
        self.map_zoom_out_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.map_zoom_out_btn.setStyleSheet(_zoom_btn_css)
        self.map_zoom_out_btn.setToolTip("Zoom out on 2D map [-]")
        self.map_zoom_out_btn.clicked.connect(lambda: self._zoom_scatter_by_step(1.28))

        self.map_zoom_in_btn = QtWidgets.QPushButton("+")
        self.map_zoom_in_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.map_zoom_in_btn.setStyleSheet(_zoom_btn_css)
        self.map_zoom_in_btn.setToolTip("Zoom in on 2D map [+]")
        self.map_zoom_in_btn.clicked.connect(lambda: self._zoom_scatter_by_step(0.76))

        self.map_reset_zoom_btn = QtWidgets.QPushButton("⟲ Reset [0]")
        self.map_reset_zoom_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.map_reset_zoom_btn.setEnabled(False)
        self.map_reset_zoom_btn.setStyleSheet(_zoom_btn_css)
        self.map_reset_zoom_btn.setToolTip(
            "Reset 2D map zoom to full view [0 or double-click empty background]"
        )
        self.map_reset_zoom_btn.clicked.connect(self._reset_scatter_zoom)

        self.expand_map_btn = QtWidgets.QPushButton("⤢ Expand [M]")
        self.expand_map_btn.setCheckable(True)
        self.expand_map_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.expand_map_btn.setStyleSheet(
            "QPushButton { padding: 2px 8px; border: 1px solid #CBD5E1; border-radius: 4px; "
            "background-color: #F8FAFC; color: #0F172A; font-weight: 600; font-size: 11.5px; }"
            "QPushButton:checked { background-color: #DBEAFE; color: #1E40AF; border-color: #93C5FD; font-weight: 700; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
        )
        self.expand_map_btn.setToolTip(
            "Expand or restore the 2D scatter map to inspect dense clusters [M]"
        )
        self.expand_map_btn.clicked.connect(self._toggle_expand_map)

        map_bar.addWidget(self.map_zoom_out_btn)
        map_bar.addWidget(self.map_zoom_in_btn)
        map_bar.addWidget(self.map_reset_zoom_btn)
        map_bar.addWidget(self.expand_map_btn)
        scatter_vbox.addWidget(map_bar_widget)

        self.scatter_fig = Figure(figsize=(6.4, 4.2), dpi=100, facecolor="#FFFFFF")
        self.scatter_ax = self.scatter_fig.add_subplot(111)
        self.scatter_canvas = _InteractiveAggCanvas(
            self.scatter_fig,
            self.scatter_ax,
            interactive_scatter=True,
            parent=scatter_panel,
        )
        self.scatter_canvas.point_clicked.connect(self._on_scatter_point_clicked)
        self.scatter_canvas.box_selected.connect(self._on_scatter_box_selected)
        self.scatter_canvas.lasso_selected.connect(self._on_scatter_lasso_selected)
        self.scatter_canvas.zoom_box_selected.connect(self._on_scatter_zoom_box)
        self.scatter_canvas.wheel_zoomed.connect(self._on_scatter_wheel_zoom)
        self.scatter_canvas.pan_dragged.connect(self._on_scatter_pan_dragged)
        self.scatter_canvas.reset_zoom_requested.connect(self._reset_scatter_zoom)
        self.scatter_canvas.context_menu_requested.connect(self._on_scatter_context_menu)
        scatter_vbox.addWidget(self.scatter_canvas, 1)
        upper_splitter.addWidget(scatter_panel)

        # Right Pane: Unified "Selected Bout Triage Card" (Zero nested bordered boxes!)
        card_panel = QtWidgets.QFrame()
        self._card_panel = card_panel
        card_panel.setMinimumWidth(260)
        card_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        card_panel.setStyleSheet("QFrame { background-color: #F8FAFC; border-radius: 6px; }")
        card_vbox = QtWidgets.QVBoxLayout(card_panel)
        card_vbox.setContentsMargins(10, 8, 10, 6)
        card_vbox.setSpacing(5)

        # Card Row 1: Elided Bout Title + Metadata + Play Clip Button
        hdr_row = QtWidgets.QHBoxLayout()
        hdr_row.setSpacing(8)
        hdr_text_col = QtWidgets.QVBoxLayout()
        hdr_text_col.setSpacing(2)
        self.card_title_lbl = _ElidedLabel("Select a bout to inspect")
        self.card_title_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #0F172A;")
        self.card_meta_lbl = _ElidedLabel("Click any row below or dot on the 2D map")
        self.card_meta_lbl.setStyleSheet("font-size: 12px; color: #475569;")
        hdr_text_col.addWidget(self.card_title_lbl)
        hdr_text_col.addWidget(self.card_meta_lbl)
        hdr_row.addLayout(hdr_text_col, 1)

        self.play_clip_btn = QtWidgets.QPushButton("▶ Play Clip [Space]")
        self.play_clip_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_clip_btn.setStyleSheet(_PLAY_CLIP_PRIMARY_CSS)
        self.play_clip_btn.setToolTip(
            "Play or pause this bout clip (±10 context frames) in the JABS Video Player [Space]"
        )
        self.play_clip_btn.clicked.connect(self._on_play_clip_clicked)
        hdr_row.addWidget(self.play_clip_btn)
        card_vbox.addLayout(hdr_row)

        # Card Row 2: Plain-English Diagnosis Callout (Flat Cool-Grey + 3.5px Left Accent Stripe)
        self.diagnosis_frame = QtWidgets.QFrame()
        self.diagnosis_frame.setMinimumWidth(0)
        self.diagnosis_frame.setStyleSheet(
            "QFrame { background-color: #F1F5F9; border-left: 4px solid #1E40AF; border-radius: 3px; }"
        )
        diag_lay = QtWidgets.QVBoxLayout(self.diagnosis_frame)
        diag_lay.setContentsMargins(10, 6, 10, 6)
        self.diagnosis_lbl = QtWidgets.QLabel("Loading bout diagnostics…")
        self.diagnosis_lbl.setWordWrap(True)
        self.diagnosis_lbl.setMinimumWidth(0)
        self.diagnosis_lbl.setStyleSheet(
            "font-size: 12px; color: #1E293B; border: none; background: transparent;"
        )
        diag_lay.addWidget(self.diagnosis_lbl)
        card_vbox.addWidget(self.diagnosis_frame)

        # Card Row 3: Context-Aware Wrapping 1-Click Triage Action Bar (hidden in 4. Videos mode)
        self.bout_act_container = QtWidgets.QWidget()
        self.bout_act_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.bout_act_container.setStyleSheet("background: transparent; border: none;")
        act_row = _FlowLayout(self.bout_act_container, h_spacing=5, v_spacing=3)
        act_row.setContentsMargins(0, 0, 0, 0)

        self.mc_target_lbl = QtWidgets.QLabel("Assign:")
        self.mc_target_lbl.setStyleSheet("font-weight: 600; color: #334155;")
        self.mc_target_lbl.setToolTip(
            "Target class saved when pressing [Z] / [1] Confirm or Switch on this bout "
            "(use top 'Behavior:' dropdown to filter the bout list)"
        )
        self.target_pos_combo = QtWidgets.QComboBox()
        self.target_pos_combo.setMinimumWidth(95)
        self.target_pos_combo.setMaximumWidth(210)
        self.target_pos_combo.setMinimumContentsLength(8)
        self.target_pos_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.target_pos_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.target_pos_combo.setToolTip(
            "Target class saved when pressing [Z] / [1] Confirm or Switch on this bout "
            "(use top 'Behavior:' dropdown to filter the bout list)"
        )
        self.target_pos_combo.currentIndexChanged.connect(self._on_target_pos_combo_changed)
        act_row.addWidget(self.mc_target_lbl)
        act_row.addWidget(self.target_pos_combo)

        self.primary_pos_btn = QtWidgets.QPushButton("✓ Keep [Z]")
        self.primary_pos_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.primary_pos_btn.setStyleSheet(
            "QPushButton { background-color: #15803D; color: #FFFFFF; font-weight: 700; "
            "padding: 6px 12px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #166534; }"
            "QPushButton:disabled { background-color: #E2E8F0; color: #94A3B8; }"
        )
        self.primary_pos_btn.clicked.connect(self._on_primary_action_1_clicked)
        act_row.addWidget(self.primary_pos_btn)

        self.secondary_neg_btn = QtWidgets.QPushButton("✗ Not-Behavior [C]")
        self.secondary_neg_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.secondary_neg_btn.setStyleSheet(
            "QPushButton { background-color: #1D4ED8; color: #FFFFFF; font-weight: 700; "
            "padding: 6px 12px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #1E40AF; }"
            "QPushButton:disabled { background-color: #E2E8F0; color: #94A3B8; }"
        )
        self.secondary_neg_btn.clicked.connect(self._on_secondary_action_2_clicked)
        act_row.addWidget(self.secondary_neg_btn)

        self.trim_btn = QtWidgets.QPushButton("✂ Trim Ends [T]")
        self.trim_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.trim_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #0F172A; font-weight: 600; "
            "padding: 5px 10px; border-radius: 5px; border: 1px solid #94A3B8; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
        )
        self.trim_btn.clicked.connect(self._on_trim_clicked)
        act_row.addWidget(self.trim_btn)

        self.split_btn = QtWidgets.QPushButton("✂ Split")
        self.split_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.split_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #0F172A; font-weight: 600; "
            "padding: 5px 10px; border-radius: 5px; border: 1px solid #94A3B8; }"
        )
        self.split_btn.clicked.connect(self._on_split_clicked)
        act_row.addWidget(self.split_btn)

        self.delete_btn = QtWidgets.QPushButton("○ No Label [X]")
        self.delete_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.delete_btn.setStyleSheet(
            "QPushButton { background-color: transparent; color: #B91C1C; font-weight: 600; "
            "padding: 5px 9px; border-radius: 5px; border: 1px solid #FCA5A5; }"
            "QPushButton:hover { background-color: #FEF2F2; }"
        )
        self.delete_btn.setToolTip("Clear label / leave unlabeled [X, Delete, or Backspace]")
        self.delete_btn.clicked.connect(self._on_no_label_x_clicked)
        act_row.addWidget(self.delete_btn)

        self.examine_btn = QtWidgets.QPushButton("✓ Examined [V]")
        self.examine_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.examine_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #0F766E; font-weight: 600; "
            "padding: 5px 9px; border-radius: 5px; border: 1px solid #99F6E4; }"
            "QPushButton:hover { background-color: #F0FDFA; }"
            "QPushButton:disabled { color: #94A3B8; border-color: #E2E8F0; }"
        )
        self.examine_btn.setToolTip(
            "Automatically marked when viewed in JABS — click or press [V] to toggle manually (persists across restarts)"
        )
        self.examine_btn.clicked.connect(self._on_toggle_examined_v_clicked)
        act_row.addWidget(self.examine_btn)

        card_vbox.addWidget(self.bout_act_container)

        # Card Row 3b: Whole-Video Quick Navigation Bar (Visible ONLY in 4. Videos mode)
        self.video_action_bar = QtWidgets.QWidget()
        self.video_action_bar.setStyleSheet("background: transparent; border: none;")
        vact_row = _FlowLayout(self.video_action_bar, h_spacing=5, v_spacing=4)
        vact_row.setContentsMargins(0, 0, 0, 0)

        self.video_open_btn = QtWidgets.QPushButton("▶ Open [Enter]")
        self.video_open_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_open_btn.setStyleSheet(
            "QPushButton { background-color: #1E40AF; color: #FFFFFF; font-weight: 700; "
            "padding: 5px 11px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #1D4ED8; }"
            "QPushButton:disabled { background-color: #E2E8F0; color: #94A3B8; }"
        )
        self.video_open_btn.setToolTip(
            "Load this video in the JABS Player at its focal event (or frame 0 if zero-behavior) [Enter]"
        )
        self.video_open_btn.clicked.connect(self._on_video_open_clicked)
        vact_row.addWidget(self.video_open_btn)

        self.video_open_start_btn = QtWidgets.QPushButton("⏮ f0")
        self.video_open_start_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_open_start_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #0F172A; font-weight: 600; "
            "padding: 5px 9px; border-radius: 5px; border: 1px solid #94A3B8; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
            "QPushButton:disabled { color: #94A3B8; border-color: #E2E8F0; }"
        )
        self.video_open_start_btn.setToolTip(
            "Open this video in JABS right at frame 0 to scan from the beginning"
        )
        self.video_open_start_btn.clicked.connect(self._on_video_open_start_clicked)
        vact_row.addWidget(self.video_open_start_btn)

        self.video_inspect_preds_btn = QtWidgets.QPushButton("🔍 Predictions")
        self.video_inspect_preds_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_inspect_preds_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #0F766E; font-weight: 600; "
            "padding: 5px 9px; border-radius: 5px; border: 1px solid #99F6E4; }"
            "QPushButton:hover { background-color: #F0FDFA; }"
            "QPushButton:disabled { color: #94A3B8; border-color: #E2E8F0; }"
        )
        self.video_inspect_preds_btn.setToolTip(
            "Switch to 2. Review Predictions pre-filtered to this exact video"
        )
        self.video_inspect_preds_btn.clicked.connect(
            lambda: self._on_video_inspect_bouts_clicked("predict")
        )
        vact_row.addWidget(self.video_inspect_preds_btn)

        self.video_inspect_labels_btn = QtWidgets.QPushButton("🔍 Labels")
        self.video_inspect_labels_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_inspect_labels_btn.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: #334155; font-weight: 600; "
            "padding: 5px 9px; border-radius: 5px; border: 1px solid #CBD5E1; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
            "QPushButton:disabled { color: #94A3B8; border-color: #E2E8F0; }"
        )
        self.video_inspect_labels_btn.setToolTip(
            "Switch to 1. Audit Labels pre-filtered to this exact video"
        )
        self.video_inspect_labels_btn.clicked.connect(
            lambda: self._on_video_inspect_bouts_clicked("audit")
        )
        vact_row.addWidget(self.video_inspect_labels_btn)

        self.video_action_bar.setVisible(False)
        card_vbox.addWidget(self.video_action_bar)

        # Hairline divider + Compact Wrapping Inspector View Switcher (hidden in 4. Videos mode)
        self.bout_plot_hdr_container = QtWidgets.QWidget()
        self.bout_plot_hdr_container.setStyleSheet("background: transparent; border: none;")
        plot_hdr = _FlowLayout(self.bout_plot_hdr_container, h_spacing=6, v_spacing=4)
        plot_hdr.setContentsMargins(0, 0, 0, 0)
        self.view_group = QtWidgets.QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_prob_btn = QtWidgets.QPushButton("Frame Probability (Click to Scrub)")
        self.view_prob_btn.setObjectName("subPill")
        self.view_prob_btn.setCheckable(True)
        self.view_prob_btn.setChecked(True)
        self.view_prob_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.view_feat_btn = QtWidgets.QPushButton("Key Features")
        self.view_feat_btn.setObjectName("subPill")
        self.view_feat_btn.setCheckable(True)
        self.view_feat_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.view_pose_btn = QtWidgets.QPushButton("Arena Path")
        self.view_pose_btn.setObjectName("subPill")
        self.view_pose_btn.setCheckable(True)
        self.view_pose_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.view_clusters_btn = QtWidgets.QPushButton("Cluster Profiles (η²)")
        self.view_clusters_btn.setObjectName("subPill")
        self.view_clusters_btn.setCheckable(True)
        self.view_clusters_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.view_clusters_btn.setVisible(False)
        self.view_clusters_btn.setToolTip(
            "View top ANOVA η² features per cluster; click any cluster row to select it"
        )

        self.view_group.addButton(self.view_prob_btn, 0)
        self.view_group.addButton(self.view_feat_btn, 1)
        self.view_group.addButton(self.view_pose_btn, 2)
        self.view_group.addButton(self.view_clusters_btn, 3)
        self.view_group.idClicked.connect(self._on_inspector_view_pill_clicked)

        plot_hdr.addWidget(self.view_clusters_btn)
        plot_hdr.addWidget(self.view_prob_btn)
        plot_hdr.addWidget(self.view_feat_btn)
        plot_hdr.addWidget(self.view_pose_btn)
        card_vbox.addWidget(self.bout_plot_hdr_container)

        # Single clean canvas inside the Triage Card
        self.insp_fig = Figure(figsize=(5.6, 2.7), dpi=100, facecolor="#FFFFFF")
        self.insp_ax = self.insp_fig.add_subplot(111)
        self.insp_canvas = _InteractiveAggCanvas(
            self.insp_fig,
            self.insp_ax,
            interactive_x_scrub=True,
            parent=card_panel,
        )
        self.insp_canvas.setMinimumSize(140, 60)
        self.insp_canvas.x_value_clicked.connect(self._on_prob_curve_frame_clicked)
        self.insp_canvas.y_value_clicked.connect(self._on_cluster_heatmap_row_clicked)
        card_vbox.addWidget(self.insp_canvas, 1)

        upper_splitter.addWidget(card_panel)
        upper_splitter.setStretchFactor(0, 2)
        upper_splitter.setStretchFactor(1, 1)
        main_splitter.addWidget(upper_splitter)

        # 3. Bottom Triage / Whole-Video Queue (Responsive Wrapping Controls + Stacked Sortable Tables)
        bottom_widget = QtWidgets.QWidget()
        bottom_widget.setMinimumWidth(0)
        bottom_vbox = QtWidgets.QVBoxLayout(bottom_widget)
        bottom_vbox.setContentsMargins(0, 4, 0, 0)
        bottom_vbox.setSpacing(5)

        self.bout_qbar_widget = QtWidgets.QWidget()
        qbar = _FlowLayout(self.bout_qbar_widget, h_spacing=6, v_spacing=4)
        qbar.setContentsMargins(0, 0, 0, 0)

        self.qfilter_group = QtWidgets.QButtonGroup(self)
        self.qfilter_group.setExclusive(True)
        self.qf_needs_btn = QtWidgets.QPushButton("Needs Review (0)")
        self.qf_needs_btn.setObjectName("subPill")
        self.qf_needs_btn.setCheckable(True)
        self.qf_needs_btn.setChecked(True)
        self.qf_needs_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.qf_halos_btn = QtWidgets.QPushButton("Label Edges (0)")
        self.qf_halos_btn.setObjectName("subPill")
        self.qf_halos_btn.setCheckable(True)
        self.qf_halos_btn.setVisible(False)
        self.qf_halos_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.qf_halos_btn.setToolTip(
            "Predicted boundary/halo fragments touching or within Window Size of an existing manual label "
            "(useful for refining bout beginnings and ends)"
        )

        self.qf_all_btn = QtWidgets.QPushButton("All Bouts (0)")
        self.qf_all_btn.setObjectName("subPill")
        self.qf_all_btn.setCheckable(True)
        self.qf_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.qf_done_btn = QtWidgets.QPushButton("Reviewed ✓ (0)")
        self.qf_done_btn.setObjectName("subPill")
        self.qf_done_btn.setCheckable(True)
        self.qf_done_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.qfilter_group.addButton(self.qf_needs_btn, 0)
        self.qfilter_group.addButton(self.qf_all_btn, 1)
        self.qfilter_group.addButton(self.qf_done_btn, 2)
        self.qfilter_group.addButton(self.qf_halos_btn, 3)
        self.qfilter_group.idClicked.connect(self._on_queue_filter_pill_clicked)

        qbar.addWidget(self.qf_needs_btn)
        qbar.addWidget(self.qf_halos_btn)
        qbar.addWidget(self.qf_all_btn)
        qbar.addWidget(self.qf_done_btn)

        self.class_filter_combo = QtWidgets.QComboBox()
        self.class_filter_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.class_filter_combo.setMinimumContentsLength(10)
        self.class_filter_combo.setMaximumWidth(195)
        self.class_filter_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.class_filter_combo.addItem("All Classes", "all")
        self.class_filter_combo.setToolTip(
            "Filter bout table & 2D plot by Positive, Negative, or specific behavior class"
        )
        self.class_filter_combo.currentIndexChanged.connect(self._on_class_filter_combo_changed)
        qbar.addWidget(self.class_filter_combo)

        self.quick_select_btn = QtWidgets.QToolButton()
        self.quick_select_btn.setText("⚡ Quick Select ▾")
        self.quick_select_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.quick_select_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.quick_select_btn.setStyleSheet(
            "QToolButton { padding: 4px 9px; border: 1px solid #CBD5E1; border-radius: 5px; "
            "background-color: #FFFFFF; color: #0F172A; font-weight: 600; font-size: 11.5px; }"
            "QToolButton:hover { background-color: #F8FAFC; }"
            "QToolButton::menu-indicator { image: none; }"
        )
        self.quick_select_btn.setToolTip(
            "Select smart cohorts for review: borderline probability, short flickers, clusters, current video, or animal track"
        )
        self.quick_select_menu = QtWidgets.QMenu(self)
        self.quick_select_menu.aboutToShow.connect(self._populate_quick_select_menu)
        self.quick_select_btn.setMenu(self.quick_select_menu)
        qbar.addWidget(self.quick_select_btn)

        self.similar_btn = QtWidgets.QPushButton("🎯 Similar [N]")
        self.similar_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.similar_btn.setStyleSheet(
            "QPushButton { padding: 4px 9px; border: 1px solid #CBD5E1; border-radius: 5px; "
            "background-color: #FFFFFF; color: #0F172A; font-weight: 600; font-size: 11.5px; }"
            "QPushButton:hover { background-color: #F8FAFC; }"
            "QPushButton:disabled { color: #94A3B8; }"
        )
        self.similar_btn.setToolTip(
            "Select the 15 bouts kinematically closest to the currently inspected bout [N]"
        )
        self.similar_btn.clicked.connect(lambda: self._select_nearest_similar_bouts(15))
        qbar.addWidget(self.similar_btn)

        self.clear_box_btn = QtWidgets.QPushButton("✕ Clear Selection")
        self.clear_box_btn.setVisible(False)
        self.clear_box_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clear_box_btn.setStyleSheet(
            "QPushButton { background-color: #DBEAFE; color: #1E40AF; font-weight: 700; "
            "padding: 3px 9px; border-radius: 5px; border: 1px solid #93C5FD; }"
        )
        self.clear_box_btn.setToolTip("Clear active map/cohort selection [Esc]")
        self.clear_box_btn.clicked.connect(self._on_clear_box_filter)
        qbar.addWidget(self.clear_box_btn)

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Filter cage, video, animal, class, C1…")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumWidth(135)
        self.search_input.setMaximumWidth(210)
        self.search_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.search_input.textChanged.connect(self._on_bout_search_text_changed)
        qbar.addWidget(self.search_input)

        self.send_queue_btn = QtWidgets.QPushButton("⇄ Send to JABS")
        self.send_queue_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.send_queue_btn.setStyleSheet(
            "QPushButton { padding: 3px 9px; border: 1px solid #CBD5E1; border-radius: 5px; "
            "background-color: #F8FAFC; color: #1E293B; font-weight: 600; font-size: 11.5px; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
        )
        self.send_queue_btn.setToolTip(
            "Load the active table queue into JABS's bottom < Prev | Next > Search Bar"
        )
        self.send_queue_btn.clicked.connect(self._on_send_queue_to_search_bar)
        qbar.addWidget(self.send_queue_btn)

        bottom_vbox.addWidget(self.bout_qbar_widget)

        # Whole-Video Queue Filter Bar (Visible ONLY in 4. Videos mode)
        self.video_qbar_widget = QtWidgets.QWidget()
        vqbar = _FlowLayout(self.video_qbar_widget, h_spacing=6, v_spacing=4)
        vqbar.setContentsMargins(0, 0, 0, 0)

        self.vfilter_group = QtWidgets.QButtonGroup(self)
        self.vfilter_group.setExclusive(True)

        self.vf_all_btn = QtWidgets.QPushButton("All Videos (0)")
        self.vf_all_btn.setObjectName("subPill")
        self.vf_all_btn.setCheckable(True)
        self.vf_all_btn.setChecked(True)
        self.vf_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.vf_high_frames_btn = QtWidgets.QPushButton("Most Frames (0)")
        self.vf_high_frames_btn.setObjectName("subPill")
        self.vf_high_frames_btn.setCheckable(True)
        self.vf_high_frames_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.vf_high_frames_btn.setToolTip(
            "Videos with the highest total behavior frames / % time across the whole video (50k E1)"
        )

        self.vf_high_bouts_btn = QtWidgets.QPushButton("Most Bouts (0)")
        self.vf_high_bouts_btn.setObjectName("subPill")
        self.vf_high_bouts_btn.setCheckable(True)
        self.vf_high_bouts_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.vf_high_bouts_btn.setToolTip(
            "Videos with the highest total behavior bout count across the whole video (50k E1)"
        )

        self.vf_zero_btn = QtWidgets.QPushButton("Zero Beh (0)")
        self.vf_zero_btn.setObjectName("subPill")
        self.vf_zero_btn.setCheckable(True)
        self.vf_zero_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.vf_zero_btn.setToolTip(
            "Videos with 0 behavior frames and 0 behavior bouts (50k E5 zero controls & missed-bout check)"
        )

        self.vf_outliers_btn = QtWidgets.QPushButton("50k Outliers (0)")
        self.vf_outliers_btn.setObjectName("subPill")
        self.vf_outliers_btn.setCheckable(True)
        self.vf_outliers_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.vf_outliers_btn.setToolTip(
            "All videos flagged by any 50k screening outlier bucket (E1-E4: high rate, long bout, E3a/E3b uncertainty, fragile)"
        )

        self.vfilter_group.addButton(self.vf_all_btn, 0)
        self.vfilter_group.addButton(self.vf_high_frames_btn, 1)
        self.vfilter_group.addButton(self.vf_high_bouts_btn, 2)
        self.vfilter_group.addButton(self.vf_zero_btn, 3)
        self.vfilter_group.addButton(self.vf_outliers_btn, 4)
        self.vfilter_group.idClicked.connect(self._on_video_quick_pill_clicked)

        vqbar.addWidget(self.vf_all_btn)
        vqbar.addWidget(self.vf_high_frames_btn)
        vqbar.addWidget(self.vf_high_bouts_btn)
        vqbar.addWidget(self.vf_zero_btn)
        vqbar.addWidget(self.vf_outliers_btn)

        self.video_clear_sel_btn = QtWidgets.QPushButton("✕ Clear Map Selection")
        self.video_clear_sel_btn.setVisible(False)
        self.video_clear_sel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_clear_sel_btn.setStyleSheet(
            "QPushButton { background-color: #DBEAFE; color: #1E40AF; font-weight: 700; "
            "padding: 3px 9px; border-radius: 5px; border: 1px solid #93C5FD; }"
        )
        self.video_clear_sel_btn.setToolTip("Clear active 2D video map selection [Esc]")
        self.video_clear_sel_btn.clicked.connect(self._on_clear_video_map_selection)
        vqbar.addWidget(self.video_clear_sel_btn)

        self.video_search_input = QtWidgets.QLineEdit()
        self.video_search_input.setPlaceholderText("Search video, cage, E1, E3b…")
        self.video_search_input.setClearButtonEnabled(True)
        self.video_search_input.setMinimumWidth(145)
        self.video_search_input.setMaximumWidth(230)
        self.video_search_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.video_search_input.textChanged.connect(self._on_video_search_text_changed)
        vqbar.addWidget(self.video_search_input)

        self.send_videos_btn = QtWidgets.QPushButton("⇄ Send to JABS")
        self.send_videos_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.send_videos_btn.setStyleSheet(
            "QPushButton { padding: 3px 9px; border: 1px solid #CBD5E1; border-radius: 5px; "
            "background-color: #F8FAFC; color: #1E293B; font-weight: 600; font-size: 11.5px; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
        )
        self.send_videos_btn.setToolTip(
            "Load the currently displayed videos (in table sort order) into JABS's bottom < Prev | Next > Search Bar"
        )
        self.send_videos_btn.clicked.connect(self._on_send_videos_to_search_bar)
        vqbar.addWidget(self.send_videos_btn)

        self.export_videos_btn = QtWidgets.QPushButton("⬇ Export CSV")
        self.export_videos_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_videos_btn.setStyleSheet(
            "QPushButton { padding: 3px 9px; border: 1px solid #CBD5E1; border-radius: 5px; "
            "background-color: #FFFFFF; color: #1E293B; font-weight: 600; font-size: 11.5px; }"
            "QPushButton:hover { background-color: #F1F5F9; }"
        )
        self.export_videos_btn.setToolTip(
            "Export the whole-video summary table & 50k screening buckets to CSV"
        )
        self.export_videos_btn.clicked.connect(self._on_export_videos_csv_clicked)
        vqbar.addWidget(self.export_videos_btn)

        self.video_qbar_widget.setVisible(False)
        bottom_vbox.addWidget(self.video_qbar_widget)

        # Clean 7-Column Sortable Triage Table (Modes 1-3) + 9-Column Whole-Video Table (Mode 4)
        self.bottom_table_stack = QtWidgets.QStackedWidget()
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Priority / Issue",
                "Current State",
                "Video",
                "Animal",
                "Frames (Dur)",
                "Suggested Fix",
                "Examined [V]",
            ]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        self.table.setStyleSheet(
            "QTableWidget { border: 1px solid #E2E8F0; gridline-color: #F1F5F9; font-size: 12px; }"
            "QHeaderView::section { background-color: #F8FAFC; color: #1E293B; font-weight: 700; "
            "border: none; border-bottom: 1.5px solid #CBD5E1; padding: 5px 8px; }"
            "QHeaderView::section:hover { background-color: #EFF6FF; }"
        )
        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionsClickable(True)
        hdr.setSortIndicatorShown(False)
        hdr.sectionClicked.connect(self._on_table_header_clicked)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 64)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 220)
        self.table.setColumnWidth(3, 96)
        self.table.setColumnWidth(4, 130)
        self.table.setColumnWidth(6, 92)
        self.table.setTabKeyNavigation(False)
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.cellClicked.connect(self._on_table_cell_clicked)
        self.bottom_table_stack.addWidget(self.table)

        # Clean 9-Column Sortable Whole-Video Screening Table (Visible ONLY in 4. Videos mode)
        self.video_table = QtWidgets.QTableWidget(0, 9)
        self.video_table.setHorizontalHeaderLabels(
            [
                "Video",
                "Screening",
                "Frames (%)",
                "Bouts",
                "Max Bout",
                "Low-Conf",
                "Near-Miss",
                "Conf",
                "Labels",
            ]
        )
        self.video_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.video_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.video_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.video_table.setAlternatingRowColors(True)
        self.video_table.verticalHeader().setVisible(False)
        self.video_table.verticalHeader().setDefaultSectionSize(25)
        self.video_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.video_table.customContextMenuRequested.connect(self._on_video_table_context_menu)
        self.video_table.setStyleSheet(
            "QTableWidget { border: 1px solid #E2E8F0; gridline-color: #F1F5F9; font-size: 12px; }"
            "QHeaderView::section { background-color: #F8FAFC; color: #1E293B; font-weight: 700; "
            "border: none; border-bottom: 1.5px solid #CBD5E1; padding: 4px 7px; }"
            "QHeaderView::section:hover { background-color: #EFF6FF; }"
        )
        vhdr = self.video_table.horizontalHeader()
        vhdr.setStretchLastSection(False)
        vhdr.setSectionsClickable(True)
        vhdr.setSortIndicatorShown(True)
        vhdr.setSortIndicator(self._video_table_sort_col, self._video_table_sort_order)
        vhdr.sectionClicked.connect(self._on_video_table_header_clicked)
        # Keep all columns Interactive (never permanent ResizeToContents) to prevent O(N^2) Qt freezes
        for c_i in range(9):
            vhdr.setSectionResizeMode(c_i, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.video_table.setColumnWidth(0, 195)
        self.video_table.setTabKeyNavigation(False)
        self.video_table.installEventFilter(self)
        self.video_table.viewport().installEventFilter(self)
        self.video_table.itemSelectionChanged.connect(self._on_video_table_selection_changed)
        self.video_table.cellClicked.connect(self._on_video_table_cell_clicked)
        self.video_table.cellDoubleClicked.connect(lambda _r, _c: self._on_video_open_clicked())
        self.bottom_table_stack.addWidget(self.video_table)

        bottom_vbox.addWidget(self.bottom_table_stack, 1)

        main_splitter.addWidget(bottom_widget)
        main_splitter.setStretchFactor(0, 58)
        main_splitter.setStretchFactor(1, 42)
        root_layout.addWidget(main_splitter, 1)

        # 4. Slim Footer Bar (Elided Status Toast + Elided Shortcut Legend; 0 px min width!)
        footer_bar = QtWidgets.QHBoxLayout()
        footer_bar.setContentsMargins(2, 0, 2, 0)
        footer_bar.setSpacing(10)

        self.status_summary_label = _ElidedLabel("")
        self.status_summary_label.setStyleSheet(
            "color: #1E40AF; font-size: 11.5px; font-weight: 600;"
        )
        footer_bar.addWidget(self.status_summary_label, 1)

        self.shortcut_hint_label = _ElidedLabel(
            "Keys: ↑/↓ Bout • Space Play/Pause • Z Beh • X No Label • C Not • V Examined • N Similar • M Expand • ⌘Z Undo"
        )
        self.shortcut_hint_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.shortcut_hint_label.setStyleSheet("color: #475569; font-size: 11.5px;")
        footer_bar.addWidget(self.shortcut_hint_label, 1)

        root_layout.addLayout(footer_bar)

    def _build_options_menu(self) -> None:
        menu = QtWidgets.QMenu(self)

        # 2D Projection submenu
        proj_menu = menu.addMenu("2D Map Projection")
        proj_grp = QtGui.QActionGroup(self)
        self._proj_menu_actions.clear()
        for label, key in (
            ("PCA (Fast Linear)", "pca"),
            ("MDS (RF Proximity)", "mds"),
            ("t-SNE (Neighborhoods)", "tsne"),
            ("Arena (Spatial Position)", "spatial"),
        ):
            act = proj_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(key == self._proj_method)
            act.triggered.connect(lambda _c=False, k=key: self._set_projection_method(k))
            proj_grp.addAction(act)
            self._proj_menu_actions[key] = act

        # Feature Space submenu
        space_menu = menu.addMenu("Feature Geometry")
        space_grp = QtGui.QActionGroup(self)
        for label, key in (
            ("Auto (Match Active Mode)", "auto"),
            ("Classifier-Weighted", "classifier"),
            ("Unsupervised Kinematic", "unsupervised"),
            ("RF Leaf Proximity", "rf_proximity"),
        ):
            act = space_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(key == self._space_override)
            act.triggered.connect(lambda _c=False, k=key: self._set_space_override(k))
            space_grp.addAction(act)

        # Color By submenu
        color_menu = menu.addMenu("Color 2D Map By")
        color_grp = QtGui.QActionGroup(self)
        for label, key in (
            ("Auto (Match Active Mode)", "auto"),
            ("Behavior Class", "class"),
            ("Issue Severity", "confusion"),
            ("Sub-Type Cluster (C1..Ck)", "cluster"),
            ("Animal ID", "identity"),
            ("Video / Cage", "video"),
        ):
            act = color_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(key == self._color_override)
            act.triggered.connect(lambda _c=False, k=key: self._set_color_override(k))
            color_grp.addAction(act)

        # Prediction sample cap submenu
        cap_menu = menu.addMenu("Max Sampled Predictions")
        cap_grp = QtGui.QActionGroup(self)
        for n_cap in (300, 500, 1000, 2000):
            act = cap_menu.addAction(f"{n_cap} bouts")
            act.setCheckable(True)
            act.setChecked(n_cap == self._max_pred_bouts)
            act.triggered.connect(lambda _c=False, n=n_cap: self._set_max_pred_bouts(n))
            cap_grp.addAction(act)

        menu.addSeparator()

        self.auto_seek_action = menu.addAction("Auto-Seek JABS Player on Selection")
        self.auto_seek_action.setCheckable(True)
        self.auto_seek_action.setChecked(True)

        self.bring_jabs_to_front_action = menu.addAction("Bring JABS Window to Front on Click")
        self.bring_jabs_to_front_action.setCheckable(True)
        self.bring_jabs_to_front_action.setChecked(True)

        self.auto_mark_viewed_action = menu.addAction("Auto-Mark Examined [V] When Viewed in JABS")
        self.auto_mark_viewed_action.setCheckable(True)
        self.auto_mark_viewed_action.setChecked(True)

        self.auto_advance_action = menu.addAction("Auto-Advance to Next Bout After Edit")
        self.auto_advance_action.setCheckable(True)
        self.auto_advance_action.setChecked(False)

        self.pair_opposite_action = menu.addAction(
            "Auto-Pair Opposite Class on Single-Class Tracks (Dual-Class CV)"
        )
        self.pair_opposite_action.setCheckable(True)
        self.pair_opposite_action.setChecked(True)

        self.show_pred_backdrop_action = menu.addAction(
            "Show Labeled Context Backdrop in Prediction Map"
        )
        self.show_pred_backdrop_action.setCheckable(True)
        self.show_pred_backdrop_action.setChecked(True)
        self.show_pred_backdrop_action.toggled.connect(lambda _c=False: self._draw_scatter())

        self.sync_plot_to_queue_tab_action = menu.addAction(
            "Restrict 2D Map to Active Queue Tab (Needs Review / Reviewed)"
        )
        self.sync_plot_to_queue_tab_action.setCheckable(True)
        self.sync_plot_to_queue_tab_action.setChecked(False)
        self.sync_plot_to_queue_tab_action.toggled.connect(lambda _c=False: self._draw_scatter())

        menu.addSeparator()
        reset_beh_act = menu.addAction("↺ Reset Examined Flags (Current Behavior)…")
        reset_beh_act.triggered.connect(lambda: self._on_reset_examined_flags(all_behaviors=False))

        reset_all_act = menu.addAction("↺ Reset All Examined Flags (Project)…")
        reset_all_act.triggered.connect(lambda: self._on_reset_examined_flags(all_behaviors=True))

        menu.addSeparator()
        send_q_act = menu.addAction("Send Current Queue to JABS Search Bar")
        send_q_act.triggered.connect(self._on_send_queue_to_search_bar)

        export_act = menu.addAction("Export CSV + Summary Figure (PNG & SVG)…")
        export_act.triggered.connect(self._on_export_clicked)

        self.options_btn.setMenu(menu)

    # ------------------------------------------------------------------
    # Signal Wiring, Lifecycle & Populate Helpers
    # ------------------------------------------------------------------
    def _wire_central_widget_signals(self) -> None:
        if self._central_widget is None:
            return
        if hasattr(self._central_widget, "labels_modified"):
            sig = self._central_widget.labels_modified
            sig.connect(self._on_external_jabs_labels_modified)
            self._central_widget_connections.append((sig, self._on_external_jabs_labels_modified))
        if hasattr(self._central_widget, "search_hit_loaded"):
            sig = self._central_widget.search_hit_loaded
            sig.connect(self._on_external_search_hit_loaded)
            self._central_widget_connections.append((sig, self._on_external_search_hit_loaded))

    def _on_initial_refresh_timeout(self) -> None:
        if not self._is_closing and self._report is None and self._active_worker is None:
            self.refresh_analysis()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Apply default 2/3 : 1/3 horizontal splitter proportions on first window reveal."""
        super().showEvent(event)
        if not self._initial_splitters_applied:
            self._initial_splitters_applied = True
            if not self._map_expanded:
                self._apply_default_splitter_sizes()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Keep Mode 4 Video column well-proportioned when resizing between desktop and compact widths."""
        super().resizeEvent(event)
        if (
            getattr(self, "_mode", None) == "videos"
            and getattr(self, "video_table", None) is not None
            and self.video_table.rowCount() > 0
        ):
            self._fit_video_table_columns()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Cancel pending background scans, release caches, and close the explorer dialog safely."""
        self._is_closing = True
        self._queued_refresh = False
        self._queued_video_refresh = False
        self._pending_bring_main_window_to_front = False
        self._init_refresh_timer.stop()
        self._scatter_redraw_timer.stop()
        self._bout_search_timer.stop()
        self._video_search_timer.stop()
        self.scatter_canvas.dispose()
        self.insp_canvas.dispose()
        for worker in (self._active_worker, self._spatial_worker, self._video_worker):
            if worker is not None:
                if hasattr(worker, "cancel_event"):
                    worker.cancel_event.set()
                if hasattr(worker, "signals"):
                    with contextlib.suppress(RuntimeError, TypeError):
                        worker.signals.disconnect(self)
        self._active_worker = None
        self._spatial_worker = None
        self._video_worker = None
        self._identity_label_cache.clear()
        self._video_ext_id_cache.clear()
        self._repr_matrix_cache = None
        clear_pose_cache()
        clear_spatial_cache()
        while self._central_widget_connections:
            sig, slot = self._central_widget_connections.pop()
            with contextlib.suppress(TypeError, RuntimeError):
                sig.disconnect(slot)
        super().closeEvent(event)

    @staticmethod
    def _ensure_origin_snapshot(rec: BoutRecord) -> None:
        """Preserve original pre-edit bout identity fields so clearing restores cleanly."""
        if not hasattr(rec, "origin_source"):
            rec.origin_source = rec.source  # type: ignore[attr-defined]
        if not hasattr(rec, "origin_class_name"):
            rec.origin_class_name = rec.class_name  # type: ignore[attr-defined]
        if not hasattr(rec, "origin_behavior"):
            rec.origin_behavior = rec.behavior  # type: ignore[attr-defined]
        if not hasattr(rec, "origin_is_positive"):
            rec.origin_is_positive = rec.is_positive  # type: ignore[attr-defined]

    def _schedule_scatter_redraw(self) -> None:
        """Coalesce rapid 2D scatter redraws so sequential hotkey edits never stutter."""
        if self._is_closing or self._report is None:
            return
        self._scatter_redraw_timer.start()

    def _sync_bouts_from_central_labels(self) -> int:
        """Reconcile in-memory bouts & table rows with edits made in the main JABS GUI (`< 1 ms`)."""
        if self._report is None or self._central_widget is None:
            return 0
        loaded_video = getattr(self._central_widget, "loaded_video", None)
        live_labels = getattr(self._central_widget, "_labels", None)
        if loaded_video is None or live_labels is None:
            return 0
        try:
            loaded_name = Path(loaded_video).name
        except TypeError:
            return 0

        records = self._report.feature_matrix.records
        sm = self._project.settings_manager
        is_mc = self._report.is_multiclass
        pos_val = int(TrackLabels.Label.BEHAVIOR.value)
        neg_val = int(TrackLabels.Label.NOT_BEHAVIOR.value)
        mc_behaviors = [MULTICLASS_NONE_BEHAVIOR, *list(sm.behavior_names)] if is_mc else []

        changed_indices: list[int] = []
        touched_identities: set[int] = set()
        n_frames_live = int(getattr(live_labels, "num_frames", 0) or 0)

        for b_idx, rec in enumerate(records):
            if Path(rec.video).name != loaded_name:
                continue
            self._ensure_origin_snapshot(rec)
            origin_src = str(getattr(rec, "origin_source", rec.source))
            ident_str = str(rec.identity)
            s = max(0, int(rec.start_frame))
            e = int(rec.end_frame)
            if n_frames_live > 0:
                e = min(n_frames_live - 1, e)
            if s > e:
                continue

            new_labeled = False
            new_is_pos = False
            new_beh = rec.behavior
            new_cls = rec.class_name

            if is_mc:
                best_cnt = 0
                best_name = ""
                for b_name in mc_behaviors:
                    try:
                        arr = live_labels.get_track_labels(ident_str, b_name).get_labels()
                    except Exception:
                        continue
                    if len(arr) <= s:
                        continue
                    cnt = int(np.count_nonzero(np.asarray(arr[s : e + 1]) == pos_val))
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_name = b_name
                if best_cnt > 0:
                    new_labeled = True
                    new_beh = best_name
                    new_cls = best_name
                    new_is_pos = best_name != MULTICLASS_NONE_BEHAVIOR
            else:
                beh_key = (
                    rec.behavior
                    if (rec.behavior and rec.behavior != MULTICLASS_NONE_BEHAVIOR)
                    else self._report.behavior
                )
                try:
                    arr = live_labels.get_track_labels(ident_str, beh_key).get_labels()
                except Exception:
                    arr = None
                if arr is not None and len(arr) > s:
                    sl = np.asarray(arr[s : e + 1])
                    n_pos = int(np.count_nonzero(sl == pos_val))
                    n_neg = int(np.count_nonzero(sl == neg_val))
                    if n_pos > 0 or n_neg > 0:
                        new_labeled = True
                        new_beh = beh_key
                        new_is_pos = n_pos >= n_neg
                        new_cls = beh_key if new_is_pos else f"Not-{beh_key}"

            if new_labeled:
                was_same = (
                    rec.source == "labeled"
                    and rec.class_name == new_cls
                    and rec.is_positive == new_is_pos
                    and rec.bout_id not in self._unlabeled_bout_ids
                    and rec.bout_id not in self._deleted_bout_ids
                )
                if not was_same:
                    rec.source = "labeled"
                    rec.behavior = new_beh
                    rec.class_name = new_cls
                    rec.is_positive = new_is_pos
                    self._deleted_bout_ids.discard(rec.bout_id)
                    self._unlabeled_bout_ids.discard(rec.bout_id)
                    self._examined_only_bout_ids.discard(rec.bout_id)
                    self._reviewed_bout_ids.add(rec.bout_id)
                    self._pinned_needs_review_bout_ids.add(rec.bout_id)
                    gui_note = f"✓ Labeled {new_cls} in JABS"
                    self._saved_notes_by_bout[rec.bout_id] = gui_note
                    self._examined_store.mark(rec, action="gui_edit", note=gui_note, persist=False)
                    changed_indices.append(b_idx)
                    touched_identities.add(int(rec.identity))
            else:
                if origin_src == "predicted" and rec.source == "labeled":
                    rec.source = "predicted"
                    rec.behavior = str(getattr(rec, "origin_behavior", rec.behavior))
                    rec.class_name = str(getattr(rec, "origin_class_name", rec.class_name))
                    rec.is_positive = bool(getattr(rec, "origin_is_positive", True))
                    self._reviewed_bout_ids.discard(rec.bout_id)
                    self._examined_only_bout_ids.discard(rec.bout_id)
                    self._unlabeled_bout_ids.discard(rec.bout_id)
                    self._saved_notes_by_bout.pop(rec.bout_id, None)
                    self._examined_store.unmark(rec, persist=False)
                    changed_indices.append(b_idx)
                    touched_identities.add(int(rec.identity))
                elif origin_src == "labeled" and rec.source == "labeled":
                    rec.source = "unlabeled"  # type: ignore[assignment]
                    rec.class_name = "No Label"
                    rec.is_positive = False
                    self._unlabeled_bout_ids.add(rec.bout_id)
                    self._examined_only_bout_ids.discard(rec.bout_id)
                    self._reviewed_bout_ids.add(rec.bout_id)
                    self._pinned_needs_review_bout_ids.add(rec.bout_id)
                    gui_clr = "○ Cleared (No Label) in JABS"
                    self._saved_notes_by_bout[rec.bout_id] = gui_clr
                    self._examined_store.mark(rec, action="no_label", note=gui_clr, persist=False)
                    changed_indices.append(b_idx)
                    touched_identities.add(int(rec.identity))

        if not changed_indices:
            return 0
        self._examined_store.save()

        # Refresh in-memory CV track counts for the modified video identities
        if self._report.cv_track_counts is not None:
            for ident in touched_identities:
                ident_str = str(ident)
                # Match existing key video naming in records
                vid_key = next(
                    (records[i].video for i in changed_indices if records[i].identity == ident),
                    loaded_name,
                )
                if is_mc:
                    beh_focus = self._report.behavior
                    pos_c = 0
                    other_c = 0
                    for b_name in mc_behaviors:
                        try:
                            arr = np.asarray(
                                live_labels.get_track_labels(ident_str, b_name).get_labels()
                            )
                        except Exception:
                            continue
                        n_b = int(np.count_nonzero(arr == pos_val))
                        if b_name == beh_focus:
                            pos_c += n_b
                        else:
                            other_c += n_b
                    if pos_c > 0 or other_c > 0:
                        self._report.cv_track_counts[(vid_key, ident)] = (pos_c, other_c)
                    else:
                        self._report.cv_track_counts.pop((vid_key, ident), None)
                else:
                    try:
                        arr = np.asarray(
                            live_labels.get_track_labels(
                                ident_str, self._report.behavior
                            ).get_labels()
                        )
                        pos_c = int(np.count_nonzero(arr == pos_val))
                        neg_c = int(np.count_nonzero(arr == neg_val))
                    except Exception:
                        pos_c, neg_c = 0, 0
                    if pos_c > 0 or neg_c > 0:
                        self._report.cv_track_counts[(vid_key, ident)] = (pos_c, neg_c)
                    else:
                        self._report.cv_track_counts.pop((vid_key, ident), None)
            cv_cnt, jabs_met, adr_met = summarize_cv_track_counts(self._report.cv_track_counts)
            self._report.cv_groups_count = cv_cnt
            self._report.jabs_threshold_met = jabs_met
            self._report.dual_class_cv_met = adr_met
            self._update_cv_badge_from_counts(cv_cnt, jabs_met, adr_met)

        self._update_queue_filter_counts()
        self._update_table_rows_in_place(changed_indices)
        if self._selected_bout_idx in set(changed_indices):
            self._update_inspector_card(sync_mc_target=False, redraw_plot=False)
        self._schedule_scatter_redraw()
        return len(changed_indices)

    def _on_external_jabs_labels_modified(self) -> None:
        if self._ignoring_self_label_signal or self._is_closing:
            return
        self._external_labels_dirty = True
        self._video_report_key = None
        self._update_recalc_button_badge()
        n_synced = self._sync_bouts_from_central_labels()
        if self._mode == "videos":
            self.refresh_video_summary(force=True)
            return
        if n_synced > 0:
            self.status_summary_label.setText(
                f"Synced {n_synced} bout{'s' if n_synced != 1 else ''} from JABS timeline — "
                "click ↻ Recalculate [R] to rescore."
            )
        else:
            self.status_summary_label.setText(
                "Timeline labels updated — click ↻ Recalculate [R] to rescore."
            )

    def _video_external_identities(self, video: str) -> list[str] | None:
        """Read `/poseest/external_identity_mapping` header (< 0.05 ms) without loading full keypoints."""
        if video in self._video_ext_id_cache:
            return self._video_ext_id_cache[video]
        ext_ids: list[str] | None = None
        try:
            vm = getattr(self._project, "video_manager", None)
            pose_path = None
            if vm is not None:
                if hasattr(vm, "get_cached_pose_path"):
                    pose_path = vm.get_cached_pose_path(video)
                elif hasattr(vm, "pose_path"):
                    pose_path = vm.pose_path(video)
            if isinstance(pose_path, str | Path) and Path(pose_path).is_file():
                import h5py

                with h5py.File(pose_path, "r") as pose_h5:
                    if "poseest/external_identity_mapping" in pose_h5:
                        raw_ids = pose_h5["poseest/external_identity_mapping"][:]
                        if np.issubdtype(raw_ids.dtype, np.integer):
                            ext_ids = [str(x) for x in raw_ids.tolist()]
                        else:
                            ext_ids = [
                                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                                for x in raw_ids.tolist()
                            ]
        except Exception:
            ext_ids = None
        self._video_ext_id_cache[video] = ext_ids
        return ext_ids

    def _animal_display_text(self, video: str, identity: int) -> str:
        """Format `identity` to match JABS's Animal pane (`PoseEstimation.identity_index_to_display`)."""
        idx = int(identity)
        key = (video, idx)
        loaded = (
            getattr(self._central_widget, "loaded_video", None) if self._central_widget else None
        )
        pose_live = (
            getattr(self._central_widget, "_pose_est", None) if self._central_widget else None
        )
        if loaded is not None and pose_live is not None and loaded.name == Path(video).name:
            try:
                raw_disp = pose_live.identity_index_to_display(idx)
                if isinstance(raw_disp, str | int | np.integer) and str(raw_disp).strip():
                    label = f"Animal {str(raw_disp).strip()}"
                    self._identity_label_cache[key] = label
                    return label
            except Exception:
                pass
        cached = self._identity_label_cache.get(key)
        if cached is not None:
            return cached
        disp = str(idx)
        try:
            ext_ids = self._video_external_identities(video)
            if ext_ids and 0 <= idx < len(ext_ids) and str(ext_ids[idx]).strip():
                disp = str(ext_ids[idx]).strip()
        except Exception:
            disp = str(idx)
        label = f"Animal {disp}"
        self._identity_label_cache[key] = label
        return label

    def _populate_behaviors(self) -> None:
        self.behavior_combo.blockSignals(True)
        self.behavior_combo.clear()
        sm = self._project.settings_manager
        behaviors = list(sm.behavior_names)
        is_mc = sm.classifier_mode == ClassifierMode.MULTICLASS
        if is_mc:
            self.behavior_combo.addItem("All Classes (Multi-Class)", "__ALL_MC__")
        for b in behaviors:
            self.behavior_combo.addItem(b, b)
        if self._central_widget is not None:
            curr = getattr(self._central_widget, "behavior", None)
            if isinstance(curr, str) and curr:
                idx = self.behavior_combo.findData(curr)
                if idx >= 0:
                    self.behavior_combo.setCurrentIndex(idx)
        self.behavior_combo.blockSignals(False)
        self._refresh_target_pos_combo()

    def _refresh_target_pos_combo(self) -> None:
        sm = self._project.settings_manager
        is_mc = sm.classifier_mode == ClassifierMode.MULTICLASS
        behaviors = list(sm.behavior_names)
        self.target_pos_combo.blockSignals(True)
        self.target_pos_combo.clear()
        for b in behaviors:
            self.target_pos_combo.addItem(b, b)
        chosen = self.behavior_combo.currentData()
        if isinstance(chosen, str) and chosen != "__ALL_MC__":
            idx = self.target_pos_combo.findData(chosen)
            if idx >= 0:
                self.target_pos_combo.setCurrentIndex(idx)
        elif is_mc and self._central_widget is not None:
            curr = getattr(self._central_widget, "behavior", None)
            if curr:
                idx = self.target_pos_combo.findData(curr)
                if idx >= 0:
                    self.target_pos_combo.setCurrentIndex(idx)
        self.target_pos_combo.blockSignals(False)
        # Progressive disclosure: only show Target Class picker in Multi-Class triage modes
        show_mc = is_mc and self._mode not in ("subtypes", "videos")
        self.mc_target_lbl.setVisible(show_mc)
        self.target_pos_combo.setVisible(show_mc)

    def _on_target_pos_combo_changed(self) -> None:
        if self._report is not None and self._selected_bout_idx is not None:
            self._update_inspector_card(sync_mc_target=False)

    def _behavior_color_hex_map(self) -> dict[str, str]:
        sm = self._project.settings_manager
        behaviors = list(sm.behavior_names)
        if sm.classifier_mode == ClassifierMode.MULTICLASS:
            try:
                qmap = make_behavior_color_map(behaviors)
            except Exception:
                qmap = {}
            out = {b: _qcolor_to_hex(c) for b, c in qmap.items()}
            out[MULTICLASS_NONE_BEHAVIOR] = _qcolor_to_hex(NOT_BEHAVIOR_COLOR, darken_light=False)
            return out
        pos_hex = _qcolor_to_hex(BEHAVIOR_COLOR, darken_light=False)
        neg_hex = _qcolor_to_hex(NOT_BEHAVIOR_COLOR, darken_light=False)
        out = dict.fromkeys(behaviors, pos_hex)
        for b in behaviors:
            out[f"Not-{b}"] = neg_hex
        return out

    def _effective_subtype_scope(self) -> SubtypeScope:
        if self._mode != "subtypes":
            return "all_labeled"
        return self._subtypes_scope

    def _effective_feature_lens(self) -> FeatureLens:
        if self._mode != "subtypes":
            return "all"
        if self._subtypes_lens_ui == "rf_proximity":
            return "all"
        if self._subtypes_lens_ui in (
            "all",
            "intrinsic",
            "locomotion",
            "posture",
            "social",
            "classifier_weighted",
        ):
            return self._subtypes_lens_ui  # type: ignore[return-value]
        return "intrinsic"

    def _effective_forced_k(self) -> int | None:
        return self._subtypes_forced_k if self._mode == "subtypes" else None

    def _effective_subtype_target_behavior(self) -> str | None:
        if (
            self._mode != "subtypes"
            or self._project.settings_manager.classifier_mode != ClassifierMode.MULTICLASS
        ):
            return None
        return self._selected_single_behavior()

    def _effective_space(self) -> RepresentationSpace:
        if self._space_override != "auto":
            return self._space_override
        if self._mode == "subtypes":
            if self._subtypes_lens_ui == "rf_proximity":
                return "rf_proximity"
            if self._subtypes_lens_ui == "classifier_weighted":
                return "classifier"
            return "unsupervised"
        return "classifier"

    def _effective_color_mode(self) -> str:
        if self._color_override != "auto":
            return self._color_override
        if self._mode == "subtypes":
            return "cluster"
        return "class"

    def _sync_subtypes_bar_from_report(self) -> None:
        if self._report is None:
            self.subtypes_summary_lbl.setText("")
            self.cage_skew_badge.setVisible(False)
            return
        emb = self._report.embedding
        cnt_parts = " • ".join(f"C{c}:{n}" for c, n in sorted(emb.cluster_counts.items()))
        scope_tag = "Pos-Only" if emb.subtype_scope == "pos_only" else "Pos+Neg"
        top_txt = ""
        if emb.cluster_feature_eta2:
            f0, e0 = next(iter(emb.cluster_feature_eta2.items()))
            if e0 > 0.0:
                top_txt = f" • Top η²: {_short_feature_label(f0, 18)} ({e0 * 100:.0f}%)"
        self.subtypes_summary_lbl.setText(
            f"{scope_tag} • k={emb.optimal_k} ({cnt_parts}) • Sil={emb.silhouette:.2f}{top_txt}"
        )

        p_val = emb.cage_skew_p
        v_val = emb.cage_skew_cramers_v
        if p_val is not None and v_val is not None and p_val < 0.005 and v_val >= 0.35:
            dom_lines = [
                f"C{cid}: {cage} ({frac * 100:.0f}%)"
                for cid, (cage, frac) in sorted(emb.dominant_cage_by_cluster.items())
            ]
            self.cage_skew_badge.setText(f"⚠ Cage Skew (V={v_val:.2f})")
            self.cage_skew_badge.setToolTip(
                f"Sub-type clusters correlate strongly with cage/video batch "
                f"(chi2 p={p_val:.2e}, Cramér's V={v_val:.2f}).\n" + "\n".join(dom_lines)
            )
            self.cage_skew_badge.setVisible(True)
        else:
            self.cage_skew_badge.setVisible(False)

    def _on_subtypes_lens_combo_changed(self) -> None:
        self._subtypes_lens_ui = str(self.subtypes_lens_combo.currentData() or "intrinsic")
        self._space_override = "auto"
        self._on_subtypes_controls_changed()

    def _on_subtypes_controls_changed(self) -> None:
        scope_val = self.subtypes_scope_combo.currentData()
        self._subtypes_scope = "all_labeled" if scope_val == "all_labeled" else "pos_only"
        self._subtypes_lens_ui = str(self.subtypes_lens_combo.currentData() or "intrinsic")
        raw_k = self.subtypes_k_combo.currentData()
        self._subtypes_forced_k = int(raw_k) if raw_k is not None else None
        if self._subtypes_scope == "pos_only" and self._class_filter == "neg":
            self._class_filter = "all"
        if self._report is None or self._mode != "subtypes":
            return

        self._clear_custom_selection_state()
        self._scatter_view_limits = None
        self._sync_zoom_reset_button()
        target_space = self._effective_space()
        self._report = recompute_embedding_only(
            self._report,
            space=target_space,
            subtype_scope=self._effective_subtype_scope(),
            feature_lens=self._effective_feature_lens(),
            target_behavior=self._effective_subtype_target_behavior(),
            compute_tsne=(self._proj_method == "tsne"),
            compute_mds=(target_space == "rf_proximity" or self._proj_method == "mds"),
            forced_k=self._effective_forced_k(),
        )
        self._repr_matrix_cache = None
        self._sync_subtypes_bar_from_report()
        self._populate_table(auto_fallback_if_clean=True, redraw_scatter_on_change=False)
        if self._proj_method == "spatial" and self._report.spatial is None:
            self._request_spatial_computation()
        self._draw_scatter()
        self._update_inspector_card()

    def _on_cluster_heatmap_row_clicked(self, y_val: float) -> None:
        if self._report is None or self._inspector_view != "clusters":
            return
        z_mat = self._report.embedding.cluster_z_matrix
        if z_mat is None or z_mat.shape[0] <= 0:
            return
        row_idx = round(y_val)
        if 0 <= row_idx < z_mat.shape[0]:
            self._select_cluster_cohort(row_idx + 1)

    # ------------------------------------------------------------------
    # Mode & Option Switching (Instant Warm Switch Where Possible)
    # ------------------------------------------------------------------
    def _on_recalc_btn_clicked(self) -> None:
        if self._mode == "videos":
            self.refresh_video_summary(force=True)
        else:
            self.refresh_analysis()

    def _on_behavior_combo_changed(self) -> None:
        self._video_report_key = None
        self._class_filter = "all"
        self._refresh_class_filter_combos()
        self._refresh_target_pos_combo()
        if self._mode == "videos":
            self._analysis_needed_on_bout_mode = True
            self.refresh_video_summary(force=True)
            return
        self._analysis_needed_on_bout_mode = False
        self.refresh_analysis()

    def _on_mode_pill_clicked(self, mode_id: int) -> None:
        new_mode: ExplorerWorkflowMode = (
            "audit"
            if mode_id == 0
            else ("predict" if mode_id == 1 else ("subtypes" if mode_id == 2 else "videos"))
        )
        if new_mode == self._mode and (
            (new_mode == "videos" and self._video_report is not None)
            or (new_mode != "videos" and self._report is not None)
        ):
            return
        self._mode = new_mode
        self._pinned_needs_review_bout_ids.clear()
        self._clear_custom_selection_state()
        if (
            new_mode == "subtypes"
            and self._effective_subtype_scope() == "pos_only"
            and self._class_filter == "neg"
        ):
            self._class_filter = "all"
        self._scatter_view_limits = None
        self._scatter_default_limits = None
        self._sync_zoom_reset_button()
        self._refresh_target_pos_combo()

        # Progressive disclosure across the 4 workflow modes
        is_sub = new_mode == "subtypes"
        is_vid = new_mode == "videos"
        self.subtypes_bar.setVisible(is_sub)
        self.videos_bar.setVisible(is_vid)
        self.undo_btn.setVisible(not is_vid)
        self.bout_act_container.setVisible(not is_vid)
        self.bout_plot_hdr_container.setVisible(not is_vid)
        self.video_action_bar.setVisible(is_vid)
        self.bout_qbar_widget.setVisible(not is_vid)
        self.video_qbar_widget.setVisible(is_vid)
        self.bottom_table_stack.setCurrentIndex(1 if is_vid else 0)
        self.view_clusters_btn.setVisible(is_sub)
        self.view_prob_btn.setVisible(not is_sub)
        self.qf_halos_btn.setVisible(new_mode == "predict")
        self.arena_map_btn.setVisible(not is_vid)
        self.map_class_combo.setVisible(not is_vid)

        if is_vid:
            self.insp_canvas._interactive_x_scrub = True
            self.insp_canvas._interactive_y_pick = False
            self.play_clip_btn.setText("▶ Play Focal [Space]")
            self.play_clip_btn.setStyleSheet(_PLAY_CLIP_SECONDARY_CSS)
            self._update_recalc_button_badge()
            self.shortcut_hint_label.setText(
                "Keys: ↑/↓ Video • Enter Open • Space Play Focal • Click Timeline Scrub • M Expand • R Rescan"
            )
            self.refresh_video_summary(force=False)
            return

        self.play_clip_btn.setText("▶ Play Clip [Space]")
        self.play_clip_btn.setStyleSheet(_PLAY_CLIP_PRIMARY_CSS)
        self._update_recalc_button_badge()
        self.shortcut_hint_label.setText(
            "Keys: ↑/↓ Bout • Space Play/Pause • Z Beh • X No Label • C Not • V Examined • N Similar • A Arena • M Expand • ⌘Z Undo"
        )

        # Default queue filter & inspector view per bout mode
        if is_sub:
            self._set_queue_filter("all", redraw=False)
            self._set_inspector_view("clusters", redraw=False)
        else:
            self._set_queue_filter("needs_review", redraw=False)
            self._set_inspector_view("prob", redraw=False)

        # If behavior changed while in 4. Videos (or initial bout report isn't ready), run bout CV now
        if self._analysis_needed_on_bout_mode or self._report is None:
            self._analysis_needed_on_bout_mode = False
            self.refresh_analysis()
            return

        # If switching to Review Predictions and predictions aren't loaded yet, fetch them once
        if new_mode == "predict" and not self._report_includes_predictions:
            self.refresh_analysis()
            return

        # Otherwise warm-switch geometry in < 60 ms without touching disk!
        if self._report is not None:
            target_space = self._effective_space()
            target_scope = self._effective_subtype_scope()
            target_lens = self._effective_feature_lens()
            target_beh = self._effective_subtype_target_behavior()
            target_k = self._effective_forced_k()
            emb = self._report.embedding
            need_tsne = self._proj_method == "tsne" and emb.coords_tsne_2d is None
            need_mds = (
                target_space == "rf_proximity" or self._proj_method == "mds"
            ) and emb.coords_mds_2d is None
            if (
                target_space != self._report.space
                or target_scope != emb.subtype_scope
                or target_lens != emb.feature_lens
                or target_beh != emb.target_behavior
                or target_k != emb.forced_k
                or need_tsne
                or need_mds
            ):
                self._report = recompute_embedding_only(
                    self._report,
                    space=target_space,
                    subtype_scope=target_scope,
                    feature_lens=target_lens,
                    target_behavior=target_beh,
                    compute_tsne=self._proj_method == "tsne",
                    compute_mds=(target_space == "rf_proximity" or self._proj_method == "mds"),
                    forced_k=target_k,
                )
                self._repr_matrix_cache = None
            self._sync_subtypes_bar_from_report()
            self._populate_table(auto_fallback_if_clean=True, redraw_scatter_on_change=False)
            self._draw_scatter()

    def _toggle_arena_map(self, checked: bool | None = None) -> None:
        if self._mode == "videos":
            self.arena_map_btn.blockSignals(True)
            self.arena_map_btn.setChecked(False)
            self.arena_map_btn.blockSignals(False)
            return
        want_arena = (self._proj_method != "spatial") if checked is None else bool(checked)
        self._set_projection_method("spatial" if want_arena else self._last_feature_proj)

    def _sync_projection_controls(self) -> None:
        is_arena = self._proj_method == "spatial"
        self.arena_map_btn.blockSignals(True)
        self.arena_map_btn.setChecked(is_arena)
        self.arena_map_btn.blockSignals(False)
        for key, act in self._proj_menu_actions.items():
            act.blockSignals(True)
            act.setChecked(key == self._proj_method)
            act.blockSignals(False)

    def _request_spatial_computation(self) -> None:
        if self._report is None or self._is_closing:
            return
        if self._spatial_worker is not None and hasattr(self._spatial_worker, "cancel_event"):
            self._spatial_worker.cancel_event.set()
        self._spatial_request_id += 1
        req_id = self._spatial_request_id
        self.status_summary_label.setText("Mapping bout arena locations…")
        worker = _SpatialComputeTask(req_id, self._project, self._report.feature_matrix.records)
        self._spatial_worker = worker
        worker.signals.finished.connect(self._on_spatial_finished)
        worker.signals.cancelled.connect(self._on_spatial_cancelled)
        worker.signals.error.connect(self._on_spatial_error)
        self._thread_pool.start(worker)

    def _on_spatial_cancelled(self, req_id: int) -> None:
        if self._is_closing or req_id != self._spatial_request_id:
            return
        if (
            self._spatial_worker is not None
            and getattr(self._spatial_worker, "req_id", None) == req_id
        ):
            self._spatial_worker = None

    def _on_spatial_finished(self, req_id: int, spatial_res: BoutSpatialResult) -> None:
        if self._is_closing or req_id != self._spatial_request_id or self._report is None:
            return
        self._spatial_worker = None
        if len(spatial_res.coords_xy) != len(self._report.feature_matrix.records):
            return
        self._report.spatial = spatial_res
        self._report.timings_ms["spatial_ms"] = float(spatial_res.elapsed_ms)
        if self._proj_method == "spatial" and self._mode != "videos":
            self._scatter_view_limits = None
            self._scatter_default_limits = None
            self._redraw_scatter_and_inspector()
            n_mapped = int(np.sum(spatial_res.valid_mask))
            self.status_summary_label.setText(
                f"Arena Spatial Map ready ({n_mapped}/{len(spatial_res.coords_xy)} bouts mapped in "
                f"{spatial_res.elapsed_ms:.0f} ms)"
            )

    def _on_spatial_error(self, req_id: int, message: str) -> None:
        if self._is_closing or req_id != self._spatial_request_id:
            return
        self._spatial_worker = None
        self.status_summary_label.setText(f"Arena spatial mapping failed: {message}")
        if self._proj_method == "spatial":
            self._draw_scatter()

    def _set_projection_method(self, method: ProjectionMethod) -> None:
        if method in ("pca", "mds", "tsne"):
            self._last_feature_proj = method
        self._proj_method = method
        self._sync_projection_controls()
        self._scatter_view_limits = None
        self._scatter_default_limits = None
        self._sync_zoom_reset_button()
        if method == "spatial":
            if self._report is not None and self._report.spatial is None:
                self._request_spatial_computation()
            self._redraw_scatter_and_inspector()
            return
        if self._report is not None and (
            (method == "tsne" and self._report.embedding.coords_tsne_2d is None)
            or (method == "mds" and self._report.embedding.coords_mds_2d is None)
        ):
            target_space = self._effective_space()
            self._report = recompute_embedding_only(
                self._report,
                space=target_space,
                subtype_scope=self._effective_subtype_scope(),
                feature_lens=self._effective_feature_lens(),
                target_behavior=self._effective_subtype_target_behavior(),
                compute_tsne=(method == "tsne"),
                compute_mds=(target_space == "rf_proximity" or method == "mds"),
                forced_k=self._effective_forced_k(),
            )
            self._sync_subtypes_bar_from_report()
        self._redraw_scatter_and_inspector()

    def _set_space_override(
        self, override: Literal["auto", "classifier", "unsupervised", "rf_proximity"]
    ) -> None:
        self._space_override = override
        self._scatter_view_limits = None
        self._sync_zoom_reset_button()
        if self._report is not None:
            target_space = self._effective_space()
            if target_space != self._report.space:
                self._report = recompute_embedding_only(
                    self._report,
                    space=target_space,
                    subtype_scope=self._effective_subtype_scope(),
                    feature_lens=self._effective_feature_lens(),
                    target_behavior=self._effective_subtype_target_behavior(),
                    compute_tsne=self._proj_method == "tsne",
                    compute_mds=(target_space == "rf_proximity" or self._proj_method == "mds"),
                    forced_k=self._effective_forced_k(),
                )
                self._repr_matrix_cache = None
                self._sync_subtypes_bar_from_report()
            self._redraw_scatter_and_inspector()

    def _set_color_override(
        self,
        override: Literal["auto", "class", "confusion", "cluster", "identity", "video"],
    ) -> None:
        self._color_override = override
        self._redraw_scatter_and_inspector()

    def _set_max_pred_bouts(self, n_cap: int) -> None:
        self._max_pred_bouts = int(n_cap)
        if self._mode == "predict":
            self._report_includes_predictions = False
            self.refresh_analysis()

    def _set_queue_filter(self, qf: QueueFilterMode, *, redraw: bool = True) -> None:
        if qf == "label_edges" and self._mode != "predict":
            qf = "needs_review"
        self._queue_filter = qf
        self.qf_needs_btn.setChecked(qf == "needs_review")
        self.qf_halos_btn.setChecked(qf == "label_edges")
        self.qf_all_btn.setChecked(qf == "all")
        self.qf_done_btn.setChecked(qf == "reviewed")
        if redraw:
            self._populate_table()

    def _on_queue_filter_pill_clicked(self, pill_id: int) -> None:
        qf: QueueFilterMode = (
            "needs_review"
            if pill_id == 0
            else ("all" if pill_id == 1 else ("reviewed" if pill_id == 2 else "label_edges"))
        )
        self._pinned_needs_review_bout_ids.clear()
        had_custom = bool(self._lasso_selected_indices)
        if had_custom:
            self._clear_custom_selection_state()
            self._set_queue_filter(qf, redraw=False)
            self._populate_table(redraw_scatter_on_change=False)
            self._draw_scatter()
        else:
            self._set_queue_filter(qf, redraw=True)

    @staticmethod
    def _is_bout_unlabeled(rec: BoutRecord, unlabeled_ids: set[int]) -> bool:
        return rec.bout_id in unlabeled_ids or rec.source == "unlabeled"

    def _class_filter_choices(self) -> list[tuple[str, str]]:
        """Build `(display_label, filter_key)` items for the synchronized Class Filter dropdowns."""
        if self._report is None:
            return [("All Classes", "all")]
        records = self._report.feature_matrix.records
        base = self._mode_eligible_indices()
        unlabeled_ids = self._unlabeled_bout_ids

        n_pos = 0
        n_neg = 0
        n_unlbl = 0
        pos_cls_counts: dict[str, int] = {}
        for gi in base:
            rec = records[gi]
            if self._is_bout_unlabeled(rec, unlabeled_ids):
                n_unlbl += 1
            elif rec.is_positive:
                n_pos += 1
                if rec.class_name != MULTICLASS_NONE_BEHAVIOR:
                    pos_cls_counts[rec.class_name] = pos_cls_counts.get(rec.class_name, 0) + 1
            else:
                n_neg += 1

        include_all_mc = bool(
            getattr(self._report, "include_all_multiclass", False)
            or self.behavior_combo.currentData() == "__ALL_MC__"
        )
        choices: list[tuple[str, str]] = [(f"All Classes ({len(base)})", "all")]
        if include_all_mc:
            choices.append((f"+ All Positive ({n_pos})", "pos"))
            choices.append((f"- None ({n_neg})", "neg"))
            for cls_name in sorted(pos_cls_counts):
                choices.append((f"● {cls_name} ({pos_cls_counts[cls_name]})", f"class:{cls_name}"))
        else:
            beh = self._selected_single_behavior() or self._report.behavior
            neg_lbl = "None" if self._report.is_multiclass else f"Not-{beh}"
            choices.append((f"+ {beh} ({n_pos})", "pos"))
            choices.append((f"- {neg_lbl} ({n_neg})", "neg"))
        if n_unlbl > 0 or self._class_filter == "unlabeled":
            choices.append((f"○ No Label ({n_unlbl})", "unlabeled"))
        return choices

    def _refresh_class_filter_combos(self) -> None:
        """Populate both Class Filter combos with live counts and synchronize active styling."""
        choices = self._class_filter_choices()
        valid_keys = {k for _, k in choices}
        if self._class_filter not in valid_keys:
            self._class_filter = "all"

        is_filtered = self._class_filter != "all"
        css = (
            "QComboBox { background-color: #DBEAFE; color: #1E40AF; "
            "border: 1px solid #93C5FD; border-radius: 4px; font-weight: 700; padding: 2px 6px; }"
            if is_filtered
            else ""
        )
        for combo in (self.map_class_combo, self.class_filter_combo):
            combo.blockSignals(True)
            combo.clear()
            for lbl, key in choices:
                combo.addItem(lbl, key)
            idx = combo.findData(self._class_filter)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.setStyleSheet(css)
            combo.blockSignals(False)

    def _set_class_filter(self, filt_key: str, *, redraw: bool = True) -> None:
        """Set active class filter (`all`, `pos`, `neg`, `unlabeled`, or `class:<Name>`)."""
        key = str(filt_key or "all")
        if self._report is not None and key.startswith("class:"):
            cls_name = key.split(":", 1)[1]
            if cls_name == MULTICLASS_NONE_BEHAVIOR:
                key = "neg"
            else:
                include_all_mc = bool(
                    getattr(self._report, "include_all_multiclass", False)
                    or self.behavior_combo.currentData() == "__ALL_MC__"
                )
                if not include_all_mc:
                    beh = self._selected_single_behavior() or self._report.behavior
                    if cls_name == beh:
                        key = "pos"
                    elif cls_name == f"Not-{beh}":
                        key = "neg"

        scope_changed = False
        if (
            self._mode == "subtypes"
            and key == "neg"
            and self._effective_subtype_scope() == "pos_only"
        ):
            self._subtypes_scope = "all_labeled"
            idx_all = self.subtypes_scope_combo.findData("all_labeled")
            if idx_all >= 0:
                self.subtypes_scope_combo.blockSignals(True)
                self.subtypes_scope_combo.setCurrentIndex(idx_all)
                self.subtypes_scope_combo.blockSignals(False)
            if self._report is not None:
                target_space = self._effective_space()
                self._report = recompute_embedding_only(
                    self._report,
                    space=target_space,
                    subtype_scope="all_labeled",
                    feature_lens=self._effective_feature_lens(),
                    target_behavior=self._effective_subtype_target_behavior(),
                    compute_tsne=(self._proj_method == "tsne"),
                    compute_mds=(target_space == "rf_proximity" or self._proj_method == "mds"),
                    forced_k=self._effective_forced_k(),
                )
                self._repr_matrix_cache = None
                self._sync_subtypes_bar_from_report()
                scope_changed = True

        self._class_filter = key
        self._refresh_class_filter_combos()
        if redraw:
            self._populate_table(redraw_scatter_on_change=False)
            self._draw_scatter()
            if scope_changed:
                self._update_inspector_card()

    def _on_class_filter_combo_changed(self, idx: int = -1) -> None:
        sender = self.sender()
        combo = sender if isinstance(sender, QtWidgets.QComboBox) else self.class_filter_combo
        if idx < 0:
            idx = combo.currentIndex()
        key = str(combo.itemData(idx) or "all")
        if key != self._class_filter:
            self._set_class_filter(key, redraw=True)

    def _set_inspector_view(self, view: InspectorViewMode, *, redraw: bool = True) -> None:
        self._inspector_view = view
        self.view_prob_btn.setChecked(view == "prob")
        self.view_feat_btn.setChecked(view == "feat")
        self.view_pose_btn.setChecked(view == "pose")
        self.view_clusters_btn.setChecked(view == "clusters")
        self.insp_canvas._interactive_x_scrub = view == "prob"
        self.insp_canvas._interactive_y_pick = view == "clusters"
        if redraw:
            self._update_inspector_card()

    def _on_inspector_view_pill_clicked(self, pill_id: int) -> None:
        view: InspectorViewMode = (
            "prob"
            if pill_id == 0
            else ("feat" if pill_id == 1 else ("pose" if pill_id == 2 else "clusters"))
        )
        self._set_inspector_view(view, redraw=True)

    # ------------------------------------------------------------------
    # Full Background Cross-Validation Analysis (`refresh_analysis`)
    # ------------------------------------------------------------------
    def refresh_analysis(self) -> None:
        """Run `run_bale` off the GUI thread and refresh all views."""
        self._init_refresh_timer.stop()
        if self._is_closing:
            return
        if self._spatial_worker is not None and hasattr(self._spatial_worker, "cancel_event"):
            self._spatial_worker.cancel_event.set()
        self._spatial_worker = None
        self._spatial_request_id += 1
        self._video_report_key = None
        if self._mode == "videos":
            self._analysis_needed_on_bout_mode = True
            self.refresh_video_summary(force=True)
            return
        self._analysis_needed_on_bout_mode = False
        if self._active_worker is not None:
            self._queued_refresh = True
            self.status_summary_label.setText(
                "Queued refresh — updating as soon as current scan finishes…"
            )
            return
        sm = self._project.settings_manager
        behaviors = list(sm.behavior_names)
        if not behaviors:
            self.status_summary_label.setText("No behaviors configured in this JABS project.")
            return

        beh_data = self.behavior_combo.currentData()
        include_all_mc = beh_data == "__ALL_MC__"
        active_beh = behaviors[0] if include_all_mc else str(beh_data or behaviors[0])
        beh_label = "All Classes" if include_all_mc else active_beh

        if (
            self._report is not None
            and self._selected_bout_idx is not None
            and 0 <= self._selected_bout_idx < len(self._report.feature_matrix.records)
        ):
            sel_rec = self._report.feature_matrix.records[self._selected_bout_idx]
            self._reselect_bout_key_on_finish = (
                sel_rec.video,
                int(sel_rec.identity),
                int(sel_rec.start_frame),
                int(sel_rec.end_frame),
            )
        else:
            self._reselect_bout_key_on_finish = None

        include_preds = self._mode == "predict" or self._report_includes_predictions
        space = self._effective_space()

        self.recalc_btn.setEnabled(False)
        self.recalc_btn.setText("Computing…")
        self.status_summary_label.setText(f"Analyzing {beh_label}…")

        worker = _ExplorerComputeTask(
            self._project,
            active_beh,
            include_positive=True,
            include_negative=True,
            include_all_multiclass=include_all_mc,
            include_predictions=include_preds,
            max_predicted_bouts=self._max_pred_bouts,
            space=space,
            subtype_scope=self._effective_subtype_scope(),
            feature_lens=self._effective_feature_lens(),
            target_behavior=self._effective_subtype_target_behavior(),
            forced_k=self._effective_forced_k(),
            compute_tsne=(self._proj_method == "tsne"),
            compute_mds=(space == "rf_proximity" or self._proj_method == "mds"),
            compute_spatial=(self._proj_method == "spatial"),
        )
        self._active_worker = worker
        worker.signals.finished.connect(self._on_analysis_finished)
        worker.signals.error.connect(self._on_analysis_error)
        self._thread_pool.start(worker)

    def _on_analysis_finished(self, report: BaleReport, included_preds: bool) -> None:
        if self._is_closing:
            return
        self._init_refresh_timer.stop()
        self._active_worker = None
        if self._queued_refresh:
            self._queued_refresh = False
            self.refresh_analysis()
            return

        target_space = self._effective_space()
        target_scope = self._effective_subtype_scope()
        target_lens = self._effective_feature_lens()
        target_beh = self._effective_subtype_target_behavior()
        target_k = self._effective_forced_k()
        need_tsne = self._proj_method == "tsne" and report.embedding.coords_tsne_2d is None
        need_mds = (
            target_space == "rf_proximity" or self._proj_method == "mds"
        ) and report.embedding.coords_mds_2d is None
        if (
            report.space != target_space
            or report.embedding.subtype_scope != target_scope
            or report.embedding.feature_lens != target_lens
            or report.embedding.target_behavior != target_beh
            or report.embedding.forced_k != target_k
            or need_tsne
            or need_mds
        ):
            report = recompute_embedding_only(
                report,
                space=target_space,
                subtype_scope=target_scope,
                feature_lens=target_lens,
                target_behavior=target_beh,
                compute_tsne=self._proj_method == "tsne",
                compute_mds=(target_space == "rf_proximity" or self._proj_method == "mds"),
                forced_k=target_k,
            )

        self._report = report
        self._sync_subtypes_bar_from_report()
        self._report_includes_predictions = included_preds
        self._examined_store.load()
        self._reviewed_bout_ids.clear()
        self._examined_only_bout_ids.clear()
        self._pinned_needs_review_bout_ids.clear()
        self._deleted_bout_ids.clear()
        self._unlabeled_bout_ids.clear()
        self._saved_notes_by_bout.clear()
        for r in report.feature_matrix.records:
            self._ensure_origin_snapshot(r)
            persisted = self._examined_store.find_match(r)
            if persisted is not None:
                self._reviewed_bout_ids.add(r.bout_id)
                act_kind = str(persisted.get("action") or "examined")
                note_txt = str(persisted.get("note") or "✓ Examined")
                self._saved_notes_by_bout[r.bout_id] = note_txt
                if act_kind == "examined":
                    self._examined_only_bout_ids.add(r.bout_id)
                elif act_kind == "no_label" and r.source == "predicted":
                    self._unlabeled_bout_ids.add(r.bout_id)
        self._orig_boundary_flags = {
            r.bout_id: (
                report.diagnostics.suggested_trim_bounds[i] is not None,
                report.diagnostics.suggested_split_frame[i] is not None,
            )
            for i, r in enumerate(report.feature_matrix.records)
        }
        reselect_key = self._reselect_bout_key_on_finish
        self._reselect_bout_key_on_finish = None
        if reselect_key is not None:
            matched_idx: int | None = None
            for i, r in enumerate(report.feature_matrix.records):
                if (
                    r.video,
                    int(r.identity),
                    int(r.start_frame),
                    int(r.end_frame),
                ) == reselect_key:
                    matched_idx = i
                    break
            self._selected_bout_idx = matched_idx
        self._ui_undo_history.clear()
        self._undo_stack = BaleUndoStack()
        self.undo_btn.setEnabled(False)
        self._pending_recalc_count = 0
        self._external_labels_dirty = False
        self._repr_matrix_cache = None
        self._last_card_bout_idx = None

        self._update_cv_badge_from_counts(
            report.cv_groups_count, report.jabs_threshold_met, report.dual_class_cv_met
        )
        # If user is currently viewing 4. Videos, preserve Mode 4 zoom, status & Rescan button!
        if self._mode == "videos":
            self._pending_select_video_after_refresh = None
            self._update_recalc_button_badge()
            return

        self._clear_custom_selection_state()
        self._scatter_view_limits = None
        self._scatter_default_limits = None
        self._sync_zoom_reset_button()
        self._refresh_class_filter_combos()
        self.recalc_btn.setEnabled(True)
        self._update_recalc_button_badge()

        # If Audit Labels opened on a behavior with 0 human labels, switch smoothly to Review Predictions
        n_labeled = sum(1 for r in report.feature_matrix.records if r.source == "labeled")
        if n_labeled == 0 and self._mode == "audit" and not included_preds:
            self.mode_predict_btn.setChecked(True)
            self._on_mode_pill_clicked(1)
            return

        pending_vid = self._pending_select_video_after_refresh
        self._pending_select_video_after_refresh = None
        if pending_vid is not None:
            self._apply_video_name_cohort_in_bout_mode(pending_vid)
            return

        self._populate_table(auto_fallback_if_clean=True, redraw_scatter_on_change=False)
        if self._proj_method == "spatial" and self._report.spatial is None:
            self._request_spatial_computation()
        self._draw_scatter()

        n_tot = len(self._mode_eligible_indices())
        total_ms = report.timings_ms.get("total_ms", 0.0)
        scope_lbl = self._active_scope_label()
        if scope_lbl:
            self.status_summary_label.setText(
                f"Loaded {n_tot} {scope_lbl} bouts in {total_ms:.0f} ms"
            )
        else:
            self.status_summary_label.setText(f"Loaded {n_tot} bouts in {total_ms:.0f} ms")

    def _on_analysis_error(self, err_msg: str) -> None:
        if self._is_closing:
            return
        self._active_worker = None
        if self._queued_refresh:
            self._queued_refresh = False
            self.refresh_analysis()
            return
        self._update_recalc_button_badge()
        if self._mode == "videos":
            return
        self.status_summary_label.setText(f"Analysis failed: {err_msg}")
        MessageDialog.error(self, title="BALE Error", message=err_msg)

    def _update_recalc_button_badge(self) -> None:
        if self._mode == "videos":
            self.recalc_btn.setToolTip("Rescan whole-video predictions & labels from disk [R]")
            if self._video_worker is not None:
                self.recalc_btn.setEnabled(False)
                self.recalc_btn.setText("Scanning…")
                self.recalc_btn.setStyleSheet("")
            elif self._external_labels_dirty:
                self.recalc_btn.setEnabled(True)
                self.recalc_btn.setText("↻ Rescan (Updated)")
                self.recalc_btn.setStyleSheet(
                    "QPushButton { background-color: #DBEAFE; color: #1E40AF; font-weight: 700; "
                    "padding: 4px 10px; border-radius: 5px; border: 1px solid #93C5FD; }"
                )
            else:
                self.recalc_btn.setEnabled(True)
                self.recalc_btn.setText("↻ Rescan")
                self.recalc_btn.setStyleSheet("")
            return

        self.recalc_btn.setToolTip(
            "Re-run cross-validation diagnostic scoring & 2D map from current disk labels [R]"
        )
        if self._active_worker is not None:
            self.recalc_btn.setEnabled(False)
            self.recalc_btn.setText("Computing…")
            self.recalc_btn.setStyleSheet("")
            return
        self.recalc_btn.setEnabled(True)
        if self._pending_recalc_count > 0 or self._external_labels_dirty:
            tag = (
                f" ({self._pending_recalc_count})"
                if self._pending_recalc_count > 0
                else " (Updated)"
            )
            self.recalc_btn.setText(f"↻ Recalculate{tag}")
            self.recalc_btn.setStyleSheet(
                "QPushButton { background-color: #DBEAFE; color: #1E40AF; font-weight: 700; "
                "padding: 4px 10px; border-radius: 5px; border: 1px solid #93C5FD; }"
            )
        else:
            self.recalc_btn.setText("↻ Recalculate")
            self.recalc_btn.setStyleSheet("")

    def _update_cv_badge_from_counts(
        self, cv_groups: int, jabs_met: bool, dual_class_cv_met: bool
    ) -> None:
        if dual_class_cv_met:
            self.cv_badge.setText(f"✓ {cv_groups} CV Tracks")
            self.cv_badge.setStyleSheet(
                "padding: 4px 9px; border-radius: 5px; background-color: #DCFCE7; "
                "color: #166534; font-weight: 700;"
            )
        elif jabs_met:
            self.cv_badge.setText(f"⚠ {cv_groups}/15 CV Tracks")
            self.cv_badge.setStyleSheet(
                "padding: 4px 9px; border-radius: 5px; background-color: #FEF3C7; "
                "color: #92400E; font-weight: 700;"
            )
        else:
            self.cv_badge.setText(f"✗ {cv_groups} CV Tracks")
            self.cv_badge.setStyleSheet(
                "padding: 4px 9px; border-radius: 5px; background-color: #FEE2E2; "
                "color: #991B1B; font-weight: 700;"
            )
        self.cv_badge.setToolTip(
            f"{cv_groups} animal tracks contain both ≥10 Positive and ≥10 Negative frames.\n"
            "JABS requires ≥2 tracks to Train; ≥15 dual-class tracks recommended (Dual-Class CV)."
        )

    # ------------------------------------------------------------------
    # Plain-English Bout Diagnosis & Queue Scoring
    # ------------------------------------------------------------------
    def _selected_single_behavior(self) -> str | None:
        """Return concrete behavior selected in `behavior_combo`, or None for All Classes."""
        chosen = self.behavior_combo.currentData()
        if isinstance(chosen, str) and chosen and chosen != "__ALL_MC__":
            return chosen
        return None

    def _active_scope_label(self) -> str:
        """Return human-readable scope label ('All Classes' or active behavior name)."""
        if self._report is None:
            return self._selected_single_behavior() or "All Classes"
        if (
            getattr(self._report, "include_all_multiclass", False)
            or self.behavior_combo.currentData() == "__ALL_MC__"
        ):
            return "All Classes"
        return self._report.behavior

    def _bout_display_behavior(self, rec: BoutRecord) -> str:
        """Return most informative non-'None' behavior name for `rec`."""
        if rec.behavior and rec.behavior != MULTICLASS_NONE_BEHAVIOR:
            return rec.behavior
        orig_beh = getattr(rec, "origin_behavior", "") or ""
        if orig_beh and orig_beh != MULTICLASS_NONE_BEHAVIOR:
            return orig_beh
        return self._report.behavior if self._report is not None else ""

    def _bout_is_label_halo(self, b_idx: int) -> bool:
        if self._report is None:
            return False
        rec = self._report.feature_matrix.records[b_idx]
        return bool(
            getattr(rec, "origin_source", rec.source) == "predicted"
            and getattr(rec, "is_label_halo", False)
        )

    def _bout_needs_review(self, b_idx: int) -> bool:
        if self._report is None:
            return False
        rec = self._report.feature_matrix.records[b_idx]
        if rec.bout_id in self._deleted_bout_ids or rec.bout_id in self._reviewed_bout_ids:
            return False
        diag = self._report.diagnostics
        if self._mode == "predict":
            # Standalone predicted bouts stay cleanly separated from Label Edge (Halo) fragments
            return rec.source == "predicted" and not bool(getattr(rec, "is_label_halo", False))
        if self._mode == "subtypes":
            return rec.source == "labeled" and (
                int(diag.outlier_votes[b_idx]) >= 3 or float(diag.outlier_scores[b_idx]) >= 0.78
            )
        # Audit Labels mode
        if rec.source != "labeled":
            return False
        return (
            float(diag.confusion_scores[b_idx]) >= 0.55
            or (
                diag.suggested_trim_bounds[b_idx] is not None
                and float(diag.transition_scores[b_idx]) >= 0.50
            )
            or (
                diag.suggested_split_frame[b_idx] is not None
                and float(diag.transition_scores[b_idx]) >= 0.50
            )
            or int(diag.outlier_votes[b_idx]) >= 4
        )

    def _bout_priority_score(self, b_idx: int) -> float:
        if self._report is None:
            return 0.0
        rec = self._report.feature_matrix.records[b_idx]
        diag = self._report.diagnostics
        if self._mode == "predict":
            # Highest doubt / disagreement standalone predictions first; Label Edge halos rank cleanly below
            base_score = float(
                0.70 * diag.confusion_scores[b_idx]
                + 0.20 * diag.knn_disagreement[b_idx]
                + 0.10 * rec.postproc_fragility
            )
            if bool(getattr(rec, "is_label_halo", False)):
                return -0.50 + 0.40 * base_score
            return base_score
        if self._mode == "subtypes":
            return float(diag.outlier_scores[b_idx])
        has_trim, has_split = self._orig_boundary_flags.get(
            rec.bout_id,
            (
                diag.suggested_trim_bounds[b_idx] is not None,
                diag.suggested_split_frame[b_idx] is not None,
            ),
        )
        return float(
            max(
                diag.confusion_scores[b_idx],
                0.88 * diag.transition_scores[b_idx] if (has_trim or has_split) else 0.0,
                0.75 * diag.outlier_scores[b_idx] if int(diag.outlier_votes[b_idx]) >= 4 else 0.0,
            )
        )

    def _describe_bout_issue(self, b_idx: int) -> tuple[str, str, str, str]:
        """Return `(issue_badge, suggested_fix, plain_diagnosis, accent_hex)` for `b_idx`."""
        assert self._report is not None
        rec = self._report.feature_matrix.records[b_idx]
        diag = self._report.diagnostics
        emb = self._report.embedding
        beh = rec.behavior if (rec.behavior and rec.behavior != "None") else self._report.behavior
        opp_name = "None" if self._report.is_multiclass else f"Not-{beh}"

        if (
            rec.bout_id in self._saved_notes_by_bout
            and rec.bout_id not in self._examined_only_bout_ids
        ):
            saved_txt = self._saved_notes_by_bout[rec.bout_id]
            if rec.bout_id in self._unlabeled_bout_ids or rec.source == "unlabeled":
                return (
                    "○ No Label",
                    saved_txt,
                    f"{saved_txt}. Saved as examined for this project — press [V] to unmark or [Z]/[C] to label.",
                    "#475569",
                )
            return (
                "✓ Saved",
                saved_txt,
                f"{saved_txt}. Annotation saved to disk and synced with the JABS timeline.",
                "#15803D",
            )

        conf = float(diag.confusion_scores[b_idx])
        knn_dis = float(diag.knn_disagreement[b_idx])
        trans = float(diag.transition_scores[b_idx])
        trim = diag.suggested_trim_bounds[b_idx]
        split_f = diag.suggested_split_frame[b_idx]
        votes = int(diag.outlier_votes[b_idx])
        cid = int(emb.cluster_labels[b_idx])
        top_feats = (
            emb.top_z_diffs_per_bout[b_idx] if b_idx < len(emb.top_z_diffs_per_bout) else []
        )
        feat_snippet = (
            ", ".join(f"{n} ({z:+.1f} SD)" for n, z in top_feats[:2])
            if top_feats
            else "core features"
        )
        knn_classes = getattr(diag, "knn_majority_class", None) or []
        oof_classes = getattr(diag, "oof_top_class", None) or []
        knn_cls = knn_classes[b_idx] if b_idx < len(knn_classes) else ""
        oof_cls = oof_classes[b_idx] if b_idx < len(oof_classes) else ""
        knn_alt = (
            knn_cls
            if (self._report.is_multiclass and knn_cls and knn_cls != rec.class_name)
            else ""
        )
        oof_alt = (
            oof_cls
            if (self._report.is_multiclass and oof_cls and oof_cls != rec.class_name)
            else ""
        )

        if rec.source == "predicted":
            mean_p = float(diag.oof_mean_probs[b_idx])
            if bool(getattr(rec, "is_label_halo", False)):
                side = str(getattr(rec, "halo_side", "") or "")
                side_tag = (
                    "Onset"
                    if side == "before"
                    else (
                        "Offset"
                        if side == "after"
                        else ("Bridge" if side == "between" else "Halo")
                    )
                )
                h_dist = getattr(rec, "halo_distance_frames", None)
                dist_txt = (
                    "touching an existing label"
                    if h_dist == 0
                    else (
                        f"{int(h_dist)}f from an existing label"
                        if h_dist is not None
                        else "adjacent to an existing label"
                    )
                )
                return (
                    f"Label Edge ({side_tag})",
                    "Extend [Z], Skip [X], or Examined [V]",
                    f"Predicted boundary/halo fragment ({rec.duration_frames}f / {rec.duration_frames / 30.0:.2f}s, "
                    f"{dist_txt}, mean P={mean_p:.2f}). Useful for refining bout beginnings & ends — "
                    f"press [Z] to extend {rec.class_name}, [X] for No Label, [C] for {opp_name}, or [V] to mark Examined.",
                    "#7C3AED",
                )
            if conf >= 0.65 and knn_dis >= 0.50:
                alt_knn = knn_alt or opp_name
                fix_txt = (
                    f"Mark {opp_name} [C]"
                    if alt_knn == opp_name
                    else f"Assign {alt_knn} [Z] or {opp_name} [C]"
                )
                return (
                    f"False Pos? ({round(conf * 100)}%)",
                    fix_txt,
                    f"Suspect prediction \u2014 predicted {rec.class_name} (mean P={mean_p:.2f}), "
                    f"but {round(knn_dis * 7)}/7 nearest human-labeled bouts are {alt_knn}.",
                    "#B91C1C",
                )
            if rec.postproc_fragility >= 0.35:
                return (
                    "Stitched Gap",
                    "Play Clip & Confirm [Z] or Split",
                    f"Post-processing stitched across raw frame dropouts ({int(rec.postproc_fragility * 100)}% gap frames). "
                    "Play clip [Space] to verify whether this is one continuous bout.",
                    "#D97706",
                )
            if float(diag.ood_scores[b_idx]) >= 0.85:
                return (
                    "Novel / Unseen",
                    "Confirm [Z], No Label [X], or Reject [C]",
                    f"Unusual kinematics outside current training coverage ({feat_snippet}). "
                    "Labeling this bout will expand classifier generalization.",
                    "#D97706",
                )
            alt_pred = oof_alt or knn_alt
            uncertain_diag = (
                f"Predicted {rec.class_name} (mean P={mean_p:.2f}; nearest class: {alt_pred}). "
                f"Watch clip [Space] and press [Z] to confirm/assign, [X] for no label, or [C] to mark as {opp_name}."
                if (alt_pred and alt_pred != opp_name)
                else f"Predicted {rec.class_name} (mean P={mean_p:.2f}). "
                "Watch clip [Space] and press [Z] to confirm, [X] for no label, or [C] to mark as hard negative."
            )
            return (
                f"Uncertain ({round(conf * 100)}%)",
                f"Confirm {rec.class_name} [Z], Skip [X], or Not [C]",
                uncertain_diag,
                "#1E40AF",
            )

        # Labeled bout in Sub-Types mode
        if self._mode == "subtypes":
            c_feats = emb.cluster_top_features.get(cid, [])
            c_desc = (
                ", ".join(f"{_short_feature_label(n, 20)} ({z:+.1f} SD)" for n, z in c_feats[:2])
                if c_feats
                else feat_snippet
            )
            if votes >= 3 or float(diag.outlier_scores[b_idx]) >= 0.78:
                return (
                    f"Outlier ({votes}/5) • C{cid}",
                    "Play Clip to Inspect",
                    f"Kinematic outlier ({votes}/5 metrics) in Sub-Type Cluster C{cid}. "
                    f"Deviates most on {feat_snippet}.",
                    "#D97706",
                )
            return (
                f"Cluster C{cid}",
                f"Sub-type signature: {c_desc[:42]}",
                f"Assigned to behavioral Sub-Type Cluster C{cid} (of {emb.optimal_k}). "
                f"Defining kinematics: {c_desc}.",
                "#0F766E",
            )

        # Labeled bout in Audit mode
        mc_alt = oof_alt or knn_alt
        if conf >= 0.65:
            target_lbl = mc_alt if mc_alt else (opp_name if rec.is_positive else beh)
            target_key = "C" if target_lbl == opp_name else "Z"
            fix_hint = (
                f"Assign {target_lbl} [Z] or {opp_name} [C]"
                if (self._report.is_multiclass and rec.is_positive and target_lbl != opp_name)
                else f"Switch to {target_lbl} [{target_key}]"
            )
            return (
                f"Likely Mislabel ({round(conf * 100)}%)",
                fix_hint,
                f"Likely mislabeled ({round(conf * 100)}% disagreement) \u2014 out-of-fold classifier "
                f"and {round(knn_dis * 7)}/7 similar bouts indicate {target_lbl}. "
                f"{fix_hint}, clear [X], or keep {rec.class_name}.",
                "#B91C1C",
            )
        if trim is not None and trans >= 0.50:
            keep_key = "Z" if rec.is_positive else "C"
            return (
                "Trim Boundary",
                f"Trim to {trim[0]}\u2013{trim[1]} [T]",
                f"Sloppy start/end boundary \u2014 frames outside {trim[0]}\u2013{trim[1]} match the opposite state. "
                f"Press [T] to trim cleanly to {trim[0]}\u2013{trim[1]}, or [{keep_key}] to keep as-is.",
                "#D97706",
            )
        if split_f is not None and trans >= 0.50:
            keep_key = "Z" if rec.is_positive else "C"
            return (
                "Chimeric Bout",
                f"Split @ frame {split_f}",
                f"Merged two-state bout \u2014 behavior shifts sharply near frame {split_f}. "
                f"Click Split to separate at frame {split_f}, or [{keep_key}] to keep as-is.",
                "#D97706",
            )
        if conf >= 0.55:
            target_lbl = mc_alt if mc_alt else (opp_name if rec.is_positive else beh)
            return (
                f"Borderline ({round(conf * 100)}%)",
                f"Keep {rec.class_name} or Switch ({target_lbl})",
                f"Borderline grey-zone bout ({round(conf * 100)}% classifier doubt; nearest class: {target_lbl}). "
                f"Play clip [Space] to keep {rec.class_name} or switch to {target_lbl}.",
                "#D97706",
            )
        if votes >= 4:
            keep_key = "Z" if rec.is_positive else "C"
            return (
                f"Outlier ({votes}/5)",
                f"Keep [{keep_key}] after verifying clip",
                f"Unusual posture/kinematics ({votes}/5 outlier votes: {feat_snippet}), "
                f"though classifier agrees with the label. Play clip [Space] and press [{keep_key}] to keep.",
                "#D97706",
            )
        return (
            "Clean",
            "No change needed",
            f"Consistent {rec.class_name} bout ({round((1.0 - conf) * 100)}% out-of-fold agreement, Cluster C{cid}).",
            "#0F766E",
        )

    # ------------------------------------------------------------------
    # Filtering & Table Population
    # ------------------------------------------------------------------
    def _mode_eligible_indices(self) -> list[int]:
        """Indices eligible for the active workflow's triage queue."""
        if self._report is None:
            return []
        records = self._report.feature_matrix.records
        single_beh = self._selected_single_behavior()
        out: list[int] = []
        for i, r in enumerate(records):
            if r.bout_id in self._deleted_bout_ids:
                continue
            if self._mode == "predict":
                # Only predicted bouts (including predictions promoted in-session) belong here
                if getattr(r, "origin_source", r.source) != "predicted":
                    continue
                if single_beh is not None:
                    orig_beh = getattr(r, "origin_behavior", "") or r.behavior
                    if orig_beh != single_beh:
                        continue
                out.append(i)
            else:
                if r.source != "labeled":
                    continue
                if (
                    single_beh is not None
                    and self._report.is_multiclass
                    and r.behavior not in (single_beh, MULTICLASS_NONE_BEHAVIOR)
                ):
                    continue
                out.append(i)
        if self._mode == "subtypes" and self._effective_subtype_scope() == "pos_only" and out:
            pos_out = [i for i in out if records[i].is_positive]
            if pos_out:
                return pos_out
        return out

    def _bout_matches_class_filter(self, b_idx: int) -> bool:
        """Return whether bout index `b_idx` satisfies the active `self._class_filter`."""
        if self._report is None or self._class_filter == "all":
            return True
        rec = self._report.feature_matrix.records[b_idx]
        is_unlbl = self._is_bout_unlabeled(rec, self._unlabeled_bout_ids)
        if self._class_filter == "unlabeled":
            return is_unlbl
        if is_unlbl:
            return False
        if self._class_filter == "pos":
            return bool(rec.is_positive)
        if self._class_filter == "neg":
            return not bool(rec.is_positive)
        if self._class_filter.startswith("class:"):
            target_cls = self._class_filter.split(":", 1)[1]
            return rec.class_name == target_cls
        return True

    def _class_filtered_indices(self) -> list[int]:
        """Return mode-eligible bout indices restricted by the active Class Filter."""
        eligible = self._mode_eligible_indices()
        if self._class_filter == "all":
            return eligible
        return [i for i in eligible if self._bout_matches_class_filter(i)]

    def _bout_matches_search_query(self, b_idx: int, q: str) -> bool:
        """Return whether bout index `b_idx` matches lowercase search query `q` without `Not-X` false hits."""
        if not q or self._report is None:
            return True
        r = self._report.feature_matrix.records[b_idx]
        emb = self._report.embedding
        anim_txt = self._animal_display_text(r.video, r.identity).lower()
        anim_num = anim_txt.removeprefix("animal ").strip()
        if (
            q in r.video.lower()
            or q in _short_video_name(r.video).lower()
            or q in anim_txt
            or q == anim_num
            or q == str(r.identity)
            or q in f"c{int(emb.cluster_labels[b_idx])}"
        ):
            return True

        is_unlbl = self._is_bout_unlabeled(r, self._unlabeled_bout_ids)
        if is_unlbl:
            return q in ("no label", "unlabeled")
        cls_l = r.class_name.lower()
        if cls_l.startswith("not-"):
            return (
                (q.startswith("not") and q in cls_l)
                or "not-" in q
                or q in ("neg", "negative", "-")
            )
        if r.class_name == MULTICLASS_NONE_BEHAVIOR:
            return q in "none" or q in ("neg", "negative", "-")
        return q in cls_l or q in ("pos", "positive", "+")

    def _scatter_backdrop_indices(self) -> list[int]:
        """Non-interactive human-labeled context dots plotted behind predictions in Predict mode."""
        if (
            self._report is None
            or self._mode != "predict"
            or not self.show_pred_backdrop_action.isChecked()
            or self._lasso_selected_indices
            or self._class_filter == "unlabeled"
        ):
            return []
        records = self._report.feature_matrix.records
        single_beh = self._selected_single_behavior()
        q = self.search_input.text().strip().lower()
        out: list[int] = []
        for i, r in enumerate(records):
            if r.bout_id in self._deleted_bout_ids:
                continue
            if getattr(r, "origin_source", r.source) != "labeled":
                continue
            if (
                single_beh is not None
                and self._report.is_multiclass
                and r.behavior not in (single_beh, MULTICLASS_NONE_BEHAVIOR)
            ):
                continue
            if self._class_filter == "pos" and not r.is_positive:
                continue
            if self._class_filter == "neg" and r.is_positive:
                continue
            if (
                self._class_filter.startswith("class:")
                and r.class_name != self._class_filter.split(":", 1)[1]
            ):
                continue
            if q and not self._bout_matches_search_query(i, q):
                continue
            out.append(i)
        return out

    def _scatter_plot_indices(self) -> list[int]:
        """Indices plotted on the 2D scatter map synchronized with class, cohort, and search filters."""
        if self._report is None:
            return []
        if self.sync_plot_to_queue_tab_action.isChecked():
            return self._filtered_bout_indices()
        vis = self._class_filtered_indices()
        if self._lasso_selected_indices:
            cohort_set = set(self._lasso_selected_indices)
            vis = [i for i in vis if i in cohort_set]
        q = self.search_input.text().strip().lower()
        if q:
            vis = [i for i in vis if self._bout_matches_search_query(i, q)]
        return vis

    def _scatter_visible_indices(self) -> list[int]:
        """Indices plotted on the 2D scatter map (includes non-pickable labeled backdrop in Predict mode)."""
        if self._report is None:
            return []
        vis = self._scatter_plot_indices()
        if self._mode == "predict":
            return vis + self._scatter_backdrop_indices()
        return vis

    def _filtered_bout_indices(self) -> list[int]:
        eligible = self._class_filtered_indices()
        if not eligible or self._report is None:
            return []

        if self._lasso_selected_indices:
            eligible_set = set(eligible)
            pool = [i for i in self._lasso_selected_indices if i in eligible_set]
        elif self._queue_filter == "needs_review":
            records = self._report.feature_matrix.records
            pool = [
                i
                for i in eligible
                if self._bout_needs_review(i)
                or (
                    records[i].bout_id in self._pinned_needs_review_bout_ids
                    and records[i].bout_id not in self._deleted_bout_ids
                )
            ]
        elif self._queue_filter == "label_edges":
            records = self._report.feature_matrix.records
            pool = [
                i
                for i in eligible
                if self._bout_is_label_halo(i)
                and (
                    records[i].bout_id not in self._reviewed_bout_ids
                    or (
                        records[i].bout_id in self._pinned_needs_review_bout_ids
                        and records[i].bout_id not in self._deleted_bout_ids
                    )
                )
            ]
        elif self._queue_filter == "reviewed":
            pool = [
                i
                for i in eligible
                if self._report.feature_matrix.records[i].bout_id in self._reviewed_bout_ids
            ]
        else:
            pool = list(eligible)

        q = self.search_input.text().strip().lower()
        if q:
            pool = [i for i in pool if self._bout_matches_search_query(i, q)]

        if not self._lasso_selected_indices:
            pool.sort(key=self._bout_priority_score, reverse=True)
        return pool

    def _update_queue_filter_counts(self) -> None:
        self._refresh_class_filter_combos()
        eligible = self._class_filtered_indices()
        if self._report is None:
            self.qf_needs_btn.setText("Needs Review (0)")
            self.qf_halos_btn.setText("Label Edges (0)")
            self.qf_all_btn.setText("All Bouts (0)")
            self.qf_done_btn.setText("Reviewed ✓ (0)")
            return
        records = self._report.feature_matrix.records
        n_needs = sum(1 for i in eligible if self._bout_needs_review(i))
        n_halos = sum(
            1
            for i in eligible
            if self._bout_is_label_halo(i) and records[i].bout_id not in self._reviewed_bout_ids
        )
        n_done = sum(1 for i in eligible if records[i].bout_id in self._reviewed_bout_ids)
        self.qf_needs_btn.setText(f"Needs Review ({n_needs})")
        self.qf_halos_btn.setText(f"Label Edges ({n_halos})")
        self.qf_all_btn.setText(f"All Bouts ({len(eligible)})")
        self.qf_done_btn.setText(f"Reviewed ✓ ({n_done})")

    def _populate_table_row(
        self,
        row_idx: int,
        b_idx: int,
        *,
        records: list[BoutRecord],
        cmap: dict[str, str],
    ) -> None:
        """Populate or refresh a single table row in-place with typed sort keys (`< 0.2 ms`)."""
        rec = records[b_idx]
        issue_badge, suggested_fix, plain_diag, accent_hex = self._describe_bout_issue(b_idx)
        prio_score = float(self._bout_priority_score(b_idx))

        # Col 0: Priority / Issue (UserRole = b_idx; _SORT_KEY_ROLE = numeric priority)
        it_issue = _SortableTableItem(issue_badge, sort_key=(prio_score, -int(rec.bout_id)))
        it_issue.setData(Qt.ItemDataRole.UserRole, b_idx)
        it_issue.setToolTip(plain_diag)
        if rec.bout_id in self._unlabeled_bout_ids or rec.source == "unlabeled":
            it_issue.setBackground(QtGui.QColor("#F1F5F9"))
            it_issue.setForeground(QtGui.QColor("#475569"))
        elif (
            rec.bout_id in self._reviewed_bout_ids
            and rec.bout_id not in self._examined_only_bout_ids
        ):
            it_issue.setBackground(QtGui.QColor("#DCFCE7"))
            it_issue.setForeground(QtGui.QColor("#166534"))
        elif accent_hex == "#7C3AED":
            it_issue.setBackground(QtGui.QColor("#F3E8FF"))
            it_issue.setForeground(QtGui.QColor("#5B21B6"))
        elif accent_hex == "#B91C1C":
            it_issue.setBackground(QtGui.QColor("#FEE2E2"))
            it_issue.setForeground(QtGui.QColor("#991B1B"))
        elif accent_hex == "#D97706":
            it_issue.setBackground(QtGui.QColor("#FEF3C7"))
            it_issue.setForeground(QtGui.QColor("#92400E"))
        self.table.setItem(row_idx, 0, it_issue)

        # Col 1: Current State with matching JABS color symbol
        origin_src = getattr(rec, "origin_source", rec.source)
        if rec.bout_id in self._unlabeled_bout_ids or rec.source == "unlabeled":
            state_txt = "○ No Label"
            dot_hex = "#475569"
            state_sort = (2, "no label", int(rec.bout_id))
        elif self._mode == "predict" and origin_src == "predicted" and rec.source == "labeled":
            state_txt = (
                f"✓ Confirmed: {rec.class_name}"
                if rec.is_positive
                else f"✗ Rejected → {rec.class_name}"
            )
            dot_hex = cmap.get(rec.class_name, "#1E40AF" if rec.is_positive else "#377EB8")
            state_sort = (1, rec.class_name.lower(), int(rec.bout_id))
        else:
            sym = "◆" if rec.source == "predicted" else "●"
            src_prefix = "Pred" if rec.source == "predicted" else "Labeled"
            state_txt = f"{sym} {src_prefix}: {rec.class_name}"
            dot_hex = cmap.get(rec.class_name, "#1E40AF" if rec.is_positive else "#377EB8")
            state_sort = (
                0 if rec.source == "predicted" else 1,
                rec.class_name.lower(),
                int(rec.bout_id),
            )
        it_state = _SortableTableItem(state_txt, sort_key=state_sort)
        it_state.setForeground(QtGui.QColor(dot_hex))
        font_bold = it_state.font()
        font_bold.setBold(True)
        it_state.setFont(font_bold)
        self.table.setItem(row_idx, 1, it_state)

        # Col 2: Scannable Video Name
        short_vid = _short_video_name(rec.video)
        it_vid = _SortableTableItem(
            short_vid,
            sort_key=(
                short_vid.lower(),
                rec.video.lower(),
                int(rec.identity),
                int(rec.start_frame),
            ),
        )
        it_vid.setToolTip(rec.video)
        self.table.setItem(row_idx, 2, it_vid)

        # Col 3: Animal (natural numeric sort by identity, then video & start frame)
        anim_text = self._animal_display_text(rec.video, rec.identity)
        it_anim = _SortableTableItem(
            anim_text,
            sort_key=(int(rec.identity), short_vid.lower(), int(rec.start_frame)),
        )
        beh_for_tip = self._bout_display_behavior(rec)
        it_anim.setToolTip(
            f"{anim_text} (0-based track index {rec.identity} • {beh_for_tip}) "
            "— click row to sync JABS Player"
        )
        self.table.setItem(row_idx, 3, it_anim)

        # Col 4: Frames (Dur) — natural numeric sort by duration (tie-break by start frame)
        dur_s = rec.duration_frames / 30.0
        it_span = _SortableTableItem(
            f"{rec.start_frame}\u2013{rec.end_frame} ({dur_s:.1f}s)",
            sort_key=(int(rec.duration_frames), int(rec.start_frame)),
        )
        it_span.setToolTip(f"{rec.duration_frames} frames ({rec.start_frame}..{rec.end_frame})")
        self.table.setItem(row_idx, 4, it_span)

        # Col 5: Suggested Fix
        it_fix = _SortableTableItem(
            suggested_fix,
            sort_key=(suggested_fix.lower(), -prio_score),
        )
        it_fix.setToolTip(plain_diag)
        self.table.setItem(row_idx, 5, it_fix)

        # Col 6: Persistent Examined [V] Tag (clickable directly in the table & sortable)
        if rec.bout_id in self._unlabeled_bout_ids or rec.source == "unlabeled":
            ex_txt = "○ No Label"
            ex_sort = (2, 0, -int(rec.bout_id))
            ex_bg = QtGui.QColor("#F1F5F9")
            ex_fg = QtGui.QColor("#475569")
            ex_bold = True
        elif rec.bout_id in self._examined_only_bout_ids:
            ex_txt = "✓ Examined"
            ex_sort = (1, 1, -int(rec.bout_id))
            ex_bg = QtGui.QColor("#ECFDF5")
            ex_fg = QtGui.QColor("#065F46")
            ex_bold = True
        elif rec.bout_id in self._reviewed_bout_ids:
            ex_txt = "✓ Saved"
            ex_sort = (1, 2, -int(rec.bout_id))
            ex_bg = QtGui.QColor("#DCFCE7")
            ex_fg = QtGui.QColor("#166534")
            ex_bold = True
        else:
            ex_txt = "—"
            ex_sort = (0, 0, -int(rec.bout_id))
            ex_bg = None
            ex_fg = QtGui.QColor("#94A3B8")
            ex_bold = False
        it_ex = _SortableTableItem(ex_txt, sort_key=ex_sort)
        it_ex.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        it_ex.setForeground(ex_fg)
        if ex_bg is not None:
            it_ex.setBackground(ex_bg)
        if ex_bold:
            ex_font = it_ex.font()
            ex_font.setBold(True)
            it_ex.setFont(ex_font)
        it_ex.setToolTip(
            "Automatically marked ✓ Examined when viewed in JABS — click this cell or press [V] to toggle"
        )
        self.table.setItem(row_idx, 6, it_ex)

    def _on_bout_search_text_changed(self, text: str) -> None:
        """Debounce live typing in `search_input` (120 ms); flush immediately when empty or programmatic."""
        if not text or not self.search_input.hasFocus():
            self._bout_search_timer.stop()
            self._on_bout_search_applied()
        else:
            self._bout_search_timer.start()

    def _on_bout_search_applied(self) -> None:
        self._bout_search_timer.stop()
        if self._report is None:
            return
        self._populate_table(redraw_scatter_on_change=False)
        self._draw_scatter()

    def _flush_bout_search_if_pending(self) -> None:
        if self._bout_search_timer.isActive():
            self._bout_search_timer.stop()
            self._on_bout_search_applied()

    def _find_table_row_for_bout(self, b_idx: int) -> int | None:
        """Return the displayed table row currently showing `b_idx`, or `None`."""
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and int(it.data(Qt.ItemDataRole.UserRole)) == b_idx:
                return r
        return None

    def _update_table_rows_in_place(self, changed_b_indices: list[int]) -> None:
        """Refresh only the affected rows in `self.table` cleanly (`< 4 ms`, zero ResizeToContents thrash)."""
        if self._report is None or not changed_b_indices:
            return
        records = self._report.feature_matrix.records
        cmap = self._behavior_color_hex_map()
        target_set = set(changed_b_indices)
        auto_cols = (0, 1, 3, 4, 6)
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        try:
            model = self.table.model()
            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it is None:
                    continue
                bi = int(it.data(Qt.ItemDataRole.UserRole))
                if bi in target_set:
                    self._populate_table_row(r, bi, records=records, cmap=cmap)
                    for c_i in auto_cols:
                        hint_w = self.table.sizeHintForIndex(model.index(r, c_i)).width() + 14
                        if hint_w > self.table.columnWidth(c_i):
                            self.table.setColumnWidth(c_i, hint_w)
        finally:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)

    def _on_table_header_clicked(self, col: int) -> None:
        """Sort the triage table by the clicked column (natural numeric/text keys)."""
        if self._report is None or self.table.rowCount() == 0:
            return
        if self._table_sort_col == col:
            order = (
                Qt.SortOrder.AscendingOrder
                if self._table_sort_order == Qt.SortOrder.DescendingOrder
                else Qt.SortOrder.DescendingOrder
            )
        else:
            # Default highest-priority / longest-bout / examined-first for Cols 0, 4, 6; A->Z for others
            order = (
                Qt.SortOrder.DescendingOrder if col in (0, 4, 6) else Qt.SortOrder.AscendingOrder
            )
        self._table_sort_col = col
        self._table_sort_order = order

        hdr = self.table.horizontalHeader()
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(col, order)

        prev_b_idx = self._selected_bout_idx
        self.table.blockSignals(True)
        self._suppress_auto_seek = True
        try:
            self.table.sortItems(col, order)
            if prev_b_idx is not None:
                new_row = self._find_table_row_for_bout(prev_b_idx)
                if new_row is not None:
                    self.table.selectRow(new_row)
        finally:
            self._suppress_auto_seek = False
            self.table.blockSignals(False)

    def _populate_table(
        self,
        *_: Any,
        auto_fallback_if_clean: bool = False,
        redraw_scatter_on_change: bool = True,
        preserve_none_selection: bool = False,
    ) -> None:
        if self._report is None:
            self.table.setRowCount(0)
            return

        if self._lasso_selected_indices:
            eligible_set = set(self._mode_eligible_indices())
            n_act = sum(1 for gi in self._lasso_selected_indices if gi in eligible_set)
            if n_act > 0:
                self.clear_box_btn.setText(f"✕ {self._custom_selection_label} ({n_act})")
                self.clear_box_btn.setVisible(True)
            else:
                self._clear_custom_selection_state()

        self._update_queue_filter_counts()
        indices = self._filtered_bout_indices()

        # If Audit Labels has 0 flagged issues on initial load, fall back to All Bouts automatically
        if (
            auto_fallback_if_clean
            and not indices
            and self._queue_filter == "needs_review"
            and not self.search_input.text().strip()
            and self._mode_eligible_indices()
        ):
            self._set_queue_filter("all", redraw=False)
            indices = self._filtered_bout_indices()

        records = self._report.feature_matrix.records
        cmap = self._behavior_color_hex_map()

        prev_b_idx = self._selected_bout_idx
        hdr = self.table.horizontalHeader()
        auto_cols = (0, 1, 3, 4, 6)
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        try:
            for c_i in auto_cols:
                hdr.setSectionResizeMode(c_i, QtWidgets.QHeaderView.ResizeMode.Interactive)
            self.table.setRowCount(0)
            self.table.setRowCount(len(indices))

            for row_idx, b_idx in enumerate(indices):
                self._populate_table_row(row_idx, b_idx, records=records, cmap=cmap)

            if self._table_sort_col is not None and len(indices) > 1:
                hdr.setSortIndicatorShown(True)
                hdr.setSortIndicator(self._table_sort_col, self._table_sort_order)
                self.table.sortItems(self._table_sort_col, self._table_sort_order)

            for c_i in auto_cols:
                self.table.resizeColumnToContents(c_i)

            reselect_row = 0
            if prev_b_idx is not None and len(indices) > 0:
                found_row = self._find_table_row_for_bout(prev_b_idx)
                if found_row is not None:
                    reselect_row = found_row
        finally:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)

        if len(indices) > 0:
            if preserve_none_selection and prev_b_idx is None:
                self.table.blockSignals(True)
                self.table.clearSelection()
                self.table.setCurrentCell(-1, -1)
                self.table.blockSignals(False)
                self._selected_bout_idx = None
                self._pending_click_already_sought_row = None
                self._scrubbed_frame = None
                self._update_empty_card_state()
                if redraw_scatter_on_change:
                    self._draw_scatter()
            else:
                self._suppress_auto_seek = True
                try:
                    self.table.selectRow(reselect_row)
                finally:
                    self._suppress_auto_seek = False
                sel_item = self.table.item(reselect_row, 0)
                self._selected_bout_idx = (
                    int(sel_item.data(Qt.ItemDataRole.UserRole))
                    if sel_item is not None
                    else indices[0]
                )
                self._pending_click_already_sought_row = None
                self._scrubbed_frame = None
                self._update_inspector_card()
                if redraw_scatter_on_change:
                    self._draw_scatter()
        else:
            self._selected_bout_idx = None
            self._update_empty_card_state()
            if redraw_scatter_on_change:
                self._draw_scatter()

    # ------------------------------------------------------------------
    # Selection, Playback & Frame Scrubbing
    # ------------------------------------------------------------------
    def _is_main_window_on_bout_track(self, rec: BoutRecord) -> bool:
        if self._central_widget is None:
            return False
        loaded = getattr(self._central_widget, "loaded_video", None)
        if loaded is None:
            return False
        try:
            loaded_name = Path(loaded).name
        except TypeError:
            return False
        if loaded_name != Path(rec.video).name:
            return False
        controls = getattr(self._central_widget, "_controls", None)
        curr_ident = (
            getattr(controls, "current_identity_index", None) if controls is not None else None
        )
        return isinstance(curr_ident, int | np.integer) and int(curr_ident) == int(rec.identity)

    def _clear_pending_click_if_released(self) -> None:
        if self._pending_click_already_sought_row is None or self._is_closing:
            return
        if not (QtWidgets.QApplication.mouseButtons() & Qt.MouseButton.LeftButton):
            self._pending_click_already_sought_row = None
        else:
            QtCore.QTimer.singleShot(50, self, self._clear_pending_click_if_released)

    def _clear_table_mouse_press_state(self) -> None:
        if self._is_closing:
            return
        self._table_mouse_press_col = None
        self._table_mouse_press_btn = None

    def _resolve_bout_index(self, rec: BoutRecord) -> int | None:
        if self._report is None:
            return None
        for idx, cand in enumerate(self._report.feature_matrix.records):
            if cand.bout_id == rec.bout_id:
                return idx
        return None

    def _mark_bout_auto_examined(self, b_idx: int) -> bool:
        """Automatically mark `b_idx` as persistently Examined when viewed in the JABS GUI (`< 2 ms`)."""
        if (
            self._report is None
            or self._central_widget is None
            or self._is_closing
            or not self.auto_mark_viewed_action.isChecked()
        ):
            return False
        records = self._report.feature_matrix.records
        if not (0 <= b_idx < len(records)):
            return False
        rec = records[b_idx]
        if rec.bout_id in self._reviewed_bout_ids or rec.bout_id in self._deleted_bout_ids:
            return False

        was_in_active_queue = (
            self._queue_filter == "needs_review" and self._bout_needs_review(b_idx)
        ) or (self._queue_filter == "label_edges" and self._bout_is_label_halo(b_idx))
        self._ensure_origin_snapshot(rec)
        self._reviewed_bout_ids.add(rec.bout_id)
        self._examined_only_bout_ids.add(rec.bout_id)
        if was_in_active_queue:
            self._pinned_needs_review_bout_ids.add(rec.bout_id)
        note = "✓ Examined (viewed in JABS)"
        self._saved_notes_by_bout[rec.bout_id] = note
        self._examined_store.mark(rec, action="examined", note=note)

        self._update_queue_filter_counts()
        self._update_table_rows_in_place([b_idx])
        if self._selected_bout_idx == b_idx:
            self._update_inspector_card(sync_mc_target=False, redraw_plot=False)
        self._schedule_scatter_redraw()
        return True

    def _on_external_search_hit_loaded(self, hit: SearchHit) -> None:
        """Auto-mark matching bout Examined when user steps through bouts in JABS Search Bar."""
        if (
            self._ignoring_self_seek_signal
            or self._is_closing
            or self._report is None
            or not self.auto_mark_viewed_action.isChecked()
            or hit is None
        ):
            return
        try:
            hit_vid = Path(hit.file).name
            hit_ident = int(hit.identity)
            hit_s = int(hit.start_frame)
            hit_e = int(hit.end_frame)
        except (TypeError, ValueError):
            return
        if hit_e < hit_s:
            hit_s, hit_e = hit_e, hit_s

        best_idx: int | None = None
        best_iou = 0.0
        for b_idx, rec in enumerate(self._report.feature_matrix.records):
            if rec.bout_id in self._deleted_bout_ids:
                continue
            if Path(rec.video).name != hit_vid or int(rec.identity) != hit_ident:
                continue
            if int(rec.start_frame) == hit_s and int(rec.end_frame) == hit_e:
                best_idx = b_idx
                break
            ov = min(int(rec.end_frame), hit_e) - max(int(rec.start_frame), hit_s) + 1
            if ov <= 0:
                continue
            union = max(int(rec.end_frame), hit_e) - min(int(rec.start_frame), hit_s) + 1
            iou = ov / max(1, union)
            if iou >= 0.50 and iou > best_iou:
                best_iou = iou
                best_idx = b_idx
        if best_idx is not None:
            self._mark_bout_auto_examined(best_idx)

    def _sync_selected_bout_to_main_window_if_enabled(self) -> None:
        if self._mode == "videos":
            self._on_video_open_clicked()
            return
        if (
            self._report is not None
            and self._selected_bout_idx is not None
            and 0 <= self._selected_bout_idx < len(self._report.feature_matrix.records)
            and self.auto_seek_action.isChecked()
            and not self._suppress_auto_seek
        ):
            self._seek_bout_in_main_window(
                self._report.feature_matrix.records[self._selected_bout_idx],
                b_idx=self._selected_bout_idx,
                bring_to_front=True,
            )

    def _on_table_selection_changed(self) -> None:
        if self._report is None:
            return
        sel = self.table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        id_item = self.table.item(row, 0)
        if id_item is None:
            return
        b_idx = int(id_item.data(Qt.ItemDataRole.UserRole))
        self._selected_bout_idx = b_idx
        self._scrubbed_frame = None
        self._update_bout_scatter_selection_ring()
        self._update_inspector_card()
        skip_seek_for_click = (
            self._table_mouse_press_col == 6
            or self._table_mouse_press_btn == Qt.MouseButton.RightButton
        )
        if (
            self.auto_seek_action.isChecked()
            and not self._suppress_auto_seek
            and not skip_seek_for_click
        ):
            from_left_click = self._table_mouse_press_btn == Qt.MouseButton.LeftButton
            self._pending_click_already_sought_row = row
            QtCore.QTimer.singleShot(0, self, self._clear_pending_click_if_released)
            rec = self._report.feature_matrix.records[b_idx]
            self._seek_bout_in_main_window(rec, b_idx=b_idx, bring_to_front=from_left_click)
        else:
            self._pending_click_already_sought_row = None

    def _on_table_cell_clicked(self, row: int, col: int) -> None:
        """Sync JABS Player on row click, or toggle Examined directly when clicking Col 6 (`Examined [V]`)."""
        self._clear_table_mouse_press_state()
        if self._report is None:
            return
        id_item = self.table.item(row, 0)
        if id_item is None:
            return
        b_idx = int(id_item.data(Qt.ItemDataRole.UserRole))
        if col == 6:
            self._pending_click_already_sought_row = None
            self._selected_bout_idx = b_idx
            self._toggle_bout_examined(b_idx)
            return
        if self._pending_click_already_sought_row == row:
            self._pending_click_already_sought_row = None
            if self.auto_seek_action.isChecked():
                self._request_bring_main_window_to_front()
            return
        self._pending_click_already_sought_row = None
        self._selected_bout_idx = b_idx
        self._scrubbed_frame = None
        self._update_bout_scatter_selection_ring()
        self._update_inspector_card()
        if self.auto_seek_action.isChecked():
            rec = self._report.feature_matrix.records[b_idx]
            self._seek_bout_in_main_window(rec, b_idx=b_idx, bring_to_front=True)

    def _bring_main_window_to_front(self) -> None:
        """Raise and activate the main JABS window so the sought bout is immediately visible."""
        if (
            self._is_closing
            or self._central_widget is None
            or not self.bring_jabs_to_front_action.isChecked()
        ):
            return
        win_fn = getattr(self._central_widget, "window", None)
        win = win_fn() if callable(win_fn) else None
        if not isinstance(win, QtWidgets.QWidget) or win is self:
            return
        if win.isMinimized():
            win.showNormal()
        elif not win.isVisible():
            win.show()
        win.raise_()
        win.activateWindow()

    def _request_bring_main_window_to_front(self) -> None:
        """Queue raising the main JABS window once the current mouse press finishes releasing."""
        if (
            self._is_closing
            or self._central_widget is None
            or not self.bring_jabs_to_front_action.isChecked()
        ):
            return
        self._pending_bring_main_window_to_front = True
        QtCore.QTimer.singleShot(0, self, self._flush_bring_main_window_to_front)

    def _flush_bring_main_window_to_front(self) -> None:
        """Raise the main JABS window after any held left-mouse press finishes releasing."""
        if not self._pending_bring_main_window_to_front or self._is_closing:
            return
        if QtWidgets.QApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            QtCore.QTimer.singleShot(25, self, self._flush_bring_main_window_to_front)
            return
        self._pending_bring_main_window_to_front = False
        self._bring_main_window_to_front()

    def _seek_bout_in_main_window(
        self,
        rec: BoutRecord,
        *,
        select_span: bool = True,
        mark_viewed: bool = True,
        b_idx: int | None = None,
        bring_to_front: bool = False,
    ) -> None:
        if self._central_widget is None or self._report is None:
            return
        self._bout_seek_epoch += 1
        beh_target = self._bout_display_behavior(rec)
        self._ignoring_self_seek_signal = True
        try:
            try:
                self._central_widget.seek_to_bout(
                    rec.video,
                    rec.identity,
                    rec.start_frame,
                    rec.end_frame if select_span else None,
                    beh_target,
                    select_span=select_span,
                )
            except TypeError:
                self._central_widget.seek_to_bout(
                    rec.video,
                    rec.identity,
                    rec.start_frame,
                    rec.end_frame if select_span else None,
                    beh_target,
                )
        finally:
            self._ignoring_self_seek_signal = False
        if mark_viewed:
            resolved_idx = b_idx if b_idx is not None else self._resolve_bout_index(rec)
            if resolved_idx is not None:
                self._mark_bout_auto_examined(resolved_idx)
        if bring_to_front:
            self._request_bring_main_window_to_front()

    def _on_play_clip_clicked(self) -> None:
        if self._mode == "videos":
            self._on_video_play_focal_clicked()
            return
        if self._report is None or self._selected_bout_idx is None or self._central_widget is None:
            return
        b_idx = self._selected_bout_idx
        rec = self._report.feature_matrix.records[b_idx]
        beh = self._bout_display_behavior(rec)
        if hasattr(self._central_widget, "play_bout_clip"):
            self._ignoring_self_seek_signal = True
            try:
                self._central_widget.play_bout_clip(
                    rec.video,
                    rec.identity,
                    rec.start_frame,
                    rec.end_frame,
                    beh,
                    context_frames=10,
                )
            finally:
                self._ignoring_self_seek_signal = False
            self._mark_bout_auto_examined(b_idx)
            self._request_bring_main_window_to_front()
        else:
            self._seek_bout_in_main_window(rec, b_idx=b_idx, bring_to_front=True)

    def _on_prob_curve_frame_clicked(self, x_frame: float) -> None:
        if self._mode == "videos":
            self._on_video_timeline_frame_clicked(x_frame)
            return
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._inspector_view != "prob"
        ):
            return
        b_idx = self._selected_bout_idx
        rec = self._report.feature_matrix.records[b_idx]
        target_f = max(rec.start_frame, min(round(x_frame), rec.end_frame))
        self._scrubbed_frame = target_f
        if self._central_widget is not None:
            # Ensure right video, animal identity, and behavior are active first, then scrub exact frame
            expected_beh = self._bout_display_behavior(rec)
            loaded = getattr(self._central_widget, "loaded_video", None)
            controls = getattr(self._central_widget, "_controls", None)
            curr_ident = (
                getattr(controls, "current_identity_index", None) if controls is not None else None
            )
            curr_beh = getattr(self._central_widget, "behavior", None)
            ident_mismatch = isinstance(curr_ident, int | np.integer) and int(curr_ident) != int(
                rec.identity
            )
            beh_mismatch = (
                bool(expected_beh) and isinstance(curr_beh, str) and curr_beh != expected_beh
            )
            if (
                loaded is None
                or Path(loaded).name != Path(rec.video).name
                or ident_mismatch
                or beh_mismatch
            ):
                self._seek_bout_in_main_window(rec, b_idx=b_idx)
            if hasattr(self._central_widget, "seek_to_frame_only"):
                self._central_widget.seek_to_frame_only(target_f)
            self._mark_bout_auto_examined(b_idx)
            self._request_bring_main_window_to_front()
        self._draw_inspector_plot()

    # ------------------------------------------------------------------
    # 2D Scatter Rendering & Selected Bout Triage Card
    # ------------------------------------------------------------------
    def _current_2d_coords(self) -> tuple[np.ndarray, str, str]:
        assert self._report is not None
        if self._proj_method == "spatial" and self._report.spatial is not None:
            sp = self._report.spatial
            unit_lbl = "normalized" if sp.normalized else "px"
            return sp.coords_xy, f"Arena X ({unit_lbl})", f"Arena Y ({unit_lbl})"
        emb = self._report.embedding
        if self._proj_method == "tsne" and emb.coords_tsne_2d is not None:
            return emb.coords_tsne_2d, "t-SNE 1", "t-SNE 2"
        if self._proj_method == "mds" and emb.coords_mds_2d is not None:
            return emb.coords_mds_2d, "MDS 1 (RF Proximity)", "MDS 2 (RF Proximity)"
        ev1, ev2 = emb.pca_explained_variance_ratio
        return (
            emb.coords_pca_2d,
            f"PC1 ({ev1 * 100:.1f}% var)",
            f"PC2 ({ev2 * 100:.1f}% var)",
        )

    def _update_bout_scatter_selection_ring(self) -> None:
        """Move the selected-bout highlight ring in O(1) without rebuilding the scatter plot."""
        if (
            self._bout_scatter_sel_artist is None
            or self._bout_scatter_vis_coords is None
            or not self._bout_scatter_vis_map
        ):
            self._draw_scatter()
            return
        b_idx = self._selected_bout_idx
        loc = self._bout_scatter_vis_map.get(int(b_idx)) if b_idx is not None else None
        if loc is not None and 0 <= loc < len(self._bout_scatter_vis_coords):
            pt = self._bout_scatter_vis_coords[loc]
            self._bout_scatter_sel_artist.set_offsets(
                np.array([[float(pt[0]), float(pt[1])]], dtype=np.float64)
            )
            self._bout_scatter_sel_artist.set_visible(True)
        else:
            self._bout_scatter_sel_artist.set_offsets(np.empty((0, 2), dtype=np.float64))
            self._bout_scatter_sel_artist.set_visible(False)
        self.scatter_canvas.draw_idle()

    def _redraw_scatter_and_inspector(self) -> None:
        if self._mode == "videos":
            self._draw_video_scatter()
            self._update_video_inspector_card()
            return
        if self._report is None:
            return
        self._draw_scatter()
        self._update_inspector_card()

    def _draw_scatter(self) -> None:
        if self._mode == "videos":
            self._draw_video_scatter()
            return
        if self._report is None:
            return
        self._bout_scatter_sel_artist = None
        self._bout_scatter_vis_coords = None
        self._bout_scatter_vis_map = {}
        ax = self.scatter_ax
        ax.clear()

        records = self._report.feature_matrix.records
        is_spatial = self._proj_method == "spatial"
        if is_spatial and self._report.spatial is None:
            ax.set_aspect("auto", adjustable="box")
            ax.text(
                0.5,
                0.5,
                "Mapping bout arena locations…",
                ha="center",
                va="center",
                fontsize=11.5,
                color="#475569",
                transform=ax.transAxes,
            )
            self.scatter_canvas.set_pick_points(None)
            self.scatter_canvas.mark_layout_dirty()
            self.scatter_canvas.draw_idle()
            return

        if is_spatial:
            sp = self._report.spatial
            assert sp is not None
            w_bound = float(sp.frame_width or 1.0)
            h_bound = float(sp.frame_height or 1.0)
            if sp.reference_frame_gray is not None and not sp.normalized:
                ax.imshow(
                    sp.reference_frame_gray,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    alpha=0.52,
                    extent=(0.0, w_bound, h_bound, 0.0),
                    interpolation="bilinear",
                    zorder=0,
                )
            draw_arena_static_landmarks(ax, sp.static_objects)
        else:
            ax.set_aspect("auto", adjustable="box")

        base_eligible = self._mode_eligible_indices()
        vis_indices = self._scatter_plot_indices()
        backdrop_indices = self._scatter_backdrop_indices()
        all_coords, xlab, ylab = self._current_2d_coords()
        if is_spatial:
            base_eligible = [gi for gi in base_eligible if np.isfinite(all_coords[gi]).all()]
            vis_indices = [gi for gi in vis_indices if np.isfinite(all_coords[gi]).all()]
            backdrop_indices = [gi for gi in backdrop_indices if np.isfinite(all_coords[gi]).all()]

        if not vis_indices:
            if is_spatial:
                sp = self._report.spatial
                assert sp is not None
                w_bound = float(sp.frame_width or 1.0)
                h_bound = float(sp.frame_height or 1.0)
                ax.set_xlim(0.0, w_bound)
                ax.set_ylim(h_bound, 0.0)
                ax.set_aspect("equal", adjustable="box")
                self._scatter_default_limits = ((0.0, w_bound), (h_bound, 0.0))
                if self._scatter_view_limits is not None:
                    ax.set_xlim(self._scatter_view_limits[0])
                    ax.set_ylim(self._scatter_view_limits[1])
                empty_msg = (
                    "No bouts match the active filter."
                    if base_eligible
                    else "No mappable arena pose coordinates for this view."
                )
            else:
                if base_eligible:
                    bc = all_coords[np.asarray(base_eligible, dtype=np.int32)]
                    x_min, x_max = float(np.nanmin(bc[:, 0])), float(np.nanmax(bc[:, 0]))
                    y_min, y_max = float(np.nanmin(bc[:, 1])), float(np.nanmax(bc[:, 1]))
                    px = 0.05 * (x_max - x_min) if (x_max - x_min) > 1e-6 else 1.0
                    py = 0.05 * (y_max - y_min) if (y_max - y_min) > 1e-6 else 1.0
                    def_xlim = (x_min - px, x_max + px)
                    def_ylim = (y_min - py, y_max + py)
                    ax.set_xlim(def_xlim)
                    ax.set_ylim(def_ylim)
                    self._scatter_default_limits = (def_xlim, def_ylim)
                    if self._scatter_view_limits is not None:
                        ax.set_xlim(self._scatter_view_limits[0])
                        ax.set_ylim(self._scatter_view_limits[1])
                if self._mode == "predict" and not base_eligible:
                    scope_lbl = self._active_scope_label()
                    empty_msg = (
                        f"No predicted {scope_lbl} bouts to review.\n"
                        f"0 unannotated predicted bouts found for {scope_lbl} in jabs/predictions/*.h5."
                    )
                elif base_eligible:
                    empty_msg = "No bouts match the active filter."
                else:
                    empty_msg = "No bouts to display for this view."
            self._sync_zoom_reset_button()
            ax.set_xlabel(xlab, fontsize=11.0, fontweight="bold", color="#1E293B")
            ax.set_ylabel(ylab, fontsize=11.0, fontweight="bold", color="#1E293B")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.5,
                0.5,
                empty_msg,
                ha="center",
                va="center",
                fontsize=11.5,
                color="#475569",
                bbox=(
                    {"boxstyle": "round,pad=0.3", "facecolor": "#FFFFFF", "alpha": 0.88}
                    if is_spatial
                    else None
                ),
                transform=ax.transAxes,
            )
            self.scatter_canvas.set_pick_points(None)
            self.scatter_canvas.mark_layout_dirty()
            self.scatter_canvas.draw_idle()
            return

        vis_arr = np.asarray(vis_indices, dtype=np.int32)
        coords = all_coords[vis_arr]
        self._bout_scatter_vis_coords = coords
        self._bout_scatter_vis_map = {int(gi): k for k, gi in enumerate(vis_indices)}
        diag = self._report.diagnostics
        emb = self._report.embedding
        color_by = self._effective_color_mode()
        beh_cmap = self._behavior_color_hex_map()

        if backdrop_indices:
            bg_arr = np.asarray(backdrop_indices, dtype=np.int32)
            bg_coords = all_coords[bg_arr]
            if color_by == "class":
                for cls_name in sorted({records[i].class_name for i in backdrop_indices}):
                    bg_loc = [
                        k
                        for k, gi in enumerate(backdrop_indices)
                        if records[gi].class_name == cls_name
                    ]
                    if bg_loc:
                        ax.scatter(
                            bg_coords[bg_loc, 0],
                            bg_coords[bg_loc, 1],
                            c=beh_cmap.get(cls_name, "#64748B"),
                            s=24,
                            alpha=0.16,
                            marker="o",
                            edgecolors="none",
                            zorder=1,
                            label=f"Labeled {cls_name} ({len(bg_loc)})",
                        )
            else:
                ax.scatter(
                    bg_coords[:, 0],
                    bg_coords[:, 1],
                    c="#94A3B8",
                    s=24,
                    alpha=0.16,
                    marker="o",
                    edgecolors="none",
                    zorder=1,
                    label=f"Labeled context ({len(backdrop_indices)})",
                )

        if color_by == "class":
            vis_classes = sorted({records[i].class_name for i in vis_indices})
            for cls_name in vis_classes:
                color = beh_cmap.get(cls_name, "#1E40AF")
                if self._mode == "predict":
                    unrev_local = [
                        k
                        for k, gi in enumerate(vis_indices)
                        if records[gi].class_name == cls_name
                        and records[gi].source == "predicted"
                        and not bool(getattr(records[gi], "is_label_halo", False))
                    ]
                    halo_local = [
                        k
                        for k, gi in enumerate(vis_indices)
                        if records[gi].class_name == cls_name
                        and records[gi].source == "predicted"
                        and bool(getattr(records[gi], "is_label_halo", False))
                    ]
                    rev_local = [
                        k
                        for k, gi in enumerate(vis_indices)
                        if records[gi].class_name == cls_name and records[gi].source == "labeled"
                    ]
                    if unrev_local:
                        ax.scatter(
                            coords[unrev_local, 0],
                            coords[unrev_local, 1],
                            c=color,
                            s=72,
                            alpha=0.92,
                            marker="D",
                            edgecolors="#0F172A",
                            linewidths=0.85,
                            zorder=4,
                            label=f"Predicted {cls_name} ({len(unrev_local)})",
                        )
                    if halo_local:
                        ax.scatter(
                            coords[halo_local, 0],
                            coords[halo_local, 1],
                            c="#7C3AED",
                            s=54,
                            alpha=0.76,
                            marker="^",
                            edgecolors="#4C1D95",
                            linewidths=0.75,
                            zorder=3.5,
                            label=f"Label Edge {cls_name} ({len(halo_local)})",
                        )
                    if rev_local:
                        ax.scatter(
                            coords[rev_local, 0],
                            coords[rev_local, 1],
                            c=color,
                            s=64,
                            alpha=0.88,
                            marker="D",
                            edgecolors="#15803D",
                            linewidths=1.35,
                            zorder=4,
                            label=f"Reviewed {cls_name} ({len(rev_local)})",
                        )
                else:
                    lab_local = [
                        k
                        for k, gi in enumerate(vis_indices)
                        if records[gi].class_name == cls_name and records[gi].source == "labeled"
                    ]
                    if lab_local:
                        ax.scatter(
                            coords[lab_local, 0],
                            coords[lab_local, 1],
                            c=color,
                            s=48,
                            alpha=0.78,
                            marker="o",
                            edgecolors="#FFFFFF",
                            linewidths=0.45,
                            label=f"{cls_name} ({len(lab_local)})",
                        )
        elif color_by == "cluster":
            pt_marker = "D" if self._mode == "predict" else "o"
            pt_edge = "#0F172A" if self._mode == "predict" else "#FFFFFF"
            pt_lw = 0.75 if self._mode == "predict" else 0.45
            pt_size = 64 if self._mode == "predict" else 50
            pt_z = 4 if self._mode == "predict" else 2
            for cid in sorted(set(emb.cluster_labels[vis_arr].tolist())):
                loc = np.flatnonzero(emb.cluster_labels[vis_arr] == cid)
                c = _CLUSTER_PALETTE[(cid - 1) % len(_CLUSTER_PALETTE)]
                ax.scatter(
                    coords[loc, 0],
                    coords[loc, 1],
                    c=c,
                    s=pt_size,
                    alpha=0.88 if self._mode == "predict" else 0.82,
                    marker=pt_marker,
                    edgecolors=pt_edge,
                    linewidths=pt_lw,
                    zorder=pt_z,
                    label=f"Cluster C{cid} ({len(loc)})",
                )
        elif color_by == "identity":
            pt_marker = "D" if self._mode == "predict" else "o"
            pt_edge = "#0F172A" if self._mode == "predict" else "#FFFFFF"
            pt_lw = 0.75 if self._mode == "predict" else 0.4
            pt_size = 62 if self._mode == "predict" else 48
            pt_z = 4 if self._mode == "predict" else 2
            for ident in sorted({records[i].identity for i in vis_indices}):
                loc = [k for k, gi in enumerate(vis_indices) if records[gi].identity == ident]
                c = _CLUSTER_PALETTE[int(ident) % len(_CLUSTER_PALETTE)]
                sample_vid = records[vis_indices[loc[0]]].video
                anim_lbl = self._animal_display_text(sample_vid, ident)
                ax.scatter(
                    coords[loc, 0],
                    coords[loc, 1],
                    c=c,
                    s=pt_size,
                    alpha=0.88 if self._mode == "predict" else 0.80,
                    marker=pt_marker,
                    edgecolors=pt_edge,
                    linewidths=pt_lw,
                    zorder=pt_z,
                    label=f"{anim_lbl} ({len(loc)})",
                )
        elif color_by == "video":
            pt_marker = "D" if self._mode == "predict" else "o"
            pt_edge = "#0F172A" if self._mode == "predict" else "#FFFFFF"
            pt_lw = 0.75 if self._mode == "predict" else 0.4
            pt_size = 60 if self._mode == "predict" else 46
            pt_z = 4 if self._mode == "predict" else 2
            cage_for_loc = [
                _short_video_name(records[gi].video).split(" • ")[0] for gi in vis_indices
            ]
            cage_counts: dict[str, int] = {}
            for cg in cage_for_loc:
                cage_counts[cg] = cage_counts.get(cg, 0) + 1
            ranked_cages = [
                cg for cg, _ in sorted(cage_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
            top_cages = ranked_cages[:7] if len(ranked_cages) > 8 else ranked_cages
            top_set = set(top_cages)
            for v_idx, cage_lbl in enumerate(top_cages):
                loc = [k for k, cg in enumerate(cage_for_loc) if cg == cage_lbl]
                c = _CLUSTER_PALETTE[v_idx % len(_CLUSTER_PALETTE)]
                ax.scatter(
                    coords[loc, 0],
                    coords[loc, 1],
                    c=c,
                    s=pt_size,
                    alpha=0.86 if self._mode == "predict" else 0.78,
                    marker=pt_marker,
                    edgecolors=pt_edge,
                    linewidths=pt_lw,
                    zorder=pt_z,
                    label=f"{cage_lbl} ({len(loc)})",
                )
            other_loc = [k for k, cg in enumerate(cage_for_loc) if cg not in top_set]
            if other_loc:
                ax.scatter(
                    coords[other_loc, 0],
                    coords[other_loc, 1],
                    c="#64748B",
                    s=50 if self._mode == "predict" else 38,
                    alpha=0.65 if self._mode == "predict" else 0.55,
                    marker=pt_marker,
                    edgecolors=pt_edge,
                    linewidths=0.35,
                    zorder=pt_z,
                    label=f"Other Cages ({len(other_loc)})",
                )
        else:
            vals = diag.confusion_scores[vis_arr]
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=vals,
                cmap="YlOrRd",
                vmin=0.0,
                vmax=1.0,
                s=64 if self._mode == "predict" else 52,
                alpha=0.90 if self._mode == "predict" else 0.86,
                marker="D" if self._mode == "predict" else "o",
                edgecolors="#0F172A" if self._mode == "predict" else "#334155",
                linewidths=0.65 if self._mode == "predict" else 0.4,
                zorder=4 if self._mode == "predict" else 2,
            )
            ax.plot([], [], " ", label=f"Mean score: {float(np.mean(vals)):.2f}")
            _ = sc

        # Highlight active custom selection / cohort on the 2D map with crisp blue rings
        if self._lasso_selected_indices:
            cohort_set = set(self._lasso_selected_indices)
            cohort_loc = [k for k, gi in enumerate(vis_indices) if gi in cohort_set]
            if cohort_loc:
                ax.scatter(
                    coords[cohort_loc, 0],
                    coords[cohort_loc, 1],
                    facecolors="none",
                    edgecolors="#1D4ED8",
                    s=138,
                    linewidths=1.85,
                    zorder=8,
                    label=f"Selected ({len(cohort_loc)})",
                )

        # Highlight unreviewed high-priority bouts with crisp crimson rings
        flagged_loc = [
            k
            for k, gi in enumerate(vis_indices)
            if self._bout_needs_review(gi) and float(diag.confusion_scores[gi]) >= 0.65
        ]
        if flagged_loc and self._mode != "subtypes":
            ax.scatter(
                coords[flagged_loc, 0],
                coords[flagged_loc, 1],
                facecolors="none",
                edgecolors="#B91C1C",
                s=112,
                linewidths=1.6,
                zorder=7,
                label=f"Flagged ({len(flagged_loc)})",
            )

        # Highlight currently selected bout with a persistent O(1) artist for fast row stepping
        sel_loc = (
            self._bout_scatter_vis_map.get(int(self._selected_bout_idx))
            if self._selected_bout_idx is not None
            else None
        )
        if sel_loc is not None and 0 <= sel_loc < len(coords):
            sel_pt = coords[sel_loc]
            self._bout_scatter_sel_artist = ax.scatter(
                [sel_pt[0]],
                [sel_pt[1]],
                facecolors="none",
                edgecolors="#0F172A",
                s=195,
                linewidths=2.5,
                zorder=10,
            )
            self._bout_scatter_sel_artist.set_visible(True)
        else:
            self._bout_scatter_sel_artist = ax.scatter(
                [],
                [],
                facecolors="none",
                edgecolors="#0F172A",
                s=195,
                linewidths=2.5,
                zorder=10,
            )
            self._bout_scatter_sel_artist.set_visible(False)

        # Keep full-cohort bounds anchored across class/search filtering (Arena inverted-Y or full-cloud PCA)
        if is_spatial:
            sp = self._report.spatial
            assert sp is not None
            w_bound = float(sp.frame_width or 1.0)
            h_bound = float(sp.frame_height or 1.0)
            ax.set_xlim(0.0, w_bound)
            ax.set_ylim(h_bound, 0.0)
            ax.set_aspect("equal", adjustable="box")
        elif base_eligible:
            base_arr = np.asarray(base_eligible, dtype=np.int32)
            base_c = all_coords[base_arr]
            x_min, x_max = float(np.nanmin(base_c[:, 0])), float(np.nanmax(base_c[:, 0]))
            y_min, y_max = float(np.nanmin(base_c[:, 1])), float(np.nanmax(base_c[:, 1]))
            px = 0.05 * (x_max - x_min) if (x_max - x_min) > 1e-6 else 1.0
            py = 0.05 * (y_max - y_min) if (y_max - y_min) > 1e-6 else 1.0
            ax.set_xlim(x_min - px, x_max + px)
            ax.set_ylim(y_min - py, y_max + py)

        # Capture full-view bounds, then re-apply persistent user zoom if active
        def_xlim = tuple(float(v) for v in ax.get_xlim())
        def_ylim = tuple(float(v) for v in ax.get_ylim())
        self._scatter_default_limits = (def_xlim, def_ylim)  # type: ignore[assignment]
        if self._scatter_view_limits is not None:
            ax.set_xlim(self._scatter_view_limits[0])
            ax.set_ylim(self._scatter_view_limits[1])
        self._sync_zoom_reset_button()

        zoom_tag = " [Zoomed]" if self._scatter_view_limits else ""
        has_active_filter = bool(
            self._class_filter != "all"
            or self._lasso_selected_indices
            or self.search_input.text().strip()
            or self.sync_plot_to_queue_tab_action.isChecked()
        )
        filt_cnt_tag = (
            f" [{len(vis_indices)}/{len(base_eligible)}]"
            if (has_active_filter and len(vis_indices) != len(base_eligible))
            else ""
        )
        pred_subtitle = "◆ Pred / ● Labeled" if backdrop_indices else "◆ Pred only"
        lens_lbl = self.subtypes_lens_combo.currentText()
        if is_spatial:
            sp = self._report.spatial
            assert sp is not None
            n_mapped = int(np.sum(sp.valid_mask))
            mode_short = {
                "audit": "Audit",
                "predict": "Predictions",
                "subtypes": f"{emb.optimal_k} Clusters",
            }.get(self._mode, self._mode)
            if has_active_filter and len(vis_indices) != len(base_eligible):
                title_txt = (
                    f"Arena Spatial Map ({len(vis_indices)}/{len(base_eligible)} shown • "
                    f"{mode_short}){zoom_tag}"
                )
            else:
                title_txt = f"Arena Spatial Map ({n_mapped}/{len(records)} mapped • {mode_short}){zoom_tag}"
        else:
            mode_titles = {
                "audit": f"2D Label Map{filt_cnt_tag}{zoom_tag}",
                "predict": f"2D Active-Learning Map ({pred_subtitle}){filt_cnt_tag}{zoom_tag}",
                "subtypes": f"2D Sub-Types ({emb.optimal_k} Clusters • {lens_lbl}){filt_cnt_tag}{zoom_tag}",
            }
            title_txt = mode_titles[self._mode]
        ax.set_title(
            title_txt,
            fontsize=11.5,
            fontweight="bold",
            color="#0F172A",
            pad=8.0,
        )
        ax.set_xlabel(xlab, fontsize=11.0, fontweight="bold", color="#1E293B")
        ax.set_ylabel(ylab, fontsize=11.0, fontweight="bold", color="#1E293B")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(
            True,
            color="#CBD5E1" if is_spatial else "#E2E8F0",
            alpha=0.55 if is_spatial else 1.0,
            linewidth=0.8,
        )
        ax.legend(
            loc="upper right",
            fontsize=8.5 if is_spatial else 9.0,
            frameon=True,
            facecolor="#FFFFFF",
            framealpha=0.92,
            edgecolor="#CBD5E1",
        )

        if is_spatial:
            sp = self._report.spatial
            assert sp is not None
            hover_tips = [
                f"#{records[gi].bout_id} • {records[gi].class_name} ({records[gi].source})\n"
                f"{_short_video_name(records[gi].video)} • {self._animal_display_text(records[gi].video, records[gi].identity)} • "
                f"{records[gi].start_frame}\u2013{records[gi].end_frame}\n"
                f"Arena: ({all_coords[gi, 0]:.2f}, {all_coords[gi, 1]:.2f})"
                if sp.normalized
                else f"#{records[gi].bout_id} • {records[gi].class_name} ({records[gi].source})\n"
                f"{_short_video_name(records[gi].video)} • {self._animal_display_text(records[gi].video, records[gi].identity)} • "
                f"{records[gi].start_frame}\u2013{records[gi].end_frame}\n"
                f"Arena: ({all_coords[gi, 0]:.0f}, {all_coords[gi, 1]:.0f}) px • {self._describe_bout_issue(gi)[0]}"
                for gi in vis_indices
            ]
        else:
            hover_tips = [
                f"#{records[gi].bout_id} • {records[gi].class_name} ({records[gi].source})\n"
                f"{_short_video_name(records[gi].video)} • {self._animal_display_text(records[gi].video, records[gi].identity)} • "
                f"{records[gi].start_frame}\u2013{records[gi].end_frame}\n"
                f"{self._describe_bout_issue(gi)[0]}"
                for gi in vis_indices
            ]
        self.scatter_canvas.set_pick_points(coords, vis_arr, hover_tips)
        self.scatter_canvas.mark_layout_dirty()
        self.scatter_canvas.draw_idle()

    def _update_empty_card_state(self) -> None:
        self._last_card_bout_idx = None
        if self._report is not None and self.table.rowCount() > 0:
            self.card_title_lbl.setText("Select a bout to inspect")
            self.card_meta_lbl.setText(
                "Click any row in the picker or a dot on the Map to examine it"
            )
            self.diagnosis_lbl.setText(
                "No bout selected. Pick a bout on the left to inspect its timeline "
                "and jump the JABS video player to that clip."
            )
            self.diagnosis_frame.setStyleSheet(
                "QFrame { background-color: #F8FAFC; border-left: 4px solid #64748B; border-radius: 3px; }"
            )
        elif (
            self._mode == "predict"
            and self._report is not None
            and not self._mode_eligible_indices()
        ):
            scope_lbl = self._active_scope_label()
            self.card_title_lbl.setText(f"No predicted {scope_lbl} bouts")
            self.card_meta_lbl.setText(
                f"0 unannotated predicted bouts found for {scope_lbl} in jabs/predictions/*.h5"
            )
            self.diagnosis_lbl.setText(
                f"0 unannotated predicted bouts were found for {scope_lbl} in jabs/predictions/*.h5. "
                "Run JABS prediction or switch Behavior / Workflow mode."
            )
            self.diagnosis_frame.setStyleSheet(
                "QFrame { background-color: #F8FAFC; border-left: 4px solid #64748B; border-radius: 3px; }"
            )
        else:
            self.card_title_lbl.setText("No bouts in current filter")
            self.card_meta_lbl.setText("Switch to 'All Bouts' below or clear the search filter")
            self.diagnosis_lbl.setText(
                "Zero bouts match the active filter. If 'Needs Review (0)' is empty, "
                "all bouts in this mode look clean!"
            )
            self.diagnosis_frame.setStyleSheet(
                "QFrame { background-color: #DCFCE7; border-left: 4px solid #15803D; border-radius: 3px; }"
            )
        self.play_clip_btn.setEnabled(False)
        self.mc_target_lbl.setVisible(False)
        self.target_pos_combo.setVisible(False)
        self.primary_pos_btn.setEnabled(False)
        self.primary_pos_btn.setVisible(self._mode != "subtypes")
        self.secondary_neg_btn.setVisible(False)
        self.trim_btn.setVisible(False)
        self.split_btn.setVisible(False)
        self.delete_btn.setVisible(False)
        self.examine_btn.setEnabled(False)
        self.examine_btn.setVisible(True)
        self.examine_btn.setText("✓ Examined [V]")
        self.insp_ax.clear()
        self.insp_ax.set_aspect("auto", adjustable="box")
        self.insp_ax.set_axis_off()
        self.insp_canvas.mark_layout_dirty()
        self.insp_canvas.draw_idle()

    def _update_inspector_card(
        self,
        *,
        sync_mc_target: bool = True,
        redraw_plot: bool = True,
    ) -> None:
        if self._report is None or self._selected_bout_idx is None:
            self._update_empty_card_state()
            return
        b_idx = self._selected_bout_idx
        records = self._report.feature_matrix.records
        if not (0 <= b_idx < len(records)):
            self._update_empty_card_state()
            return

        rec = records[b_idx]
        diag = self._report.diagnostics
        is_mc = self._report.is_multiclass
        beh = rec.behavior if (rec.behavior and rec.behavior != "None") else self._report.behavior
        opp_lbl = "None" if is_mc else f"Not-{beh}"

        # When switching to a new bout in Multi-Class mode, default Class: picker to its positive class
        if (
            sync_mc_target
            and is_mc
            and b_idx != self._last_card_bout_idx
            and rec.class_name != MULTICLASS_NONE_BEHAVIOR
            and rec.source != "unlabeled"
        ):
            idx = self.target_pos_combo.findData(rec.class_name)
            if idx >= 0:
                self.target_pos_combo.blockSignals(True)
                self.target_pos_combo.setCurrentIndex(idx)
                self.target_pos_combo.blockSignals(False)
        self._last_card_bout_idx = b_idx

        origin_src = getattr(rec, "origin_source", rec.source)
        if rec.bout_id in self._unlabeled_bout_ids or rec.source == "unlabeled":
            src_tag = "Unlabeled"
        elif self._mode == "predict" and origin_src == "predicted" and rec.source == "labeled":
            src_tag = "Reviewed Prediction"
        elif rec.source == "predicted":
            src_tag = "Predicted"
        else:
            src_tag = "Labeled"
        dur_s = rec.duration_frames / 30.0
        anim_lbl = self._animal_display_text(rec.video, rec.identity)
        self.card_title_lbl.setText(f"Bout #{rec.bout_id}  •  {src_tag}: {rec.class_name}")
        self.card_meta_lbl.setText(
            f"{_short_video_name(rec.video)}  •  {anim_lbl}  •  "
            f"Frames {rec.start_frame}\u2013{rec.end_frame} ({dur_s:.1f}s / {rec.duration_frames}f)"
        )
        self.card_meta_lbl.setToolTip(rec.video)
        self.play_clip_btn.setEnabled(True)

        _, _, plain_diag, accent_hex = self._describe_bout_issue(b_idx)
        if rec.bout_id in self._examined_only_bout_ids:
            plain_diag = f"{plain_diag} (✓ Examined — press [V] or click table tag to unmark.)"
        bg_hex = (
            "#FEF2F2"
            if accent_hex == "#B91C1C"
            else ("#FFFBEB" if accent_hex == "#D97706" else "#F1F5F9")
        )
        self.diagnosis_frame.setStyleSheet(
            f"QFrame {{ background-color: {bg_hex}; border-left: 4px solid {accent_hex}; border-radius: 3px; }}"
        )
        self.diagnosis_lbl.setText(plain_diag)

        # Update persistent Examined toggle button across all 3 modes
        is_already_reviewed = rec.bout_id in self._reviewed_bout_ids
        self.examine_btn.setVisible(True)
        self.examine_btn.setEnabled(True)
        if is_already_reviewed:
            self.examine_btn.setText("↺ Unmark [V]")
            self.examine_btn.setStyleSheet(
                "QPushButton { background-color: #ECFDF5; color: #065F46; font-weight: 700; "
                "padding: 5px 9px; border-radius: 5px; border: 1px solid #6EE7B7; }"
                "QPushButton:hover { background-color: #D1FAE5; }"
            )
            self.examine_btn.setToolTip(
                "Clear persistent Examined flag so this bout returns to Needs Review [Key: V]"
            )
        else:
            self.examine_btn.setText("✓ Examined [V]")
            self.examine_btn.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: #0F766E; font-weight: 600; "
                "padding: 5px 9px; border-radius: 5px; border: 1px solid #99F6E4; }"
                "QPushButton:hover { background-color: #F0FDFA; }"
            )
            self.examine_btn.setToolTip(
                "Automatically marked when viewed in JABS — click or press [V] to mark/unmark manually (persists across restarts)"
            )

        # In 3. Sub-Types mode, keep inspection uncluttered (hide relabeling controls)
        if self._mode == "subtypes":
            self.mc_target_lbl.setVisible(False)
            self.target_pos_combo.setVisible(False)
            self.primary_pos_btn.setVisible(False)
            self.secondary_neg_btn.setVisible(False)
            self.trim_btn.setVisible(False)
            self.split_btn.setVisible(False)
            self.delete_btn.setVisible(False)
            if redraw_plot:
                self._draw_inspector_plot()
            return

        self.mc_target_lbl.setVisible(is_mc)
        self.target_pos_combo.setVisible(is_mc)
        self.primary_pos_btn.setVisible(True)
        self.primary_pos_btn.setEnabled(True)

        # Configure JABS-native [Z] Behavior / [C] Not-Behavior / [X] No Label buttons
        trim = diag.suggested_trim_bounds[b_idx]
        split_f = diag.suggested_split_frame[b_idx]
        pos_target = str(self.target_pos_combo.currentData() or beh)

        if rec.source in ("predicted", "unlabeled"):
            if is_mc:
                if rec.source == "predicted" and pos_target == rec.class_name:
                    self.primary_pos_btn.setText(f"✓ Confirm {rec.class_name} [Z]")
                    self.primary_pos_btn.setToolTip(
                        "Save this predicted bout as a verified Positive training annotation [Key: Z or 1]"
                    )
                else:
                    self.primary_pos_btn.setText(f"⇄ Save as {pos_target} [Z]")
                    self.primary_pos_btn.setToolTip(
                        f"Save this bout as {pos_target} [Key: Z or 1]"
                    )
            else:
                self.primary_pos_btn.setText(f"✓ Confirm {beh} [Z]")
                self.primary_pos_btn.setToolTip(
                    "Save this bout as a verified Positive training annotation [Key: Z or 1]"
                )
            self.secondary_neg_btn.setVisible(True)
            self.secondary_neg_btn.setEnabled(True)
            self.secondary_neg_btn.setText(f"✗ Mark {opp_lbl} [C]")
            self.secondary_neg_btn.setToolTip(
                f"Save as explicit {opp_lbl} training label [Key: C or 2]"
            )
            self.trim_btn.setVisible(False)
            self.split_btn.setVisible(
                rec.source == "predicted"
                and (rec.postproc_fragility >= 0.35 or split_f is not None)
            )
            if self.split_btn.isVisible():
                sf = split_f if split_f is not None else (rec.start_frame + rec.end_frame) // 2
                self.split_btn.setText(f"✂ Split @ {sf}")
            self.delete_btn.setVisible(True)
            self.delete_btn.setEnabled(True)
            self.delete_btn.setText("○ No Label [X]")
            self.delete_btn.setToolTip("Leave unlabeled / skip without adding a label [Key: X]")
        else:
            if is_mc:
                if rec.class_name == MULTICLASS_NONE_BEHAVIOR:
                    self.primary_pos_btn.setText("✓ Keep None [C]")
                    self.primary_pos_btn.setToolTip(
                        "Verify this None bout as correct [Key: C or 1]"
                    )
                    self.secondary_neg_btn.setVisible(True)
                    self.secondary_neg_btn.setEnabled(True)
                    self.secondary_neg_btn.setText(f"⇄ Switch to {pos_target} [Z]")
                    self.secondary_neg_btn.setToolTip(
                        f"Relabel this bout from None to {pos_target} [Key: Z or 2]"
                    )
                else:
                    if pos_target == rec.class_name:
                        self.primary_pos_btn.setText(f"✓ Keep {rec.class_name} [Z]")
                        self.primary_pos_btn.setToolTip(
                            f"Verify this {rec.class_name} label as correct [Key: Z or 1]"
                        )
                    else:
                        self.primary_pos_btn.setText(f"⇄ Switch to {pos_target} [Z]")
                        self.primary_pos_btn.setToolTip(
                            f"Relabel this bout from {rec.class_name} to {pos_target} [Key: Z or 1]"
                        )
                    self.secondary_neg_btn.setVisible(True)
                    self.secondary_neg_btn.setEnabled(True)
                    self.secondary_neg_btn.setText("✗ Set None [C]")
                    self.secondary_neg_btn.setToolTip(
                        "Relabel this bout to None (Not-Behavior) [Key: C or 2]"
                    )
            else:
                if rec.is_positive:
                    self.primary_pos_btn.setText(f"✓ Keep {rec.class_name} [Z]")
                    self.primary_pos_btn.setToolTip(
                        f"Verify this {rec.class_name} label as correct [Key: Z or 1]"
                    )
                    self.secondary_neg_btn.setVisible(True)
                    self.secondary_neg_btn.setEnabled(True)
                    self.secondary_neg_btn.setText(f"✗ Set {opp_lbl} [C]")
                    self.secondary_neg_btn.setToolTip(
                        f"Flip this bout's label to {opp_lbl} [Key: C or 2]"
                    )
                else:
                    self.primary_pos_btn.setText(f"✓ Keep {rec.class_name} [C]")
                    self.primary_pos_btn.setToolTip(
                        f"Verify this {rec.class_name} label as correct [Key: C or 1]"
                    )
                    self.secondary_neg_btn.setVisible(True)
                    self.secondary_neg_btn.setEnabled(True)
                    self.secondary_neg_btn.setText(f"⇄ Switch to {beh} [Z]")
                    self.secondary_neg_btn.setToolTip(
                        f"Flip this bout's label to {beh} [Key: Z or 2]"
                    )

            self.trim_btn.setVisible(trim is not None)
            if trim is not None:
                self.trim_btn.setText(f"✂ Trim to {trim[0]}\u2013{trim[1]} [T]")
                self.trim_btn.setToolTip(
                    f"Trim sloppy start/end frames to clean span {trim[0]}\u2013{trim[1]} [Key: T]"
                )
            self.split_btn.setVisible(split_f is not None)
            if split_f is not None:
                self.split_btn.setText(f"✂ Split @ {split_f}")
            self.delete_btn.setVisible(True)
            self.delete_btn.setEnabled(True)
            self.delete_btn.setText("○ No Label [X]")
            self.delete_btn.setToolTip(
                "Clear this human label back to unlabeled [Key: X / Delete / Backspace]"
            )

        if redraw_plot:
            self._draw_inspector_plot()

    def _draw_inspector_plot(self) -> None:
        if self._report is None or self._selected_bout_idx is None:
            return
        b_idx = self._selected_bout_idx
        rec = self._report.feature_matrix.records[b_idx]
        diag = self._report.diagnostics
        emb = self._report.embedding
        fm = self._report.feature_matrix

        ax = self.insp_ax
        ax.clear()
        ax.set_aspect("auto", adjustable="box")
        ax.set_axis_on()

        if self._inspector_view == "prob":
            p_curve = fm.frame_probs_by_bout.get(rec.bout_id)
            frames = np.arange(rec.start_frame, rec.end_frame + 1)
            if p_curve is not None and len(p_curve) == len(frames):
                ax.plot(frames, p_curve, color="#1E40AF", linewidth=2.2, label="Class Probability")
                ax.fill_between(frames, 0.0, p_curve, color="#DBEAFE", alpha=0.45)
            else:
                mean_p = float(diag.oof_mean_probs[b_idx])
                ax.plot(
                    [rec.start_frame, rec.end_frame],
                    [mean_p, mean_p],
                    color="#1E40AF",
                    linewidth=2.2,
                    label=f"Mean Probability ({mean_p:.2f})",
                )

            ax.axhline(0.50, color="#64748B", linestyle=":", linewidth=1.1, label="0.50 Threshold")
            trim = diag.suggested_trim_bounds[b_idx]
            if trim is not None:
                ax.axvspan(trim[0], trim[1], color="#F1F5F9", alpha=0.85, zorder=0)
                ax.axvline(
                    trim[0],
                    color="#B91C1C",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Clean Core ({trim[0]}\u2013{trim[1]})",
                )
                ax.axvline(trim[1], color="#B91C1C", linestyle="--", linewidth=1.5)
            split_f = diag.suggested_split_frame[b_idx]
            if split_f is not None:
                ax.axvline(
                    split_f,
                    color="#B91C1C",
                    linestyle="-.",
                    linewidth=1.7,
                    label=f"Split @ {split_f}",
                )
            if self._scrubbed_frame is not None:
                ax.axvline(
                    self._scrubbed_frame,
                    color="#0F766E",
                    linewidth=1.8,
                    label=f"Frame {self._scrubbed_frame}",
                )

            ax.set_ylim(-0.04, 1.06)
            ax.set_xlabel("Video Frame (click to scrub)", fontsize=10.5, fontweight="bold")
            ax.set_ylabel("Probability", fontsize=10.5, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="#E2E8F0", linewidth=0.8)
            ax.legend(
                loc="best", fontsize=8.8, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1"
            )

        elif self._inspector_view == "feat":
            top_diffs = (
                emb.top_z_diffs_per_bout[b_idx] if b_idx < len(emb.top_z_diffs_per_bout) else []
            )
            if top_diffs:
                top_k = top_diffs[:7][::-1]
                names = [n[:28] for n, _ in top_k]
                zvals = [float(z) for _, z in top_k]
                colors = ["#B91C1C" if abs(z) >= 2.0 else "#1E40AF" for z in zvals]
                ax.barh(range(len(top_k)), zvals, color=colors, height=0.60)
                ax.axvline(0.0, color="#334155", linewidth=1.0)
                ax.set_yticks(range(len(top_k)))
                ax.set_yticklabels(names, fontsize=9.5)
            ax.set_xlabel(
                f"Deviation vs. Class Median (SD) • Sub-Type C{int(emb.cluster_labels[b_idx])}",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="x", color="#E2E8F0", linewidth=0.8)

        elif self._inspector_view == "clusters":
            z_mat = emb.cluster_z_matrix
            z_names = emb.cluster_z_feature_names
            if z_mat is None or z_mat.size == 0 or not z_names:
                ax.text(
                    0.5,
                    0.5,
                    "Need >= 4 bouts to compute Cluster Profiles.",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="#475569",
                    transform=ax.transAxes,
                )
            else:
                n_clusters = z_mat.shape[0]
                top_m = min(8, z_mat.shape[1], len(z_names))
                mat = np.asarray(z_mat[:, :top_m], dtype=np.float32)
                feat_cols = z_names[:top_m]
                active_cid = (
                    int(emb.cluster_labels[b_idx]) if b_idx < len(emb.cluster_labels) else 1
                )

                ax.imshow(
                    mat,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-2.5,
                    vmax=2.5,
                    interpolation="nearest",
                )
                for r_i in range(n_clusters):
                    for c_j in range(top_m):
                        val = float(mat[r_i, c_j])
                        txt_col = "#FFFFFF" if abs(val) >= 1.35 else "#0F172A"
                        ax.text(
                            c_j,
                            r_i,
                            f"{val:+.1f}",
                            ha="center",
                            va="center",
                            fontsize=8.3,
                            fontweight="bold" if (r_i + 1 == active_cid) else "normal",
                            color=txt_col,
                        )

                if 1 <= active_cid <= n_clusters:
                    ax.add_patch(
                        Rectangle(
                            (-0.49, (active_cid - 1) - 0.48),
                            top_m - 0.02,
                            0.96,
                            fill=False,
                            edgecolor="#0F172A",
                            linewidth=2.2,
                            zorder=5,
                        )
                    )

                show_dom = (
                    emb.cage_skew_p is not None
                    and emb.cage_skew_cramers_v is not None
                    and emb.cage_skew_p < 0.005
                    and emb.cage_skew_cramers_v >= 0.35
                )
                y_labels: list[str] = []
                for c_id in range(1, n_clusters + 1):
                    n_c = emb.cluster_counts.get(c_id, int(np.sum(emb.cluster_labels == c_id)))
                    prefix = "▶ " if c_id == active_cid else ""
                    dom = emb.dominant_cage_by_cluster.get(c_id)
                    if show_dom and dom is not None and dom[1] >= 0.55:
                        dom_lbl = dom[0].replace("cage_", "c")[:14]
                        y_labels.append(
                            f"{prefix}C{c_id} (n={n_c}, {dom_lbl} {dom[1] * 100:.0f}%)"
                        )
                    else:
                        y_labels.append(f"{prefix}C{c_id} (n={n_c})")
                ax.set_yticks(range(n_clusters))
                ax.set_yticklabels(y_labels, fontsize=9.0, fontweight="bold")

                x_labels: list[str] = []
                for fn in feat_cols:
                    eta2_pct = emb.cluster_feature_eta2.get(fn, 0.0) * 100.0
                    short = _short_feature_label(fn, max_len=16)
                    x_labels.append(f"{short}\n(η²={eta2_pct:.0f}%)" if eta2_pct >= 1.0 else short)
                ax.set_xticks(range(top_m))
                ax.set_xticklabels(x_labels, fontsize=8.0, rotation=24, ha="right")
                ax.set_title(
                    f"Sub-Type Cluster Profiles (Median Z-SD) • Click row to filter C1\u2013C{n_clusters}",
                    fontsize=10.2,
                    fontweight="bold",
                    color="#0F172A",
                    pad=6.0,
                )
                for sp in ax.spines.values():
                    sp.set_edgecolor("#CBD5E1")
                    sp.set_linewidth(0.8)

        else:
            try:
                traj = load_bout_pose_trajectory(
                    self._project,
                    rec.video,
                    rec.identity,
                    rec.start_frame,
                    rec.end_frame,
                )
                ax.plot(
                    traj.centroid_xy[:, 0],
                    traj.centroid_xy[:, 1],
                    color="#1E40AF",
                    linewidth=2.0,
                    label=f"{self._animal_display_text(rec.video, rec.identity)} Body",
                )
                ax.plot(
                    traj.nose_xy[:, 0],
                    traj.nose_xy[:, 1],
                    color="#B91C1C",
                    linewidth=1.2,
                    alpha=0.82,
                    label="Nose",
                )
                for pid, p_xy in traj.partner_centroids_xy.items():
                    ax.plot(
                        p_xy[:, 0],
                        p_xy[:, 1],
                        color="#0F766E",
                        linestyle="--",
                        linewidth=1.3,
                        alpha=0.75,
                        label=f"Partner {self._animal_display_text(rec.video, int(pid)).removeprefix('Animal ')}",
                    )
                if traj.lixit_xy is not None and len(traj.lixit_xy) > 0:
                    ax.scatter(
                        traj.lixit_xy[:, 0],
                        traj.lixit_xy[:, 1],
                        c="#475569",
                        marker="D",
                        s=55,
                        label="Lixit",
                    )
                ax.invert_yaxis()
                ax.set_aspect("equal", adjustable="datalim")
                ax.legend(loc="best", fontsize=8.5)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Pose trajectory unavailable:\n{exc}", ha="center", va="center")
            ax.set_xlabel("Arena X (px)", fontsize=10.5, fontweight="bold")
            ax.set_ylabel("Arena Y (px)", fontsize=10.5, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="#E2E8F0", linewidth=0.8)

        self.insp_canvas.mark_layout_dirty()
        self.insp_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Expandable / Zoomable 2D Map & Multi-Method Cohort Selection
    # ------------------------------------------------------------------
    def _clear_custom_selection_state(self) -> None:
        self._lasso_selected_indices.clear()
        self._custom_selection_label = ""
        self._custom_selection_preserve_order = False
        self.clear_box_btn.setVisible(False)

    def _on_drag_mode_pill_clicked(self, pill_id: int) -> None:
        mode: ScatterDragMode = "box" if pill_id == 0 else ("lasso" if pill_id == 1 else "zoom")
        self.scatter_canvas.set_drag_mode(mode)

    def _sync_zoom_reset_button(self) -> None:
        is_zoomed = self._scatter_view_limits is not None
        self.map_reset_zoom_btn.setEnabled(is_zoomed)
        if is_zoomed:
            self.map_reset_zoom_btn.setStyleSheet(
                "QPushButton { padding: 2px 7px; border: 1px solid #93C5FD; border-radius: 4px; "
                "background-color: #DBEAFE; color: #1E40AF; font-weight: 700; font-size: 11.5px; }"
                "QPushButton:hover { background-color: #BFDBFE; }"
            )
        else:
            self.map_reset_zoom_btn.setStyleSheet(
                "QPushButton { padding: 2px 7px; border: 1px solid #CBD5E1; border-radius: 4px; "
                "background-color: #FFFFFF; color: #334155; font-weight: 700; font-size: 11.5px; }"
                "QPushButton:disabled { color: #CBD5E1; border-color: #E2E8F0; }"
            )

    def _apply_scatter_zoom_limits(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        *,
        fast_preview: bool = False,
    ) -> None:
        """Update 2D scatter viewport cleanly without rebuilding Matplotlib collections."""
        if self._report is None and self._video_report is None:
            return
        if self._scatter_default_limits is None:
            dx0, dx1 = (float(v) for v in self.scatter_ax.get_xlim())
            dy0, dy1 = (float(v) for v in self.scatter_ax.get_ylim())
            self._scatter_default_limits = ((dx0, dx1), (dy0, dy1))

        (def_x0, def_x1), (def_y0, def_y1) = self._scatter_default_limits
        def_w = max(1e-3, abs(def_x1 - def_x0))
        def_h = max(1e-3, abs(def_y1 - def_y0))
        is_inverted_y = def_y0 > def_y1

        x0, x1 = sorted((float(xlim[0]), float(xlim[1])))
        y0, y1 = sorted((float(ylim[0]), float(ylim[1])))
        w = min(max(x1 - x0, 1e-3 * def_w), 5.0 * def_w)
        h = min(max(y1 - y0, 1e-3 * def_h), 5.0 * def_h)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        # Keep center within a generous margin around full data bounds so user never gets lost
        cx = min(max(cx, min(def_x0, def_x1) - def_w), max(def_x0, def_x1) + def_w)
        cy = min(max(cy, min(def_y0, def_y1) - def_h), max(def_y0, def_y1) + def_h)

        new_xlim = (cx - 0.5 * w, cx + 0.5 * w)
        new_ylim = (cy + 0.5 * h, cy - 0.5 * h) if is_inverted_y else (cy - 0.5 * h, cy + 0.5 * h)
        self._scatter_view_limits = (new_xlim, new_ylim)
        self.scatter_ax.set_xlim(new_xlim)
        self.scatter_ax.set_ylim(new_ylim)
        self._sync_zoom_reset_button()
        if fast_preview:
            self.scatter_canvas.draw_viewport_preview()
        else:
            self.scatter_canvas.draw_idle()

    def _on_scatter_wheel_zoom(self, x_anchor: float, y_anchor: float, scale: float) -> None:
        if self._report is None and self._video_report is None:
            return
        x0, x1 = (float(v) for v in self.scatter_ax.get_xlim())
        y0, y1 = (float(v) for v in self.scatter_ax.get_ylim())
        nx0 = x_anchor - (x_anchor - x0) * scale
        nx1 = x_anchor + (x1 - x_anchor) * scale
        ny0 = y_anchor - (y_anchor - y0) * scale
        ny1 = y_anchor + (y1 - y_anchor) * scale
        self._apply_scatter_zoom_limits((nx0, nx1), (ny0, ny1), fast_preview=True)

    def _zoom_scatter_by_step(self, scale: float) -> None:
        if self._report is None and self._video_report is None:
            return
        x0, x1 = (float(v) for v in self.scatter_ax.get_xlim())
        y0, y1 = (float(v) for v in self.scatter_ax.get_ylim())
        xa = 0.5 * (x0 + x1)
        ya = 0.5 * (y0 + y1)
        if self._mode == "videos":
            if (
                self._selected_video_name is not None
                and self._video_scatter_coords is not None
                and self._selected_video_name in self._video_scatter_names
            ):
                v_i = self._video_scatter_names.index(self._selected_video_name)
                sx, sy = (
                    float(self._video_scatter_coords[v_i, 0]),
                    float(self._video_scatter_coords[v_i, 1]),
                )
                if min(x0, x1) <= sx <= max(x0, x1) and min(y0, y1) <= sy <= max(y0, y1):
                    xa, ya = sx, sy
        elif self._report is not None and self._selected_bout_idx is not None:
            coords, _, _ = self._current_2d_coords()
            if 0 <= self._selected_bout_idx < len(coords):
                sx, sy = (
                    float(coords[self._selected_bout_idx, 0]),
                    float(coords[self._selected_bout_idx, 1]),
                )
                if min(x0, x1) <= sx <= max(x0, x1) and min(y0, y1) <= sy <= max(y0, y1):
                    xa, ya = sx, sy
        self._on_scatter_wheel_zoom(xa, ya, scale)

    def _on_scatter_zoom_box(self, x0: float, x1: float, y0: float, y1: float) -> None:
        if (self._report is None and self._video_report is None) or (
            (x1 - x0) < 1e-4 and (y1 - y0) < 1e-4
        ):
            return
        self._apply_scatter_zoom_limits((x0, x1), (y0, y1))

    def _on_scatter_pan_dragged(self, x0: float, x1: float, y0: float, y1: float) -> None:
        if self._report is None and self._video_report is None:
            return
        self._apply_scatter_zoom_limits((x0, x1), (y0, y1), fast_preview=True)

    def _reset_scatter_zoom(self) -> None:
        if self._scatter_view_limits is None:
            return
        self._scatter_view_limits = None
        if self._scatter_default_limits is not None:
            self.scatter_ax.set_xlim(self._scatter_default_limits[0])
            self.scatter_ax.set_ylim(self._scatter_default_limits[1])
            self._sync_zoom_reset_button()
            self.scatter_canvas.draw_idle()
        else:
            self._draw_scatter()

    def _toggle_expand_map(self, checked: bool | None = None) -> None:
        target = (not self._map_expanded) if checked is None else bool(checked)
        self._map_expanded = target
        self.expand_map_btn.blockSignals(True)
        self.expand_map_btn.setChecked(target)
        self.expand_map_btn.setText("⤡ Split [M]" if target else "⤢ Expand [M]")
        self.expand_map_btn.blockSignals(False)

        if target:
            u_sizes = self._upper_splitter.sizes()
            m_sizes = self._main_splitter.sizes()
            if sum(u_sizes) > 0 and self._card_panel.isVisible():
                self._saved_upper_sizes = list(u_sizes)
            if sum(m_sizes) > 0:
                self._saved_main_sizes = list(m_sizes)
            self._card_panel.setVisible(False)
            tot_h = max(560, sum(m_sizes) or self.height())
            self._main_splitter.setSizes([int(0.78 * tot_h), max(140, int(0.22 * tot_h))])
        else:
            self._card_panel.setVisible(True)
            if self._saved_upper_sizes and sum(self._saved_upper_sizes) > 0:
                self._upper_splitter.setSizes(self._saved_upper_sizes)
            else:
                tot_w = max(800, sum(self._upper_splitter.sizes()) or self.width())
                left_w = round((2.0 / 3.0) * tot_w)
                self._upper_splitter.setSizes([left_w, max(160, tot_w - left_w)])
            if self._saved_main_sizes and sum(self._saved_main_sizes) > 0:
                self._main_splitter.setSizes(self._saved_main_sizes)
            else:
                tot_h = max(560, sum(self._main_splitter.sizes()) or self.height())
                top_h = round(0.60 * tot_h)
                self._main_splitter.setSizes([top_h, max(140, tot_h - top_h)])

        self.scatter_canvas.mark_layout_dirty()
        self.insp_canvas.mark_layout_dirty()
        self.scatter_canvas.draw_idle()
        self.insp_canvas.draw_idle()

    def _apply_custom_selection(
        self,
        indices: list[int],
        label: str,
        *,
        additive: bool = False,
        preserve_order: bool = False,
    ) -> None:
        if self._report is None:
            return
        eligible_set = set(self._mode_eligible_indices())
        valid = [i for i in indices if i in eligible_set]
        if additive and self._lasso_selected_indices:
            seen: set[int] = set()
            merged: list[int] = []
            for gi in [*self._lasso_selected_indices, *valid]:
                if gi in eligible_set and gi not in seen:
                    seen.add(gi)
                    merged.append(gi)
            valid = merged
            label = "Map Selection"
            preserve_order = False

        if not valid:
            self.status_summary_label.setText(f"0 bouts matched '{label}' in the current mode.")
            return

        if not preserve_order:
            valid.sort(key=self._bout_priority_score, reverse=True)
        else:
            self._table_sort_col = None
            self.table.horizontalHeader().setSortIndicatorShown(False)

        self._lasso_selected_indices = valid
        self._custom_selection_label = label
        self._custom_selection_preserve_order = preserve_order
        self.clear_box_btn.setText(f"✕ {label} ({len(valid)})")
        self.clear_box_btn.setVisible(True)
        if preserve_order or self._selected_bout_idx not in valid:
            self._selected_bout_idx = valid[0]
        self._populate_table(redraw_scatter_on_change=False)
        self._draw_scatter()
        self.status_summary_label.setText(
            f"Selected {len(valid)} bouts ({label}) — press Esc to clear"
        )

    def _on_scatter_point_clicked(self, b_idx: int, *, seek_and_mark: bool = True) -> None:
        if self._mode == "videos":
            self._clear_video_table_mouse_press_state()
            if 0 <= b_idx < len(self._video_scatter_names):
                self._select_video_by_name(
                    self._video_scatter_names[b_idx], seek_in_jabs=seek_and_mark
                )
            return
        self._clear_table_mouse_press_state()
        if self._report is None or not (0 <= b_idx < len(self._report.feature_matrix.records)):
            return
        if self._mode == "predict" and b_idx not in set(self._mode_eligible_indices()):
            return
        self._selected_bout_idx = b_idx
        self._scrubbed_frame = None
        seek_epoch_before = self._bout_seek_epoch
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) == b_idx:
                prev_sup = self._suppress_auto_seek
                if not seek_and_mark:
                    self._suppress_auto_seek = True
                try:
                    selected_rows = {it.row() for it in self.table.selectedItems()}
                    if selected_rows == {r}:
                        if self.table.currentRow() != r:
                            self.table.blockSignals(True)
                            try:
                                self.table.setCurrentCell(r, 0)
                            finally:
                                self.table.blockSignals(False)
                        self._on_table_selection_changed()
                    else:
                        self.table.selectRow(r)
                finally:
                    self._suppress_auto_seek = prev_sup
                self._pending_click_already_sought_row = None
                if (
                    seek_and_mark
                    and self.auto_seek_action.isChecked()
                    and not self._suppress_auto_seek
                ):
                    if self._bout_seek_epoch == seek_epoch_before:
                        self._seek_bout_in_main_window(
                            self._report.feature_matrix.records[b_idx],
                            b_idx=b_idx,
                            bring_to_front=True,
                        )
                    else:
                        self._request_bring_main_window_to_front()
                return
        # Clicked an eligible point outside the active queue filter (e.g. clean bout in Audit mode)
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)
        self.table.blockSignals(False)
        self._update_bout_scatter_selection_ring()
        self._update_inspector_card()
        if seek_and_mark and self.auto_seek_action.isChecked():
            self._seek_bout_in_main_window(
                self._report.feature_matrix.records[b_idx],
                b_idx=b_idx,
                bring_to_front=True,
            )

    def _on_scatter_box_selected(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        additive: bool = False,
    ) -> None:
        if self._mode == "videos":
            self._on_video_scatter_box_selected(x0, x1, y0, y1, additive=additive)
            return
        if self._report is None or ((x1 - x0) < 1e-4 and (y1 - y0) < 1e-4):
            return
        coords, _, _ = self._current_2d_coords()
        q = self.search_input.text().strip().lower()
        eligible = [
            gi
            for gi in self._class_filtered_indices()
            if np.isfinite(coords[gi]).all() and (not q or self._bout_matches_search_query(gi, q))
        ]
        selected = [
            gi
            for gi in eligible
            if x0 <= float(coords[gi, 0]) <= x1 and y0 <= float(coords[gi, 1]) <= y1
        ]
        self._apply_custom_selection(selected, "Map Box", additive=additive)

    def _on_scatter_lasso_selected(self, poly_xy: Any, additive: bool = False) -> None:
        if self._mode == "videos":
            self._on_video_scatter_lasso_selected(poly_xy, additive=additive)
            return
        if self._report is None or poly_xy is None:
            return
        poly_arr = np.asarray(poly_xy, dtype=np.float64)
        if poly_arr.ndim != 2 or len(poly_arr) < 3:
            return
        coords, _, _ = self._current_2d_coords()
        q = self.search_input.text().strip().lower()
        eligible = [
            gi
            for gi in self._class_filtered_indices()
            if np.isfinite(coords[gi]).all() and (not q or self._bout_matches_search_query(gi, q))
        ]
        if not eligible:
            return
        el_arr = np.asarray(eligible, dtype=np.int32)
        inside = MplPath(poly_arr).contains_points(coords[el_arr])
        selected = [int(el_arr[k]) for k in np.flatnonzero(inside)]
        self._apply_custom_selection(selected, "Lasso", additive=additive)

    def _on_clear_box_filter(self) -> None:
        had_sel = bool(self._lasso_selected_indices)
        self._clear_custom_selection_state()
        if had_sel:
            self._populate_table(redraw_scatter_on_change=False)
            self._draw_scatter()
        else:
            self._populate_table()

    def _get_active_repr_matrix(self) -> np.ndarray:
        """Return cached `(N_bouts, D)` standardized/weighted feature matrix (`< 0.8 ms`)."""
        assert self._report is not None
        space = self._effective_space()
        lens = self._effective_feature_lens()
        cache_key = f"{space}:{lens}"
        cache_key_id = id(self._report)
        if (
            self._repr_matrix_cache is not None
            and self._repr_matrix_cache[0] == cache_key_id
            and self._repr_matrix_cache[1] == cache_key
        ):
            return self._repr_matrix_cache[2]

        fm = self._report.feature_matrix
        X_bout = fm.X_bout
        n_bouts, n_stats = X_bout.shape
        if n_bouts == 0:
            empty = np.zeros((0, 1), dtype=np.float32)
            self._repr_matrix_cache = (cache_key_id, cache_key, empty)
            return empty

        X_clean = np.nan_to_num(X_bout, nan=0.0, posinf=0.0, neginf=0.0)
        lens_mask = select_feature_lens_mask(fm.stat_names, lens)
        std_mask = lens_mask & (np.std(X_clean, axis=0) > 1e-8)
        if not np.any(std_mask):
            std_mask = lens_mask
        if not np.any(std_mask):
            std_mask = np.ones(n_stats, dtype=bool)
        X_sub = X_clean[:, std_mask]
        s_sorted = np.sort(X_sub, axis=0)
        med = (
            s_sorted[n_bouts // 2]
            if (n_bouts % 2 == 1)
            else 0.5 * (s_sorted[n_bouts // 2 - 1] + s_sorted[n_bouts // 2])
        )
        q25 = s_sorted[int(0.25 * max(0, n_bouts - 1))]
        q75 = s_sorted[int(0.75 * max(0, n_bouts - 1))]
        scale = np.maximum((q75 - q25) / 1.3489795, 0.5 * np.std(X_sub, axis=0))
        robust_sd = np.where(scale > 1e-5, scale, 1.0).astype(np.float32)
        X_scaled = np.clip(
            np.nan_to_num(
                (X_sub - med[None, :]) / robust_sd[None, :],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            -3.5,
            3.5,
        ).astype(np.float32)
        if space in ("classifier", "rf_proximity") or lens == "classifier_weighted":
            w_act = self._report.diagnostics.stat_feature_weights[std_mask]
            X_repr = X_scaled * np.sqrt(np.maximum(w_act, 1e-6))[None, :]
        else:
            X_repr = X_scaled
        self._repr_matrix_cache = (cache_key_id, cache_key, X_repr)
        return X_repr

    def _select_nearest_similar_bouts(
        self,
        k: int = 15,
        anchor_b_idx: int | None = None,
    ) -> None:
        """Select `anchor_b_idx` + its `k - 1` nearest kinematic neighbors in `< 1 ms`."""
        if self._report is None:
            return
        anchor = self._selected_bout_idx if anchor_b_idx is None else int(anchor_b_idx)
        records = self._report.feature_matrix.records
        if anchor is None or not (0 <= anchor < len(records)):
            return
        eligible = self._mode_eligible_indices()
        if not eligible:
            return

        space = self._effective_space()
        rf_prox = self._report.diagnostics.rf_proximity_matrix
        if (
            space == "rf_proximity"
            and rf_prox is not None
            and rf_prox.shape == (len(records), len(records))
        ):
            dists_all = 1.0 - np.asarray(rf_prox[anchor], dtype=np.float32)
        else:
            X_repr = self._get_active_repr_matrix()
            diff = X_repr - X_repr[anchor : anchor + 1]
            dists_all = np.linalg.norm(diff, axis=1)

        ranked_eligible = sorted(
            eligible, key=lambda gi: (0 if gi == anchor else 1, float(dists_all[gi]))
        )
        chosen = ranked_eligible[: max(1, int(k))]
        rec_anchor = records[anchor]
        self._apply_custom_selection(
            chosen,
            f"{len(chosen)} Similar to #{rec_anchor.bout_id}",
            preserve_order=True,
        )

    def _select_cluster_cohort(self, cluster_id: int) -> None:
        if self._report is None:
            return
        cl = self._report.embedding.cluster_labels
        eligible = self._mode_eligible_indices()
        chosen = [gi for gi in eligible if int(cl[gi]) == int(cluster_id)]
        self._apply_custom_selection(chosen, f"Cluster C{cluster_id}")

    def _select_same_track_cohort(self, b_idx: int | None = None) -> None:
        if self._report is None:
            return
        anchor = self._selected_bout_idx if b_idx is None else int(b_idx)
        records = self._report.feature_matrix.records
        if anchor is None or not (0 <= anchor < len(records)):
            return
        ref = records[anchor]
        eligible = self._mode_eligible_indices()
        chosen = sorted(
            (
                gi
                for gi in eligible
                if records[gi].video == ref.video
                and int(records[gi].identity) == int(ref.identity)
            ),
            key=lambda gi: records[gi].start_frame,
        )
        anim_lbl = self._animal_display_text(ref.video, ref.identity)
        self._apply_custom_selection(
            chosen,
            f"{_short_video_name(ref.video).split(' • ')[0]} {anim_lbl}",
            preserve_order=True,
        )

    def _select_same_video_cohort(self, b_idx: int | None = None) -> None:
        if self._report is None:
            return
        anchor = self._selected_bout_idx if b_idx is None else int(b_idx)
        records = self._report.feature_matrix.records
        if anchor is None or not (0 <= anchor < len(records)):
            return
        ref = records[anchor]
        eligible = self._mode_eligible_indices()
        chosen = sorted(
            (gi for gi in eligible if records[gi].video == ref.video),
            key=lambda gi: (int(records[gi].identity), records[gi].start_frame),
        )
        self._apply_custom_selection(
            chosen,
            _short_video_name(ref.video),
            preserve_order=True,
        )

    def _select_current_jabs_video_cohort(self) -> None:
        if self._report is None or self._central_widget is None:
            return
        loaded = getattr(self._central_widget, "loaded_video", None)
        if loaded is None:
            self.status_summary_label.setText("No video currently loaded in JABS Player.")
            return
        loaded_name = Path(loaded).name
        records = self._report.feature_matrix.records
        eligible = self._mode_eligible_indices()
        chosen = sorted(
            (gi for gi in eligible if Path(records[gi].video).name == loaded_name),
            key=lambda gi: (int(records[gi].identity), records[gi].start_frame),
        )
        self._apply_custom_selection(
            chosen,
            f"JABS Video: {_short_video_name(loaded_name)}",
            preserve_order=True,
        )

    def _quick_slice_matches(self, slice_key: str) -> tuple[list[int], str]:
        if self._report is None:
            return [], slice_key
        eligible = self._mode_eligible_indices()
        records = self._report.feature_matrix.records
        diag = self._report.diagnostics

        if slice_key == "high_doubt":
            return (
                [gi for gi in eligible if float(diag.confusion_scores[gi]) >= 0.65],
                "High Doubt ≥65%",
            )
        if slice_key == "borderline":
            return (
                [
                    gi
                    for gi in eligible
                    if 0.35 <= float(diag.oof_mean_probs[gi]) <= 0.65
                    or 0.45 <= float(diag.confusion_scores[gi]) < 0.65
                ],
                "Borderline Grey-Zone",
            )
        if slice_key == "boundary":
            return (
                [
                    gi
                    for gi in eligible
                    if (
                        diag.suggested_trim_bounds[gi] is not None
                        or diag.suggested_split_frame[gi] is not None
                        or records[gi].postproc_fragility >= 0.35
                    )
                ],
                "Boundary / Split",
            )
        if slice_key == "outliers":
            return (
                [
                    gi
                    for gi in eligible
                    if int(diag.outlier_votes[gi]) >= 3 or float(diag.outlier_scores[gi]) >= 0.78
                ],
                "Kinematic Outliers",
            )
        if slice_key == "ood":
            return (
                [gi for gi in eligible if float(diag.ood_scores[gi]) >= 0.80],
                "Novel / OOD",
            )
        if slice_key == "label_edges":
            return (
                [gi for gi in eligible if self._bout_is_label_halo(gi)],
                "Label Edge / Halo",
            )
        if slice_key == "short_flickers":
            return (
                [gi for gi in eligible if records[gi].duration_frames < 15],
                "Short Flickers <0.5s",
            )
        if slice_key == "long_bouts":
            return (
                [gi for gi in eligible if records[gi].duration_frames > 150],
                "Long Bouts >5.0s",
            )
        if slice_key.startswith("class:"):
            cls_name = slice_key.split(":", 1)[1]
            return (
                [gi for gi in eligible if records[gi].class_name == cls_name],
                f"Class: {cls_name}",
            )
        return [], slice_key

    def _select_quick_slice(self, slice_key: str) -> None:
        matched, label = self._quick_slice_matches(slice_key)
        self._apply_custom_selection(matched, label)

    def _populate_quick_select_menu(self) -> None:
        menu = self.quick_select_menu
        menu.clear()
        if self._report is None:
            act = menu.addAction("Loading bouts…")
            act.setEnabled(False)
            return

        records = self._report.feature_matrix.records
        emb = self._report.embedding
        eligible = self._mode_eligible_indices()

        # 1. Similarity & Animal/Video Context
        if self._selected_bout_idx is not None and 0 <= self._selected_bout_idx < len(records):
            sel_rec = records[self._selected_bout_idx]
            anim_lbl = self._animal_display_text(sel_rec.video, sel_rec.identity)
            act_sim = menu.addAction(f"🎯 15 Nearest Similar to #{sel_rec.bout_id} [N]")
            act_sim.triggered.connect(lambda: self._select_nearest_similar_bouts(15))

            n_trk = sum(
                1
                for gi in eligible
                if records[gi].video == sel_rec.video
                and int(records[gi].identity) == int(sel_rec.identity)
            )
            act_trk = menu.addAction(
                f"🐁 Same Animal Track ({_short_video_name(sel_rec.video).split(' • ')[0]} • {anim_lbl} — {n_trk})"
            )
            act_trk.setEnabled(n_trk > 0)
            act_trk.triggered.connect(
                lambda: self._select_same_track_cohort(self._selected_bout_idx)
            )

            n_vid = sum(1 for gi in eligible if records[gi].video == sel_rec.video)
            act_vid = menu.addAction(
                f"🏠 Same Video ({_short_video_name(sel_rec.video)} — {n_vid})"
            )
            act_vid.setEnabled(n_vid > 0)
            act_vid.triggered.connect(
                lambda: self._select_same_video_cohort(self._selected_bout_idx)
            )

        loaded = (
            getattr(self._central_widget, "loaded_video", None)
            if self._central_widget is not None
            else None
        )
        if loaded is not None:
            loaded_name = Path(loaded).name
            n_jvid = sum(1 for gi in eligible if Path(records[gi].video).name == loaded_name)
            act_jv = menu.addAction(
                f"🎬 Current Video in JABS Player ({_short_video_name(loaded_name)} — {n_jvid})"
            )
            act_jv.setEnabled(n_jvid > 0)
            act_jv.triggered.connect(self._select_current_jabs_video_cohort)

        menu.addSeparator()

        # 2. Smart Diagnostic & Duration Slices
        for key, title in (
            ("high_doubt", "High Disagreement / Doubt (≥65%)"),
            ("borderline", "Borderline Grey-Zone (Prob 35%\u201365%)"),
            ("label_edges", "Label Edge / Halo Fragments (Onset/Offset)"),
            ("boundary", "Boundary Trim & Chimeric Split Candidates"),
            ("outliers", "Kinematic Outliers (≥3/5 Votes)"),
            ("ood", "Novel / Out-of-Distribution Kinematics"),
            ("short_flickers", "Short Flickers (<0.5 s / <15 frames)"),
            ("long_bouts", "Long Bouts (>5.0 s / >150 frames)"),
        ):
            matched, _ = self._quick_slice_matches(key)
            act = menu.addAction(f"{title} ({len(matched)})")
            act.setEnabled(len(matched) > 0)
            act.triggered.connect(lambda _c=False, k=key: self._select_quick_slice(k))

        menu.addSeparator()

        # 3. Sub-Type Clusters submenu
        cl_menu = menu.addMenu(f"By Sub-Type Cluster (C1..C{emb.optimal_k})")
        for cid in sorted(set(emb.cluster_labels[eligible].tolist()) if eligible else []):
            cnt = sum(1 for gi in eligible if int(emb.cluster_labels[gi]) == int(cid))
            top_f = emb.cluster_top_features.get(int(cid), [])
            hint = f" — {_short_feature_label(top_f[0][0], 20)}" if top_f else ""
            act_c = cl_menu.addAction(f"Cluster C{cid} ({cnt}){hint[:28]}")
            act_c.triggered.connect(lambda _c=False, c=int(cid): self._select_cluster_cohort(c))

        # 4. Behavior Class submenu
        cls_menu = menu.addMenu("By Behavior Class")
        for cls_name in sorted({records[gi].class_name for gi in eligible}):
            cnt = sum(1 for gi in eligible if records[gi].class_name == cls_name)
            act_cls = cls_menu.addAction(f"{cls_name} ({cnt})")
            act_cls.triggered.connect(
                lambda _c=False, cn=cls_name: self._select_quick_slice(f"class:{cn}")
            )

        if self._lasso_selected_indices:
            menu.addSeparator()
            act_clr = menu.addAction("✕ Clear Active Selection [Esc]")
            act_clr.triggered.connect(self._on_clear_box_filter)

        menu.addSeparator()
        act_vids = menu.addAction(
            "🎬 Whole-Video Screener (Frames / Bouts / 0-Behavior / 50k) → 4. Videos"
        )
        act_vids.triggered.connect(self._switch_to_videos_mode)

    def _on_scatter_context_menu(self, global_pos: QtCore.QPoint, hit_b_idx: int) -> None:
        if self._mode == "videos":
            if hit_b_idx >= 0 and hit_b_idx < len(self._video_scatter_names):
                vname = self._video_scatter_names[hit_b_idx]
                self._select_video_by_name(vname, seek_in_jabs=False)
                self._show_video_context_menu(global_pos, vname)
            else:
                self._show_video_context_menu(global_pos, self._selected_video_name)
            return
        if self._report is None:
            return
        if hit_b_idx >= 0:
            self._on_scatter_point_clicked(hit_b_idx, seek_and_mark=False)
            self._show_bout_context_menu(global_pos, hit_b_idx)
        else:
            self._show_bout_context_menu(global_pos, self._selected_bout_idx)

    def _on_table_context_menu(self, local_pos: QtCore.QPoint) -> None:
        self._clear_table_mouse_press_state()
        if self._report is None:
            return
        row = self.table.rowAt(local_pos.y())
        b_idx = self._selected_bout_idx
        if row >= 0:
            it = self.table.item(row, 0)
            if it is not None:
                b_idx = int(it.data(Qt.ItemDataRole.UserRole))
                if self.table.currentRow() != row:
                    self._suppress_auto_seek = True
                    try:
                        self.table.selectRow(row)
                    finally:
                        self._suppress_auto_seek = False
                self._pending_click_already_sought_row = None
        self._show_bout_context_menu(self.table.viewport().mapToGlobal(local_pos), b_idx)

    def _show_bout_context_menu(self, global_pos: QtCore.QPoint, b_idx: int | None) -> None:
        if self._report is None:
            return
        menu = QtWidgets.QMenu(self)
        records = self._report.feature_matrix.records
        emb = self._report.embedding
        eligible = self._mode_eligible_indices()

        if b_idx is not None and 0 <= b_idx < len(records):
            rec = records[b_idx]
            cid = int(emb.cluster_labels[b_idx])
            anim_lbl = self._animal_display_text(rec.video, rec.identity)

            act_k15 = menu.addAction(f"🎯 Select 15 Nearest Similar to #{rec.bout_id} [N]")
            act_k15.triggered.connect(lambda: self._select_nearest_similar_bouts(15, b_idx))

            act_k30 = menu.addAction(f"🎯 Select 30 Nearest Similar to #{rec.bout_id}")
            act_k30.triggered.connect(lambda: self._select_nearest_similar_bouts(30, b_idx))

            n_cid = sum(1 for gi in eligible if int(emb.cluster_labels[gi]) == cid)
            act_cl = menu.addAction(f"🧩 Select Sub-Type Cluster C{cid} ({n_cid} bouts)")
            act_cl.setEnabled(n_cid > 0)
            act_cl.triggered.connect(lambda: self._select_cluster_cohort(cid))

            n_trk = sum(
                1
                for gi in eligible
                if records[gi].video == rec.video
                and int(records[gi].identity) == int(rec.identity)
            )
            act_trk = menu.addAction(
                f"🐁 Select Same Animal Track ({anim_lbl} — {n_trk} bouts, chronological)"
            )
            act_trk.setEnabled(n_trk > 0)
            act_trk.triggered.connect(lambda: self._select_same_track_cohort(b_idx))

            n_vid = sum(1 for gi in eligible if records[gi].video == rec.video)
            act_vid = menu.addAction(
                f"🏠 Select Same Video ({_short_video_name(rec.video)} — {n_vid} bouts)"
            )
            act_vid.setEnabled(n_vid > 0)
            act_vid.triggered.connect(lambda: self._select_same_video_cohort(b_idx))

            act_cls = menu.addAction(f"🔍 Filter Class: {rec.class_name}")
            act_cls.triggered.connect(
                lambda _c=False, cn=rec.class_name: self._set_class_filter(f"class:{cn}")
            )

            menu.addSeparator()
            is_rev = rec.bout_id in self._reviewed_bout_ids
            act_ex = menu.addAction(
                f"↺ Unmark Bout #{rec.bout_id} as Examined [V]"
                if is_rev
                else f"✓ Mark Bout #{rec.bout_id} as Examined [V]"
            )
            act_ex.triggered.connect(lambda: self._toggle_bout_examined(b_idx))

        if self._lasso_selected_indices:
            n_sel = len(self._lasso_selected_indices)
            act_ex_all = menu.addAction(f"✓ Mark All {n_sel} Selected Bouts as Examined")
            act_ex_all.triggered.connect(
                lambda: self._mark_cohort_examined(list(self._lasso_selected_indices))
            )

        if (
            self._scatter_view_limits is not None
            or self._lasso_selected_indices
            or self._class_filter != "all"
        ):
            menu.addSeparator()
        if self._scatter_view_limits is not None:
            act_rz = menu.addAction("⟲ Reset 2D Map Zoom [0]")
            act_rz.triggered.connect(self._reset_scatter_zoom)
        if self._lasso_selected_indices:
            act_clr = menu.addAction("✕ Clear Active Selection [Esc]")
            act_clr.triggered.connect(self._on_clear_box_filter)
        if self._class_filter != "all":
            act_all_cls = menu.addAction("✕ Show All Classes")
            act_all_cls.triggered.connect(lambda: self._set_class_filter("all"))

        if menu.actions():
            menu.exec(global_pos)

    # ------------------------------------------------------------------
    # Instant (< 25 ms) Guarded 1-Click Triage + Auto-Advance + Undo
    # ------------------------------------------------------------------
    def _live_pose_for_video(self, video: str) -> Any | None:
        """Return already-loaded `CentralWidget._pose_est` when `video` is open in JABS (`0 ms`)."""
        if self._central_widget is None:
            return None
        loaded = getattr(self._central_widget, "loaded_video", None)
        pose_live = getattr(self._central_widget, "_pose_est", None)
        if loaded is None or pose_live is None:
            return None
        try:
            if Path(loaded).name == Path(video).name:
                return pose_live
        except TypeError:
            return None
        return None

    def _finish_fast_inplace_triage_ui(
        self,
        edited_b_idx: int,
        changed_b_indices: list[int],
        *,
        next_b_idx: int | None,
        removed_from_table: bool = False,
        redraw_inspector_plot: bool = False,
        select_span_on_seek: bool = True,
        force_seek_same_track: bool = False,
    ) -> None:
        """Update table rows, inspector, and scatter in `< 5 ms` without rebuilding all rows."""
        assert self._report is not None
        records = self._report.feature_matrix.records
        rec = records[edited_b_idx]
        auto_adv = self.auto_advance_action.isChecked() and next_b_idx is not None

        if self._lasso_selected_indices:
            eligible_set = set(self._mode_eligible_indices())
            n_act = sum(1 for gi in self._lasso_selected_indices if gi in eligible_set)
            if n_act > 0:
                self.clear_box_btn.setText(f"✕ {self._custom_selection_label} ({n_act})")
                self.clear_box_btn.setVisible(True)
            else:
                self._clear_custom_selection_state()

        self._update_queue_filter_counts()

        if removed_from_table:
            if auto_adv:
                self._selected_bout_idx = next_b_idx
            else:
                self._selected_bout_idx = None
            row_to_rm = self._find_table_row_for_bout(edited_b_idx)
            self.table.blockSignals(True)
            self._suppress_auto_seek = True
            try:
                if row_to_rm is not None:
                    self.table.removeRow(row_to_rm)
                if self._selected_bout_idx is not None:
                    nxt_row = self._find_table_row_for_bout(self._selected_bout_idx)
                    if nxt_row is not None:
                        self.table.selectRow(nxt_row)
                else:
                    self.table.clearSelection()
                    self.table.setCurrentCell(-1, -1)
            finally:
                self._suppress_auto_seek = False
                self.table.blockSignals(False)
            if self._selected_bout_idx is not None:
                self._update_inspector_card(redraw_plot=True)
            else:
                self._update_empty_card_state()
        else:
            # In "Reviewed ✓" filter (or when a flipped bout leaves the active Class Filter),
            # rebuild visible table rows so membership stays synchronized
            edited_row = self._find_table_row_for_bout(edited_b_idx)
            if (
                edited_row is None
                or self._queue_filter == "reviewed"
                or not self._bout_matches_class_filter(edited_b_idx)
            ):
                if auto_adv:
                    self._selected_bout_idx = next_b_idx
                self._populate_table(redraw_scatter_on_change=False)
            else:
                self._update_table_rows_in_place(changed_b_indices)
                if auto_adv:
                    self._selected_bout_idx = next_b_idx
                    nxt_row = self._find_table_row_for_bout(next_b_idx)
                    self.table.blockSignals(True)
                    self._suppress_auto_seek = True
                    try:
                        if nxt_row is not None:
                            self.table.selectRow(nxt_row)
                    finally:
                        self._suppress_auto_seek = False
                        self.table.blockSignals(False)
                    self._update_inspector_card(redraw_plot=True)
                else:
                    self._update_inspector_card(
                        sync_mc_target=False,
                        redraw_plot=redraw_inspector_plot,
                    )

        self._schedule_scatter_redraw()

        if self.auto_seek_action.isChecked():
            if auto_adv and self._selected_bout_idx is not None:
                self._seek_bout_in_main_window(
                    records[self._selected_bout_idx],
                    b_idx=self._selected_bout_idx,
                    mark_viewed=True,
                )
            elif removed_from_table:
                if not self._is_main_window_on_bout_track(rec):
                    self._seek_bout_in_main_window(rec, select_span=False, mark_viewed=False)
            elif not self._is_main_window_on_bout_track(rec) or force_seek_same_track:
                self._seek_bout_in_main_window(
                    rec,
                    select_span=select_span_on_seek,
                    mark_viewed=False,
                )

    def _mark_current_bout_kept(self) -> None:
        """Verify and keep an existing human label as-is (`< 2 ms`, persists Examined flag, Undoable)."""
        if self._report is None or self._selected_bout_idx is None:
            return
        b_idx = self._selected_bout_idx
        rec = self._report.feature_matrix.records[b_idx]
        self._ensure_origin_snapshot(rec)
        diag = self._report.diagnostics
        next_b_idx = self._peek_next_bout_after_current(b_idx)
        prev_examined_entry = self._examined_store.find_match(rec)

        ui_snap: dict[str, Any] = {
            "disk_mutated": False,
            "b_idx": b_idx,
            "rec_before": replace(rec),
            "was_reviewed": rec.bout_id in self._reviewed_bout_ids,
            "was_examined_only": rec.bout_id in self._examined_only_bout_ids,
            "was_pinned": rec.bout_id in self._pinned_needs_review_bout_ids,
            "was_deleted": rec.bout_id in self._deleted_bout_ids,
            "was_unlabeled": rec.bout_id in self._unlabeled_bout_ids,
            "prev_note": self._saved_notes_by_bout.get(rec.bout_id),
            "prev_examined_entry": prev_examined_entry,
            "prev_trim": diag.suggested_trim_bounds[b_idx],
            "prev_split": diag.suggested_split_frame[b_idx],
            "track_key": None,
            "prev_track_count": None,
            "paired_overlap_ids": [],
            "paired_pinned_ids": [],
        }
        self._ui_undo_history.append(ui_snap)
        if len(self._ui_undo_history) > 200:
            del self._ui_undo_history[:-200]
        self.undo_btn.setEnabled(True)

        self._unlabeled_bout_ids.discard(rec.bout_id)
        self._examined_only_bout_ids.discard(rec.bout_id)
        self._reviewed_bout_ids.add(rec.bout_id)
        self._pinned_needs_review_bout_ids.add(rec.bout_id)
        note = f"✓ Kept {rec.class_name}"
        self._saved_notes_by_bout[rec.bout_id] = note
        self._examined_store.mark(rec, action="confirmed", note=note)

        self._finish_fast_inplace_triage_ui(
            b_idx,
            [b_idx],
            next_b_idx=next_b_idx,
            removed_from_table=False,
            redraw_inspector_plot=False,
        )

        anim_lbl = self._animal_display_text(rec.video, rec.identity)
        self.status_summary_label.setText(
            f"{note} on {_short_video_name(rec.video)} ({anim_lbl}) — Undo [⌘Z]"
        )

    def _mark_current_bout_unlabeled(self) -> None:
        """Mark an unlabeled/predicted bout as intentionally No Label (`< 2 ms`, persists Examined flag)."""
        if self._report is None or self._selected_bout_idx is None:
            return
        b_idx = self._selected_bout_idx
        rec = self._report.feature_matrix.records[b_idx]
        self._ensure_origin_snapshot(rec)
        diag = self._report.diagnostics
        next_b_idx = self._peek_next_bout_after_current(b_idx)
        prev_examined_entry = self._examined_store.find_match(rec)

        ui_snap: dict[str, Any] = {
            "disk_mutated": False,
            "b_idx": b_idx,
            "rec_before": replace(rec),
            "was_reviewed": rec.bout_id in self._reviewed_bout_ids,
            "was_examined_only": rec.bout_id in self._examined_only_bout_ids,
            "was_pinned": rec.bout_id in self._pinned_needs_review_bout_ids,
            "was_deleted": rec.bout_id in self._deleted_bout_ids,
            "was_unlabeled": rec.bout_id in self._unlabeled_bout_ids,
            "prev_note": self._saved_notes_by_bout.get(rec.bout_id),
            "prev_examined_entry": prev_examined_entry,
            "prev_trim": diag.suggested_trim_bounds[b_idx],
            "prev_split": diag.suggested_split_frame[b_idx],
            "track_key": None,
            "prev_track_count": None,
            "paired_overlap_ids": [],
            "paired_pinned_ids": [],
        }
        self._ui_undo_history.append(ui_snap)
        if len(self._ui_undo_history) > 200:
            del self._ui_undo_history[:-200]
        self.undo_btn.setEnabled(True)

        self._unlabeled_bout_ids.add(rec.bout_id)
        self._examined_only_bout_ids.discard(rec.bout_id)
        self._reviewed_bout_ids.add(rec.bout_id)
        self._pinned_needs_review_bout_ids.add(rec.bout_id)
        note = "○ No label (left unlabeled)"
        self._saved_notes_by_bout[rec.bout_id] = note
        self._examined_store.mark(rec, action="no_label", note=note)

        self._finish_fast_inplace_triage_ui(
            b_idx,
            [b_idx],
            next_b_idx=next_b_idx,
            removed_from_table=False,
            redraw_inspector_plot=False,
        )

        anim_lbl = self._animal_display_text(rec.video, rec.identity)
        self.status_summary_label.setText(
            f"{note} on {_short_video_name(rec.video)} ({anim_lbl}) — Undo [⌘Z]"
        )

    def _on_toggle_examined_v_clicked(self) -> None:
        """Key `[V]`: Toggle persistent Examined flag on the selected bout without editing labels."""
        if self._report is None or self._selected_bout_idx is None or self._mode == "videos":
            return
        self._toggle_bout_examined(self._selected_bout_idx)

    def _toggle_bout_examined(self, b_idx: int) -> None:
        if self._report is None:
            return
        records = self._report.feature_matrix.records
        if not (0 <= b_idx < len(records)):
            return
        rec = records[b_idx]
        self._ensure_origin_snapshot(rec)
        diag = self._report.diagnostics
        next_b_idx = self._peek_next_bout_after_current(b_idx)
        prev_examined_entry = self._examined_store.find_match(rec)
        was_rev = rec.bout_id in self._reviewed_bout_ids

        ui_snap: dict[str, Any] = {
            "disk_mutated": False,
            "b_idx": b_idx,
            "rec_before": replace(rec),
            "was_reviewed": was_rev,
            "was_examined_only": rec.bout_id in self._examined_only_bout_ids,
            "was_pinned": rec.bout_id in self._pinned_needs_review_bout_ids,
            "was_deleted": rec.bout_id in self._deleted_bout_ids,
            "was_unlabeled": rec.bout_id in self._unlabeled_bout_ids,
            "prev_note": self._saved_notes_by_bout.get(rec.bout_id),
            "prev_examined_entry": prev_examined_entry,
            "prev_trim": diag.suggested_trim_bounds[b_idx],
            "prev_split": diag.suggested_split_frame[b_idx],
            "track_key": None,
            "prev_track_count": None,
            "paired_overlap_ids": [],
            "paired_pinned_ids": [],
        }
        self._ui_undo_history.append(ui_snap)
        if len(self._ui_undo_history) > 200:
            del self._ui_undo_history[:-200]
        self.undo_btn.setEnabled(True)

        if was_rev:
            self._reviewed_bout_ids.discard(rec.bout_id)
            self._examined_only_bout_ids.discard(rec.bout_id)
            self._unlabeled_bout_ids.discard(rec.bout_id)
            self._pinned_needs_review_bout_ids.discard(rec.bout_id)
            self._saved_notes_by_bout.pop(rec.bout_id, None)
            self._examined_store.unmark(rec)
            status_txt = "↺ Unmarked Examined"
        else:
            self._reviewed_bout_ids.add(rec.bout_id)
            self._examined_only_bout_ids.add(rec.bout_id)
            self._pinned_needs_review_bout_ids.add(rec.bout_id)
            note = "✓ Examined (saved)"
            self._saved_notes_by_bout[rec.bout_id] = note
            self._examined_store.mark(rec, action="examined", note=note)
            status_txt = "✓ Marked Examined"

        self._finish_fast_inplace_triage_ui(
            b_idx,
            [b_idx],
            next_b_idx=next_b_idx if not was_rev else None,
            removed_from_table=False,
            redraw_inspector_plot=False,
        )
        anim_lbl = self._animal_display_text(rec.video, rec.identity)
        self.status_summary_label.setText(
            f"{status_txt} on {_short_video_name(rec.video)} ({anim_lbl}) — Undo [⌘Z]"
        )

    def _mark_cohort_examined(self, indices: list[int]) -> None:
        """Batch-mark a selected cohort of bouts as Examined with a single disk write."""
        if self._report is None or not indices:
            return
        records = self._report.feature_matrix.records
        valid_idxs = [
            gi
            for gi in indices
            if 0 <= gi < len(records) and records[gi].bout_id not in self._deleted_bout_ids
        ]
        if not valid_idxs:
            return
        note = "✓ Examined (saved)"
        to_mark: list[BoutRecord] = []
        for gi in valid_idxs:
            rec = records[gi]
            self._ensure_origin_snapshot(rec)
            self._reviewed_bout_ids.add(rec.bout_id)
            self._examined_only_bout_ids.add(rec.bout_id)
            self._pinned_needs_review_bout_ids.add(rec.bout_id)
            self._saved_notes_by_bout[rec.bout_id] = note
            to_mark.append(rec)
        self._examined_store.mark_many(to_mark, action="examined", note=note)
        self._update_queue_filter_counts()
        self._update_table_rows_in_place(valid_idxs)
        self._update_inspector_card(sync_mc_target=False, redraw_plot=False)
        self._schedule_scatter_redraw()
        self.status_summary_label.setText(
            f"✓ Marked {len(to_mark)} selected bout(s) as Examined (persisted for this project)"
        )

    def _on_reset_examined_flags(self, *, all_behaviors: bool = False) -> None:
        """Clear persisted Examined flags for the active behavior or the entire project."""
        scope_beh = None if all_behaviors else self._selected_single_behavior()
        scope_desc = (
            "all behaviors in this project"
            if (all_behaviors or scope_beh is None)
            else f"behavior '{scope_beh}'"
        )
        existing_cnt = self._examined_store.count(behavior=scope_beh)
        if existing_cnt == 0:
            self.status_summary_label.setText(
                f"No saved Examined flags to reset for {scope_desc}."
            )
            return
        confirmed = MessageDialog.confirm(
            self,
            title="Reset Examined Flags",
            message=(
                f"Clear {existing_cnt} saved Examined flag(s) for {scope_desc}?\n\n"
                "Your manual JABS annotations are not affected — those bouts will simply "
                "re-enter the Needs Review queue."
            ),
        )
        if not confirmed:
            return
        cleared = self._examined_store.clear(behavior=scope_beh)
        if self._report is not None:
            for r in self._report.feature_matrix.records:
                if scope_beh is not None:
                    r_beh = str(getattr(r, "origin_behavior", "") or r.behavior)
                    if r_beh != scope_beh:
                        continue
                self._reviewed_bout_ids.discard(r.bout_id)
                self._examined_only_bout_ids.discard(r.bout_id)
                self._unlabeled_bout_ids.discard(r.bout_id)
                self._pinned_needs_review_bout_ids.discard(r.bout_id)
                self._saved_notes_by_bout.pop(r.bout_id, None)
            self._populate_table(redraw_scatter_on_change=False)
            self._draw_scatter()
        self.status_summary_label.setText(
            f"↺ Cleared {cleared} saved Examined flag(s) for {scope_desc}."
        )

    def _execute_guarded_edit(
        self,
        action: EditAction,
        *,
        target_is_positive: bool | None = None,
        target_behavior: str | None = None,
        trim_bounds: tuple[int, int] | None = None,
        split_frame: int | None = None,
    ) -> None:
        if self._report is None or self._selected_bout_idx is None:
            return
        b_idx = self._selected_bout_idx
        rec = self._report.feature_matrix.records[b_idx]
        self._ensure_origin_snapshot(rec)
        diag = self._report.diagnostics
        src_is_pred = rec.source in ("predicted", "unlabeled")
        behavior = (
            target_behavior
            or (rec.behavior if rec.behavior and rec.behavior != "None" else None)
            or self._report.behavior
        )
        precomputed_counts = self._report.cv_track_counts or None
        pair_opposite = self.pair_opposite_action.isChecked()

        impact = evaluate_cv_group_impact(
            self._project,
            behavior,
            video=rec.video,
            identity=rec.identity,
            action=action,
            start_frame=rec.start_frame,
            end_frame=rec.end_frame,
            current_is_positive=rec.is_positive,
            target_is_positive=target_is_positive,
            trim_bounds=trim_bounds,
            source_is_predicted=src_is_pred,
            precomputed_track_counts=precomputed_counts,
            find_pairing_candidate=(action == "promote_prediction" and pair_opposite),
        )

        if not impact.safe_to_apply and impact.warning_message:
            ok = MessageDialog.confirm(
                self,
                title="CV Group Guard Warning (Dual-Class CV)",
                message=f"{impact.warning_message}\n\nApply this edit anyway?",
            )
            if not ok:
                return

        origin_src = str(getattr(rec, "origin_source", rec.source))
        keep_row_on_delete = action == "delete" and origin_src == "predicted"

        # Determine next bout in the current visible table BEFORE mutating filter membership
        next_b_idx = self._peek_next_bout_after_current(
            b_idx, for_delete=(action == "delete" and not keep_row_on_delete)
        )

        track_key = (rec.video, int(rec.identity))
        prev_track_count = (
            self._report.cv_track_counts.get(track_key)
            if self._report.cv_track_counts is not None
            else None
        )

        def _sync_cb(video_name: str, updated_labels: VideoLabels) -> None:
            if self._central_widget is not None:
                self._central_widget.sync_external_video_labels(video_name, updated_labels)

        eff_target_beh = target_behavior or behavior
        old_start, old_end = rec.start_frame, rec.end_frame
        old_is_pos = rec.is_positive
        live_pose = self._live_pose_for_video(rec.video)

        self._ignoring_self_label_signal = True
        try:
            edit_snap = apply_bout_edit(
                self._project,
                behavior,
                video=rec.video,
                identity=rec.identity,
                action=action,
                start_frame=old_start,
                end_frame=old_end,
                current_is_positive=old_is_pos,
                target_is_positive=target_is_positive,
                target_behavior=eff_target_beh,
                trim_bounds=trim_bounds,
                split_frame=split_frame,
                source_is_predicted=src_is_pred,
                pair_with_opposite=pair_opposite,
                precomputed_track_counts=precomputed_counts,
                precomputed_impact=impact,
                pose_est=live_pose,
                undo_stack=self._undo_stack,
                on_video_labels_saved=_sync_cb,
            )
        finally:
            self._ignoring_self_label_signal = False
        self._video_report_key = None

        # Snapshot in-memory UI state for instant < 20 ms Undo
        paired_overlap_ids: list[int] = []
        paired_pinned_ids: list[int] = []
        changed_b_indices: list[int] = [b_idx]
        if edit_snap.paired_bout is not None:
            pb = edit_snap.paired_bout
            for cand_idx, cand in enumerate(self._report.feature_matrix.records):
                if (
                    cand.source == "predicted"
                    and cand.bout_id not in self._reviewed_bout_ids
                    and cand.video == pb.video
                    and cand.identity == pb.identity
                    and cand.start_frame <= pb.end_frame
                    and cand.end_frame >= pb.start_frame
                ):
                    paired_overlap_ids.append(cand.bout_id)
                    changed_b_indices.append(cand_idx)
                    if cand.bout_id not in self._pinned_needs_review_bout_ids:
                        self._pinned_needs_review_bout_ids.add(cand.bout_id)
                        paired_pinned_ids.append(cand.bout_id)
                    self._reviewed_bout_ids.add(cand.bout_id)
                    self._saved_notes_by_bout[cand.bout_id] = (
                        f"✓ Auto-paired as {pb.class_name} (Dual-Class CV)"
                    )

        prev_examined_entry = self._examined_store.find_match(rec)
        ui_snap: dict[str, Any] = {
            "disk_mutated": True,
            "b_idx": b_idx,
            "rec_before": replace(rec),
            "was_reviewed": rec.bout_id in self._reviewed_bout_ids,
            "was_examined_only": rec.bout_id in self._examined_only_bout_ids,
            "was_pinned": rec.bout_id in self._pinned_needs_review_bout_ids,
            "was_deleted": rec.bout_id in self._deleted_bout_ids,
            "was_unlabeled": rec.bout_id in self._unlabeled_bout_ids,
            "prev_note": self._saved_notes_by_bout.get(rec.bout_id),
            "prev_examined_entry": prev_examined_entry,
            "prev_trim": diag.suggested_trim_bounds[b_idx],
            "prev_split": diag.suggested_split_frame[b_idx],
            "track_key": track_key,
            "prev_track_count": prev_track_count,
            "paired_overlap_ids": paired_overlap_ids,
            "paired_pinned_ids": paired_pinned_ids,
        }
        self._ui_undo_history.append(ui_snap)
        if len(self._ui_undo_history) > 200:
            del self._ui_undo_history[:-200]
        self.undo_btn.setEnabled(True)

        # Clear stale trim/split diagnostic lines on the edited bout (restored on Undo)
        diag.suggested_trim_bounds[b_idx] = None
        diag.suggested_split_frame[b_idx] = None

        # Mutate in-memory BoutRecord immediately (< 1 ms) so we never block on 5-fold CV!
        if action == "delete":
            if keep_row_on_delete:
                rec.source = "predicted"
                rec.behavior = str(getattr(rec, "origin_behavior", rec.behavior))
                rec.class_name = str(getattr(rec, "origin_class_name", rec.class_name))
                rec.is_positive = bool(getattr(rec, "origin_is_positive", True))
                self._unlabeled_bout_ids.add(rec.bout_id)
                self._reviewed_bout_ids.add(rec.bout_id)
                self._pinned_needs_review_bout_ids.add(rec.bout_id)
                note = f"○ Cleared to No Label ({old_start}\u2013{old_end})"
            else:
                self._deleted_bout_ids.add(rec.bout_id)
                self._unlabeled_bout_ids.discard(rec.bout_id)
                note = f"Removed label ({old_start}\u2013{old_end})"
        elif action == "trim" and trim_bounds is not None:
            rec.start_frame, rec.end_frame = int(trim_bounds[0]), int(trim_bounds[1])
            rec.duration_frames = rec.end_frame - rec.start_frame + 1
            self._unlabeled_bout_ids.discard(rec.bout_id)
            self._reviewed_bout_ids.add(rec.bout_id)
            note = f"✓ Trimmed to {rec.start_frame}\u2013{rec.end_frame}"
        elif action == "split" and split_frame is not None:
            gap_s = max(old_start + 1, min(int(split_frame) - 1, old_end - 1))
            gap_e = min(old_end - 1, gap_s + 2)
            if src_is_pred:
                new_pos = True if target_is_positive is None else bool(target_is_positive)
                new_cls = (
                    eff_target_beh
                    if new_pos
                    else (
                        MULTICLASS_NONE_BEHAVIOR
                        if self._report.is_multiclass
                        else f"Not-{behavior}"
                    )
                )
                rec.source = "labeled"
                rec.is_positive = new_pos
                rec.behavior = (
                    eff_target_beh
                    if (new_pos or not self._report.is_multiclass)
                    else MULTICLASS_NONE_BEHAVIOR
                )
                rec.class_name = new_cls
            rec.end_frame = max(old_start, gap_s - 1)
            rec.duration_frames = rec.end_frame - rec.start_frame + 1
            self._unlabeled_bout_ids.discard(rec.bout_id)
            self._reviewed_bout_ids.add(rec.bout_id)
            note = f"✓ Split into {old_start}\u2013{rec.end_frame} & {gap_e + 1}\u2013{old_end}"
        else:
            new_pos = (
                bool(target_is_positive) if target_is_positive is not None else rec.is_positive
            )
            new_cls = (
                eff_target_beh
                if new_pos
                else (
                    MULTICLASS_NONE_BEHAVIOR if self._report.is_multiclass else f"Not-{behavior}"
                )
            )
            rec.source = "labeled"
            rec.is_positive = new_pos
            rec.behavior = (
                eff_target_beh
                if (new_pos or not self._report.is_multiclass)
                else MULTICLASS_NONE_BEHAVIOR
            )
            rec.class_name = new_cls
            self._unlabeled_bout_ids.discard(rec.bout_id)
            self._reviewed_bout_ids.add(rec.bout_id)
            note = f"✓ Saved as {new_cls}"
            if edit_snap.paired_bout is not None:
                pb = edit_snap.paired_bout
                note += f" (+ paired {pb.class_name} {pb.start_frame}\u2013{pb.end_frame})"

        self._examined_only_bout_ids.discard(rec.bout_id)
        if action != "delete" or keep_row_on_delete:
            self._pinned_needs_review_bout_ids.add(rec.bout_id)

        self._saved_notes_by_bout[rec.bout_id] = note
        if action == "delete" and not keep_row_on_delete:
            self._examined_store.unmark(rec)
        else:
            persist_kind = "no_label" if (action == "delete" and keep_row_on_delete) else action
            self._examined_store.mark(rec, action=persist_kind, note=note)
        self._pending_recalc_count += 1
        self._update_recalc_button_badge()

        # O(T) incremental CV group health update (< 0.1 ms, zero JSON disk rescan!)
        pos_b, neg_b = self._report.cv_track_counts.get(track_key, (0, 0))
        pos_a, neg_a = simulate_track_count_delta(
            pos_b,
            neg_b,
            action=action,
            start_frame=old_start,
            end_frame=old_end,
            current_is_positive=old_is_pos,
            target_is_positive=target_is_positive,
            trim_bounds=trim_bounds,
            source_is_predicted=src_is_pred,
        )
        if edit_snap.paired_bout is not None:
            pb_dur = edit_snap.paired_bout.duration_frames
            if edit_snap.paired_bout.is_positive:
                pos_a += pb_dur
            else:
                neg_a += pb_dur
        if pos_a > 0 or neg_a > 0:
            self._report.cv_track_counts[track_key] = (pos_a, neg_a)
        else:
            self._report.cv_track_counts.pop(track_key, None)
        cv_cnt, jabs_met, adr_met = summarize_cv_track_counts(self._report.cv_track_counts)
        self._report.cv_groups_count = cv_cnt
        self._report.jabs_threshold_met = jabs_met
        self._report.dual_class_cv_met = adr_met
        self._update_cv_badge_from_counts(cv_cnt, jabs_met, adr_met)

        removed_from_table = action == "delete" and not keep_row_on_delete
        self._finish_fast_inplace_triage_ui(
            b_idx,
            changed_b_indices,
            next_b_idx=next_b_idx,
            removed_from_table=removed_from_table,
            redraw_inspector_plot=(action in ("trim", "split")),
            select_span_on_seek=(action == "trim"),
            force_seek_same_track=(action == "trim"),
        )

        anim_lbl = self._animal_display_text(rec.video, rec.identity)
        self.status_summary_label.setText(
            f"{note} on {_short_video_name(rec.video)} ({anim_lbl}) — Undo [⌘Z]"
        )

    def _peek_next_bout_after_current(
        self, curr_b_idx: int, *, for_delete: bool = False
    ) -> int | None:
        row_count = self.table.rowCount()
        if row_count == 0 or self._report is None:
            return None
        curr_row = -1
        b_list: list[int] = []
        for r in range(row_count):
            it = self.table.item(r, 0)
            if it is not None:
                bi = int(it.data(Qt.ItemDataRole.UserRole))
                b_list.append(bi)
                if bi == curr_b_idx:
                    curr_row = len(b_list) - 1
        if not b_list:
            return None

        # In "Reviewed ✓" without a custom cohort, advance down the list and stop at the final row
        if (
            self._queue_filter == "reviewed"
            and not self._lasso_selected_indices
            and not for_delete
        ):
            if curr_row >= 0 and curr_row + 1 < len(b_list):
                return b_list[curr_row + 1]
            return None

        records = self._report.feature_matrix.records
        ordered_candidates = (
            b_list[curr_row + 1 :] + b_list[:curr_row] if curr_row >= 0 else list(b_list)
        )
        for bi in ordered_candidates:
            if (
                bi != curr_b_idx
                and records[bi].bout_id not in self._reviewed_bout_ids
                and records[bi].bout_id not in self._deleted_bout_ids
            ):
                return bi

        if for_delete and curr_row >= 0:
            if curr_row + 1 < len(b_list):
                return b_list[curr_row + 1]
            if curr_row - 1 >= 0:
                return b_list[curr_row - 1]
        return None

    # ------------------------------------------------------------------
    # Button & Keyboard Action Handlers
    # ------------------------------------------------------------------
    def _on_label_behavior_z_clicked(self) -> None:
        """Key `[Z]`: Always assign Behavior / active target positive class (matches main JABS GUI)."""
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._mode in ("subtypes", "videos")
        ):
            return
        rec = self._report.feature_matrix.records[self._selected_bout_idx]
        target_cls = str(self.target_pos_combo.currentData() or self._report.behavior)
        if rec.source in ("predicted", "unlabeled"):
            self._execute_guarded_edit(
                "promote_prediction",
                target_is_positive=True,
                target_behavior=target_cls
                if self._report.is_multiclass
                else self._report.behavior,
            )
            return
        if self._report.is_multiclass:
            if rec.class_name == target_cls:
                self._mark_current_bout_kept()
            else:
                self._execute_guarded_edit(
                    "flip",
                    target_is_positive=True,
                    target_behavior=target_cls,
                )
        else:
            if rec.is_positive:
                self._mark_current_bout_kept()
            else:
                self._execute_guarded_edit(
                    "flip",
                    target_is_positive=True,
                    target_behavior=self._report.behavior,
                )

    def _on_label_not_behavior_c_clicked(self) -> None:
        """Key `[C]`: Always assign Not-Behavior / None (matches main JABS GUI)."""
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._mode in ("subtypes", "videos")
        ):
            return
        rec = self._report.feature_matrix.records[self._selected_bout_idx]
        if rec.source in ("predicted", "unlabeled"):
            pred_beh = self._bout_display_behavior(rec)
            self._execute_guarded_edit(
                "promote_prediction",
                target_is_positive=False,
                target_behavior=pred_beh if self._report.is_multiclass else self._report.behavior,
            )
            return
        if self._report.is_multiclass:
            if rec.class_name == MULTICLASS_NONE_BEHAVIOR:
                self._mark_current_bout_kept()
            else:
                self._execute_guarded_edit(
                    "flip",
                    target_is_positive=False,
                    target_behavior=rec.class_name,
                )
        else:
            if not rec.is_positive:
                self._mark_current_bout_kept()
            else:
                self._execute_guarded_edit(
                    "flip",
                    target_is_positive=False,
                    target_behavior=self._report.behavior,
                )

    def _on_no_label_x_clicked(self) -> None:
        """Key `[X]`: Clear human label back to unlabeled, or leave predicted bout unlabeled."""
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._mode in ("subtypes", "videos")
        ):
            return
        rec = self._report.feature_matrix.records[self._selected_bout_idx]
        if rec.source in ("predicted", "unlabeled"):
            self._mark_current_bout_unlabeled()
            return
        self._execute_guarded_edit("delete")

    def _on_primary_action_1_clicked(self) -> None:
        """Key `[1]`: Keep verified label in Audit mode (or apply chosen MC class); Confirm Positive in Predict mode."""
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._mode in ("subtypes", "videos")
        ):
            return
        rec = self._report.feature_matrix.records[self._selected_bout_idx]
        target_cls = str(self.target_pos_combo.currentData() or self._report.behavior)
        if rec.source in ("predicted", "unlabeled"):
            self._execute_guarded_edit(
                "promote_prediction",
                target_is_positive=True,
                target_behavior=target_cls
                if self._report.is_multiclass
                else self._report.behavior,
            )
            return

        # Labeled bout in Audit mode
        if self._report.is_multiclass:
            if rec.class_name != MULTICLASS_NONE_BEHAVIOR and target_cls != rec.class_name:
                self._execute_guarded_edit(
                    "flip",
                    target_is_positive=True,
                    target_behavior=target_cls,
                )
            else:
                self._mark_current_bout_kept()
        else:
            self._mark_current_bout_kept()

    def _on_secondary_action_2_clicked(self) -> None:
        """Key `[2]`: Switch to Opposite / None in Audit mode; Mark Not-Behavior in Predict mode."""
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._mode in ("subtypes", "videos")
        ):
            return
        rec = self._report.feature_matrix.records[self._selected_bout_idx]
        target_cls = str(self.target_pos_combo.currentData() or self._report.behavior)
        if rec.source in ("predicted", "unlabeled"):
            pred_beh = self._bout_display_behavior(rec)
            self._execute_guarded_edit(
                "promote_prediction",
                target_is_positive=False,
                target_behavior=pred_beh if self._report.is_multiclass else self._report.behavior,
            )
            return

        if self._report.is_multiclass:
            if rec.class_name == MULTICLASS_NONE_BEHAVIOR:
                self._execute_guarded_edit(
                    "flip",
                    target_is_positive=True,
                    target_behavior=target_cls,
                )
            else:
                self._execute_guarded_edit(
                    "flip",
                    target_is_positive=False,
                    target_behavior=rec.class_name,
                )
        else:
            self._execute_guarded_edit(
                "flip",
                target_is_positive=not rec.is_positive,
                target_behavior=self._report.behavior,
            )

    def _on_trim_clicked(self) -> None:
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._mode in ("subtypes", "videos")
        ):
            return
        trim = self._report.diagnostics.suggested_trim_bounds[self._selected_bout_idx]
        if trim is not None:
            self._execute_guarded_edit("trim", trim_bounds=trim)

    def _on_split_clicked(self) -> None:
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._mode in ("subtypes", "videos")
        ):
            return
        rec = self._report.feature_matrix.records[self._selected_bout_idx]
        sf = self._report.diagnostics.suggested_split_frame[self._selected_bout_idx]
        if sf is None:
            sf = (rec.start_frame + rec.end_frame) // 2
        target_cls = str(self.target_pos_combo.currentData() or self._report.behavior)
        self._execute_guarded_edit(
            "split",
            split_frame=sf,
            target_is_positive=rec.is_positive,
            target_behavior=target_cls if self._report.is_multiclass else self._report.behavior,
        )

    def _on_delete_clicked(self) -> None:
        if (
            self._report is None
            or self._selected_bout_idx is None
            or self._mode in ("subtypes", "videos")
        ):
            return
        rec = self._report.feature_matrix.records[self._selected_bout_idx]
        if rec.source == "predicted":
            return
        self._execute_guarded_edit("delete")

    def _on_undo_clicked(self) -> None:
        if not self._ui_undo_history:
            return
        ui_snap = self._ui_undo_history.pop()
        disk_mutated = bool(ui_snap.get("disk_mutated", True))

        if disk_mutated:

            def _sync_cb(video_name: str, updated_labels: VideoLabels) -> None:
                if self._central_widget is not None:
                    self._central_widget.sync_external_video_labels(video_name, updated_labels)

            self._ignoring_self_label_signal = True
            try:
                snap = self._undo_stack.undo(self._project, on_video_labels_saved=_sync_cb)
            finally:
                self._ignoring_self_label_signal = False
            if snap is None:
                self.undo_btn.setEnabled(bool(self._ui_undo_history))
                return
            self._video_report_key = None

        self.undo_btn.setEnabled(bool(self._ui_undo_history))

        if self._report is not None:
            b_idx = int(ui_snap["b_idx"])
            rec_before: BoutRecord = ui_snap["rec_before"]
            self._report.feature_matrix.records[b_idx] = rec_before
            self._report.diagnostics.suggested_trim_bounds[b_idx] = ui_snap.get("prev_trim")
            self._report.diagnostics.suggested_split_frame[b_idx] = ui_snap.get("prev_split")

            if ui_snap["was_reviewed"]:
                self._reviewed_bout_ids.add(rec_before.bout_id)
            else:
                self._reviewed_bout_ids.discard(rec_before.bout_id)
            if ui_snap.get("was_examined_only", False):
                self._examined_only_bout_ids.add(rec_before.bout_id)
            else:
                self._examined_only_bout_ids.discard(rec_before.bout_id)
            if ui_snap.get("was_pinned", False):
                self._pinned_needs_review_bout_ids.add(rec_before.bout_id)
            else:
                self._pinned_needs_review_bout_ids.discard(rec_before.bout_id)
            if ui_snap["was_deleted"]:
                self._deleted_bout_ids.add(rec_before.bout_id)
            else:
                self._deleted_bout_ids.discard(rec_before.bout_id)
            if ui_snap.get("was_unlabeled", False):
                self._unlabeled_bout_ids.add(rec_before.bout_id)
            else:
                self._unlabeled_bout_ids.discard(rec_before.bout_id)
            if ui_snap["prev_note"] is not None:
                self._saved_notes_by_bout[rec_before.bout_id] = ui_snap["prev_note"]
            else:
                self._saved_notes_by_bout.pop(rec_before.bout_id, None)

            # Restore persisted Examined ledger state cleanly on Undo
            self._examined_store.unmark(rec_before, persist=False)
            prev_ex_ent = ui_snap.get("prev_examined_entry")
            if isinstance(prev_ex_ent, dict):
                self._examined_store.restore_entry(prev_ex_ent, persist=False)
            self._examined_store.save()

            for p_bid in ui_snap.get("paired_overlap_ids", []):
                self._reviewed_bout_ids.discard(p_bid)
                self._saved_notes_by_bout.pop(p_bid, None)
            for p_bid in ui_snap.get("paired_pinned_ids", []):
                self._pinned_needs_review_bout_ids.discard(p_bid)

            if disk_mutated:
                self._pending_recalc_count = max(0, self._pending_recalc_count - 1)
                self._update_recalc_button_badge()

                t_key = ui_snap.get("track_key")
                prev_tc = ui_snap.get("prev_track_count")
                if t_key is not None:
                    if prev_tc is not None:
                        self._report.cv_track_counts[t_key] = prev_tc
                    else:
                        self._report.cv_track_counts.pop(t_key, None)
                cv_cnt, jabs_met, adr_met = summarize_cv_track_counts(self._report.cv_track_counts)
                self._report.cv_groups_count = cv_cnt
                self._report.jabs_threshold_met = jabs_met
                self._report.dual_class_cv_met = adr_met
                self._update_cv_badge_from_counts(cv_cnt, jabs_met, adr_met)

            self._selected_bout_idx = b_idx
            self._last_card_bout_idx = None
            filter_switched = False
            if not ui_snap["was_reviewed"] and self._queue_filter == "reviewed":
                self._set_queue_filter("needs_review", redraw=False)
                filter_switched = True

            paired_ids = set(ui_snap.get("paired_overlap_ids", []))
            cur_rows: list[int] = []
            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it is not None:
                    cur_rows.append(int(it.data(Qt.ItemDataRole.UserRole)))
            new_filtered = self._filtered_bout_indices() if not filter_switched else []
            can_inplace = (
                not filter_switched
                and b_idx in new_filtered
                and len(new_filtered) == len(cur_rows)
                and (
                    (new_filtered == cur_rows)
                    if self._table_sort_col is None
                    else (set(new_filtered) == set(cur_rows))
                )
            )
            if can_inplace:
                changed_undo = [b_idx]
                if paired_ids:
                    for c_i, c_rec in enumerate(self._report.feature_matrix.records):
                        if c_rec.bout_id in paired_ids and c_i != b_idx:
                            changed_undo.append(c_i)
                self._update_queue_filter_counts()
                self._refresh_class_filter_combos()
                self._update_table_rows_in_place(changed_undo)
                row_for_b = self._find_table_row_for_bout(b_idx)
                if row_for_b is not None:
                    prev_sup = self._suppress_auto_seek
                    self._suppress_auto_seek = True
                    try:
                        self.table.selectRow(row_for_b)
                    finally:
                        self._suppress_auto_seek = prev_sup
                self._update_inspector_card()
                self._update_bout_scatter_selection_ring()
                self._schedule_scatter_redraw()
            else:
                self._populate_table(redraw_scatter_on_change=False)
                self._draw_scatter()
            if self.auto_seek_action.isChecked() and not self._is_main_window_on_bout_track(
                rec_before
            ):
                self._seek_bout_in_main_window(rec_before, mark_viewed=False)
            anim_lbl = self._animal_display_text(rec_before.video, rec_before.identity)
            self.status_summary_label.setText(
                f"Undid step on {_short_video_name(rec_before.video)} ({anim_lbl})"
            )
        else:
            self.refresh_analysis()

    def _step_table_row(self, delta: int) -> None:
        self._clear_table_mouse_press_state()
        self._clear_video_table_mouse_press_state()
        active_tbl = self.video_table if self._mode == "videos" else self.table
        n = active_tbl.rowCount()
        if n <= 0:
            return
        curr = active_tbl.currentRow()
        nxt = max(0, min(n - 1, (curr + delta) if curr >= 0 else 0))
        active_tbl.selectRow(nxt)
        self._pending_click_already_sought_row = None

    def _clear_video_table_mouse_press_state(self) -> None:
        if self._is_closing:
            return
        self._video_table_mouse_press_btn = None

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Intercept table shortcut keys and capture clicked table column/button before selection changes."""
        tbl = getattr(self, "table", None)
        vtbl = getattr(self, "video_table", None)
        if tbl is not None:
            try:
                vp = tbl.viewport()
            except RuntimeError:
                return super().eventFilter(source, event)
            if source is vp:
                if event.type() == QtCore.QEvent.Type.MouseButtonPress and isinstance(
                    event, QtGui.QMouseEvent
                ):
                    self._table_mouse_press_col = tbl.columnAt(int(event.position().x()))
                    self._table_mouse_press_btn = event.button()
                elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                    QtCore.QTimer.singleShot(0, self, self._clear_table_mouse_press_state)
                    if self._pending_bring_main_window_to_front:
                        QtCore.QTimer.singleShot(0, self, self._flush_bring_main_window_to_front)
                elif event.type() in (
                    QtCore.QEvent.Type.Leave,
                    QtCore.QEvent.Type.FocusOut,
                ) and not (
                    QtWidgets.QApplication.mouseButtons()
                    & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
                ):
                    self._clear_table_mouse_press_state()
                return False
        if vtbl is not None:
            try:
                vvp = vtbl.viewport()
            except RuntimeError:
                return super().eventFilter(source, event)
            if source is vvp:
                if event.type() == QtCore.QEvent.Type.MouseButtonPress and isinstance(
                    event, QtGui.QMouseEvent
                ):
                    self._video_table_mouse_press_btn = event.button()
                elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                    QtCore.QTimer.singleShot(0, self, self._clear_video_table_mouse_press_state)
                    if self._pending_bring_main_window_to_front:
                        QtCore.QTimer.singleShot(0, self, self._flush_bring_main_window_to_front)
                elif event.type() in (
                    QtCore.QEvent.Type.Leave,
                    QtCore.QEvent.Type.FocusOut,
                ) and not (
                    QtWidgets.QApplication.mouseButtons()
                    & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
                ):
                    self._clear_video_table_mouse_press_state()
                return False
        if (
            (source is tbl or (vtbl is not None and source is vtbl))
            and event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and self._dispatch_shortcut_key(event)
        ):
            return True
        return super().eventFilter(source, event)

    def _dispatch_shortcut_key(self, event: QtGui.QKeyEvent) -> bool:
        mods = event.modifiers()
        key = event.key()

        # Cmd+Z / Ctrl+Z Undo applies only to bout editing modes (disabled in 4. Videos)
        if key == Qt.Key.Key_Z and (
            mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        ):
            if self._mode != "videos" and self.undo_btn.isEnabled():
                self._on_undo_clicked()
            event.accept()
            return True

        # Do not intercept typing inside either search filter box
        active_search = (
            self.video_search_input
            if (
                self.video_search_input.hasFocus() or self.focusWidget() is self.video_search_input
            )
            else (
                self.search_input
                if (self.search_input.hasFocus() or self.focusWidget() is self.search_input)
                else None
            )
        )
        if active_search is not None:
            if key == Qt.Key.Key_Escape:
                active_search.clearFocus()
                event.accept()
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if active_search is self.video_search_input:
                    self._flush_video_search_if_pending()
                elif active_search is self.search_input:
                    self._flush_bout_search_if_pending()
                active_search.clearFocus()
                self._sync_selected_bout_to_main_window_if_enabled()
                event.accept()
                return True
            return False

        if mods == Qt.KeyboardModifier.NoModifier:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._sync_selected_bout_to_main_window_if_enabled()
                event.accept()
                return True
            if key == Qt.Key.Key_Escape:
                if self._mode == "videos" and self._video_lasso_selected:
                    self._on_clear_video_map_selection()
                    event.accept()
                    return True
                if self._lasso_selected_indices:
                    self._on_clear_box_filter()
                    event.accept()
                    return True
                if self._map_expanded:
                    self._toggle_expand_map(False)
                    event.accept()
                    return True
                if self._scatter_view_limits is not None:
                    self._reset_scatter_zoom()
                    event.accept()
                    return True
                if self._class_filter != "all":
                    self._set_class_filter("all")
                    event.accept()
                    return True
                return False
            if key == Qt.Key.Key_Space:
                self._on_play_clip_clicked()
                event.accept()
                return True
            if key in (Qt.Key.Key_Down, Qt.Key.Key_J):
                self._step_table_row(1)
                event.accept()
                return True
            if key in (Qt.Key.Key_Up, Qt.Key.Key_K):
                self._step_table_row(-1)
                event.accept()
                return True
            if key == Qt.Key.Key_Z:
                if self._mode != "videos":
                    self._on_label_behavior_z_clicked()
                event.accept()
                return True
            if key == Qt.Key.Key_X:
                if self._mode != "videos":
                    self._on_no_label_x_clicked()
                event.accept()
                return True
            if key == Qt.Key.Key_C:
                if self._mode != "videos":
                    self._on_label_not_behavior_c_clicked()
                event.accept()
                return True
            if key == Qt.Key.Key_V:
                if self._mode != "videos":
                    self._on_toggle_examined_v_clicked()
                event.accept()
                return True
            if key == Qt.Key.Key_1:
                if self._mode != "videos":
                    self._on_primary_action_1_clicked()
                event.accept()
                return True
            if key == Qt.Key.Key_2:
                if self._mode != "videos":
                    self._on_secondary_action_2_clicked()
                event.accept()
                return True
            if key == Qt.Key.Key_T:
                if self._mode != "videos" and not self.trim_btn.isHidden():
                    self._on_trim_clicked()
                event.accept()
                return True
            if key == Qt.Key.Key_N:
                if self._mode != "videos":
                    self._select_nearest_similar_bouts(15)
                event.accept()
                return True
            if key == Qt.Key.Key_M:
                self._toggle_expand_map()
                event.accept()
                return True
            if key == Qt.Key.Key_A:
                if self._mode != "videos":
                    self._toggle_arena_map()
                event.accept()
                return True
            if key == Qt.Key.Key_0:
                self._reset_scatter_zoom()
                event.accept()
                return True
            if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self._zoom_scatter_by_step(0.76)
                event.accept()
                return True
            if key == Qt.Key.Key_Minus:
                self._zoom_scatter_by_step(1.28)
                event.accept()
                return True
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                if self._mode != "videos" and not self.delete_btn.isHidden():
                    self._on_no_label_x_clicked()
                event.accept()
                return True
            if key == Qt.Key.Key_R:
                if self._mode == "videos":
                    self.refresh_video_summary(force=True)
                else:
                    self.refresh_analysis()
                event.accept()
                return True
        elif mods == Qt.KeyboardModifier.ShiftModifier and key == Qt.Key.Key_Plus:
            self._zoom_scatter_by_step(0.76)
            event.accept()
            return True
        return False

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Route dialog-level keyboard triage and map shortcuts."""
        if self._dispatch_shortcut_key(event):
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Search Bar Queue & Export
    # ------------------------------------------------------------------
    def _on_send_queue_to_search_bar(self) -> None:
        if self._mode == "videos":
            self._on_send_videos_to_search_bar()
            return
        if self._report is None or self._central_widget is None:
            return
        records = self._report.feature_matrix.records
        # Preserve the exact displayed table sort order when sending to JABS Search Bar
        if self.table.rowCount() > 0:
            indices: list[int] = []
            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it is not None:
                    indices.append(int(it.data(Qt.ItemDataRole.UserRole)))
        else:
            indices = self._filtered_bout_indices()
        if (
            self._mode == "predict"
            and self._queue_filter == "needs_review"
            and not self._lasso_selected_indices
        ):
            indices = [i for i in indices if records[i].bout_id not in self._reviewed_bout_ids]
        hits = [
            SearchHit(
                file=records[i].video,
                identity=str(records[i].identity),
                behavior=self._bout_display_behavior(records[i]),
                start_frame=records[i].start_frame,
                end_frame=records[i].end_frame,
            )
            for i in indices
        ]
        if not hits:
            MessageDialog.warning(self, message="No bouts in the current queue to send.")
            return
        scope_lbl = self._active_scope_label()
        desc = f"BALE: {scope_lbl} ({len(hits)} bouts)"
        self._central_widget.load_custom_search_queue(hits, description=desc)
        self.status_summary_label.setText(
            f"Sent {len(hits)} {scope_lbl} bouts to JABS Search Bar (< Prev | Next >)"
        )

    def _on_export_clicked(self) -> None:
        if self._mode == "videos":
            self._on_export_videos_csv_clicked()
            return
        if self._report is None:
            return
        default_dir = Path(self._report.project_dir) / "jabs" / "training_logs"
        default_dir.mkdir(parents=True, exist_ok=True)
        scope_lbl = self._active_scope_label()
        stem_lbl = "All_Classes" if scope_lbl == "All Classes" else scope_lbl
        stem = f"bale_{stem_lbl}"
        csv_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export BALE Report (CSV + PNG + Editable SVG)",
            str(default_dir / f"{stem}.csv"),
            "CSV Files (*.csv)",
        )
        if not csv_path:
            return
        out_csv = Path(csv_path)
        out_png = out_csv.with_suffix(".png")
        out_svg = out_csv.with_suffix(".svg")

        df = self._report.to_dataframe(exclude_bout_ids=self._deleted_bout_ids)
        df.to_csv(out_csv, index=False)
        export_bale_summary_figure(
            self._report,
            out_png,
            out_svg,
            projection=self._proj_method,
        )
        MessageDialog.information(
            self,
            title=f"BALE Exported ({scope_lbl})",
            message=f"Saved {scope_lbl} CSV, 240 DPI PNG, and editable SVG:\n- {out_csv}\n- {out_png}\n- {out_svg}",
        )

    # ------------------------------------------------------------------
    # Mode 4: Whole-Video Screener (Frames, Bouts, Zero-Behavior & 50k Criteria)
    # ------------------------------------------------------------------
    def _switch_to_videos_mode(self) -> None:
        self.mode_videos_btn.setChecked(True)
        self._on_mode_pill_clicked(3)

    def _active_video_scope_key(self) -> tuple[str, bool]:
        sm = self._project.settings_manager
        behaviors = list(sm.behavior_names)
        beh_data = self.behavior_combo.currentData()
        include_all_mc = beh_data == "__ALL_MC__"
        active_beh = (
            behaviors[0]
            if include_all_mc
            else str(beh_data or (behaviors[0] if behaviors else ""))
        )
        return active_beh, include_all_mc

    def refresh_video_summary(self, *, force: bool = False) -> None:
        """Scan all project videos off the GUI thread using only compact `.h5` & `.json` files."""
        if self._is_closing:
            return
        sm = self._project.settings_manager
        if not list(sm.behavior_names):
            self.status_summary_label.setText("No behaviors configured in this JABS project.")
            return

        active_beh, include_all_mc = self._active_video_scope_key()
        key = (active_beh, include_all_mc)
        if not force and self._video_report is not None and self._video_report_key == key:
            self._video_report.thresholds = classify_video_outliers(
                self._video_report.records, source=self._video_source
            )
            self._update_video_filter_counts()
            self._populate_video_table(redraw_scatter_on_change=False)
            self._draw_video_scatter()
            self._update_video_inspector_card()
            self._update_recalc_button_badge()
            return

        if self._video_worker is not None:
            if hasattr(self._video_worker, "cancel_event"):
                self._video_worker.cancel_event.set()
            self._queued_video_refresh = True
            self.status_summary_label.setText(
                "Queued video scan — updating as soon as current scan finishes…"
            )
            return

        self._queued_video_refresh = False
        self._video_request_id += 1
        req_id = self._video_request_id
        beh_label = "All Classes" if include_all_mc else active_beh
        self.videos_summary_lbl.setText(f"Scanning all videos for {beh_label}…")
        self.status_summary_label.setText(f"Scanning whole-video stats for {beh_label}…")

        worker = _VideoScreenerTask(
            self._project,
            None if include_all_mc else active_beh,
            include_all_multiclass=include_all_mc,
            use_postprocessed=True,
            source=self._video_source,
            key=key,
            req_id=req_id,
        )
        self._video_worker = worker
        self._update_recalc_button_badge()
        worker.signals.finished.connect(self._on_video_summary_finished)
        worker.signals.cancelled.connect(self._on_video_summary_cancelled)
        worker.signals.error.connect(self._on_video_summary_error)
        self._thread_pool.start(worker)

    def _on_video_summary_cancelled(self, req_id: int) -> None:
        """Clear a cancelled video screener worker and immediately run any queued scan."""
        if self._is_closing:
            return
        if (
            self._video_worker is not None
            and getattr(self._video_worker, "req_id", None) == req_id
        ):
            self._video_worker = None
        queued = self._queued_video_refresh
        self._queued_video_refresh = False
        self._update_recalc_button_badge()
        if queued and self._mode == "videos":
            self.refresh_video_summary(force=True)

    def _on_video_summary_finished(
        self,
        report: VideoScreenerReport,
        key: tuple[str, bool],
        req_id: int | None = None,
    ) -> None:
        if self._is_closing:
            return
        self._video_worker = None
        queued = self._queued_video_refresh
        self._queued_video_refresh = False
        current_key = self._active_video_scope_key()
        if (req_id is not None and req_id != self._video_request_id) or key != current_key:
            self._update_recalc_button_badge()
            if self._mode == "videos" and (queued or key != current_key):
                self.refresh_video_summary(force=True)
            return
        if queued:
            self._update_recalc_button_badge()
            if self._mode == "videos":
                self.refresh_video_summary(force=True)
            return

        self._video_report = report
        self._video_report_key = key

        # Auto-select Human Labels if no predictions exist yet so the biologist gets immediate value
        if (
            not report.has_any_predictions
            and report.has_any_labels
            and self._video_source != "labels"
        ):
            self._video_source = "labels"
            idx = self.videos_source_combo.findData("labels")
            if idx >= 0:
                self.videos_source_combo.blockSignals(True)
                self.videos_source_combo.setCurrentIndex(idx)
                self.videos_source_combo.blockSignals(False)

        report.thresholds = classify_video_outliers(report.records, source=self._video_source)
        self._video_lasso_selected.clear()
        self.video_clear_sel_btn.setVisible(False)
        self._scatter_view_limits = None
        self._scatter_default_limits = None
        self._sync_zoom_reset_button()

        # Keep CV Tracks badge synced when user opens Mode 4 directly or switches behavior in Mode 4
        if self._report is None or self._analysis_needed_on_bout_mode:
            cv_cnt = sum(int(r.label_dual_class_tracks) for r in report.records)
            self._update_cv_badge_from_counts(cv_cnt, cv_cnt >= 2, cv_cnt >= 15)

        self._update_video_filter_counts()
        self._update_recalc_button_badge()
        if self._mode == "videos":
            self._populate_video_table(redraw_scatter_on_change=False)
            self._draw_video_scatter()
            self._update_video_inspector_card()
            n_v = len(report.records)
            n_zero = sum(
                1
                for r in report.records
                if r.active_frames(self._video_source) == 0
                and r.active_bouts(self._video_source) == 0
            )
            scope_lbl = "All Classes" if report.include_all_multiclass else report.behavior
            self.status_summary_label.setText(
                f"Screened {_plural(n_v, 'video')} for {scope_lbl} in {report.elapsed_ms:.0f} ms "
                f"({n_v - n_zero} with behavior, {n_zero} zero)"
            )

    def _on_video_summary_error(self, err_msg: str, req_id: int | None = None) -> None:
        if self._is_closing:
            return
        self._video_worker = None
        queued = self._queued_video_refresh
        self._queued_video_refresh = False
        if queued and self._mode == "videos":
            self.refresh_video_summary(force=True)
            return
        if req_id is not None and req_id != self._video_request_id:
            self._update_recalc_button_badge()
            return
        self._update_recalc_button_badge()
        self.videos_summary_lbl.setText("Video scan failed.")
        self.status_summary_label.setText(f"Video screener failed: {err_msg}")
        MessageDialog.error(self, title="Video Screener Error", message=err_msg)

    def _get_video_record(self, video_name: str | None) -> VideoSummaryRecord | None:
        if self._video_report is None or not video_name:
            return None
        for r in self._video_report.records:
            if r.video == video_name:
                return r
        return None

    def _default_sort_for_video_preset(
        self, preset: VideoScreeningPreset
    ) -> tuple[int, Qt.SortOrder]:
        if preset == "high_bouts":
            return 3, Qt.SortOrder.DescendingOrder
        if preset == "long_bout":
            return 4, Qt.SortOrder.DescendingOrder
        if preset == "low_conf_pos":
            return 5, Qt.SortOrder.DescendingOrder
        if preset in ("low_conf_none", "zero_all"):
            return 6, Qt.SortOrder.DescendingOrder
        if preset == "zero_high_conf":
            return 7, Qt.SortOrder.DescendingOrder
        if preset in ("any_outlier", "fragile"):
            return 1, Qt.SortOrder.DescendingOrder
        return 2, Qt.SortOrder.DescendingOrder

    def _on_video_source_changed(self) -> None:
        raw = str(self.videos_source_combo.currentData() or "predictions")
        if raw in ("predictions", "labels", "predictions_unlabeled"):
            self._video_source = raw  # type: ignore[assignment]
        if self._video_report is None:
            return
        self._video_report.thresholds = classify_video_outliers(
            self._video_report.records, source=self._video_source
        )
        self._scatter_view_limits = None
        self._scatter_default_limits = None
        self._sync_zoom_reset_button()
        self._update_video_filter_counts()
        self._populate_video_table(redraw_scatter_on_change=False)
        self._draw_video_scatter()
        self._update_video_inspector_card()

    def _on_video_preset_changed(self) -> None:
        raw = str(self.videos_preset_combo.currentData() or "all")
        self._video_preset = raw  # type: ignore[assignment]
        self._video_lasso_selected.clear()
        self.video_clear_sel_btn.setVisible(False)
        self._sync_video_quick_pills_from_preset()
        col, order = self._default_sort_for_video_preset(self._video_preset)
        self._video_table_sort_col = col
        self._video_table_sort_order = order
        self.video_table.horizontalHeader().setSortIndicator(col, order)
        self._populate_video_table(redraw_scatter_on_change=False)
        self._draw_video_scatter()

    def _on_video_unlabeled_toggled(self, checked: bool) -> None:
        self._video_unlabeled_only = bool(checked)
        self._update_video_filter_counts()
        self._populate_video_table(redraw_scatter_on_change=False)
        self._draw_video_scatter()

    def _on_video_quick_pill_clicked(self, pill_id: int) -> None:
        preset_map: dict[int, VideoScreeningPreset] = {
            0: "all",
            1: "high_frames",
            2: "high_bouts",
            3: "zero_all",
            4: "any_outlier",
        }
        target_preset = preset_map.get(pill_id, "all")
        self._video_preset = target_preset
        self._video_lasso_selected.clear()
        self.video_clear_sel_btn.setVisible(False)
        idx = self.videos_preset_combo.findData(target_preset)
        if idx >= 0:
            self.videos_preset_combo.blockSignals(True)
            self.videos_preset_combo.setCurrentIndex(idx)
            self.videos_preset_combo.blockSignals(False)
        col, order = self._default_sort_for_video_preset(target_preset)
        self._video_table_sort_col = col
        self._video_table_sort_order = order
        self.video_table.horizontalHeader().setSortIndicator(col, order)
        self._populate_video_table(redraw_scatter_on_change=False)
        self._draw_video_scatter()

    def _sync_video_quick_pills_from_preset(self) -> None:
        btn_by_preset = {
            "all": self.vf_all_btn,
            "high_frames": self.vf_high_frames_btn,
            "high_bouts": self.vf_high_bouts_btn,
            "zero_all": self.vf_zero_btn,
            "any_outlier": self.vf_outliers_btn,
        }
        target_btn = btn_by_preset.get(self._video_preset)
        self.vfilter_group.setExclusive(False)
        for b in (
            self.vf_all_btn,
            self.vf_high_frames_btn,
            self.vf_high_bouts_btn,
            self.vf_zero_btn,
            self.vf_outliers_btn,
        ):
            b.setChecked(b is target_btn)
        self.vfilter_group.setExclusive(True)

    def _update_video_filter_counts(self) -> None:
        if self._video_report is None:
            self.vf_all_btn.setText("All Videos (0)")
            self.vf_high_frames_btn.setText("Most Frames (0)")
            self.vf_high_bouts_btn.setText("Most Bouts (0)")
            self.vf_zero_btn.setText("Zero Beh (0)")
            self.vf_outliers_btn.setText("50k Outliers (0)")
            self.videos_summary_lbl.setText("")
            return

        recs = self._video_report.records
        unlbl = self._video_unlabeled_only
        src = self._video_source
        n_all = len(filter_video_records(recs, preset="all", source=src, unlabeled_only=unlbl))
        n_hf = len(
            filter_video_records(recs, preset="high_frames", source=src, unlabeled_only=unlbl)
        )
        n_hb = len(
            filter_video_records(recs, preset="high_bouts", source=src, unlabeled_only=unlbl)
        )
        n_zero = len(
            filter_video_records(recs, preset="zero_all", source=src, unlabeled_only=unlbl)
        )
        n_out = len(
            filter_video_records(recs, preset="any_outlier", source=src, unlabeled_only=unlbl)
        )
        n_e3a = len(
            filter_video_records(recs, preset="low_conf_pos", source=src, unlabeled_only=unlbl)
        )
        n_e3b = len(
            filter_video_records(recs, preset="low_conf_none", source=src, unlabeled_only=unlbl)
        )

        self.vf_all_btn.setText(f"All Videos ({n_all})")
        self.vf_high_frames_btn.setText(f"Most Frames ({n_hf})")
        self.vf_high_bouts_btn.setText(f"Most Bouts ({n_hb})")
        self.vf_zero_btn.setText(f"Zero Beh ({n_zero})")
        self.vf_outliers_btn.setText(f"50k Outliers ({n_out})")

        self.videos_summary_lbl.setText(
            f"{_plural(n_all, 'video')} • {n_out} outliers • "
            f"E3a {n_e3a} • E3b {n_e3b} • Zero {n_zero}"
        )

    def _on_video_search_text_changed(self, _text: str = "") -> None:
        self._video_search_timer.start()

    def _on_video_search_applied(self) -> None:
        if self._is_closing or self._video_report is None:
            return
        self._populate_video_table(redraw_scatter_on_change=False)
        self._draw_video_scatter()

    def _flush_video_search_if_pending(self) -> None:
        if self._video_search_timer.isActive():
            self._video_search_timer.stop()
            self._on_video_search_applied()

    def _on_clear_video_map_selection(self) -> None:
        self._video_lasso_selected.clear()
        self.video_clear_sel_btn.setVisible(False)
        self._populate_video_table(redraw_scatter_on_change=False)
        self._draw_video_scatter()

    def _filtered_video_records(self) -> list[VideoSummaryRecord]:
        if self._video_report is None:
            return []
        pool = filter_video_records(
            self._video_report.records,
            preset=self._video_preset,
            source=self._video_source,
            unlabeled_only=self._video_unlabeled_only,
            text_query=self.video_search_input.text(),
        )
        if self._video_lasso_selected:
            pool = [r for r in pool if r.video in self._video_lasso_selected]
        return pool

    def _find_video_table_row(self, video_name: str) -> int | None:
        for r in range(self.video_table.rowCount()):
            it = self.video_table.item(r, 0)
            if it is not None and str(it.data(Qt.ItemDataRole.UserRole) or "") == video_name:
                return r
        return None

    def _fit_video_table_columns(self) -> None:
        """One-shot column sizing protecting Col 0 (`Video`) from collapsing on compact windows."""
        max_widths = {1: 168, 2: 122, 3: 62, 4: 112, 5: 94, 6: 104, 7: 90, 8: 112}
        other_total = 0
        for c_i in range(1, 9):
            self.video_table.resizeColumnToContents(c_i)
            w = min(max(self.video_table.columnWidth(c_i) + 14, 54), max_widths[c_i])
            self.video_table.setColumnWidth(c_i, w)
            other_total += w
        vp_w = self.video_table.viewport().width()
        col0_w = max(185, (vp_w - other_total - 4) if vp_w > 240 else 195)
        self.video_table.setColumnWidth(0, col0_w)

    def _populate_video_table(
        self,
        *_: Any,
        redraw_scatter_on_change: bool = True,
    ) -> None:
        if self._video_report is None:
            self.video_table.setRowCount(0)
            return

        src = self._video_source
        recs = self._filtered_video_records()
        prev_video = self._selected_video_name

        self.video_table.blockSignals(True)
        self.video_table.setUpdatesEnabled(False)
        try:
            self.video_table.setRowCount(0)
            self.video_table.setRowCount(len(recs))

            for row_idx, rec in enumerate(recs):
                short_vid = _short_video_name(rec.video)
                # Col 0: Video
                it_vid = _SortableTableItem(
                    short_vid, sort_key=(short_vid.lower(), rec.video.lower())
                )
                it_vid.setData(Qt.ItemDataRole.UserRole, rec.video)
                it_vid.setToolTip(f"{rec.video} ({_plural(rec.n_animals, 'animal track')})")
                self.video_table.setItem(row_idx, 0, it_vid)

                # Col 1: Screening Bucket (subtle tint reserved exclusively for this status column)
                bg_hex, fg_hex, _ = _VIDEO_BUCKET_STYLE.get(
                    rec.primary_bucket, ("#F8FAFC", "#475569", "#94A3B8")
                )
                it_bkt = _SortableTableItem(
                    rec.primary_bucket,
                    sort_key=(
                        float(rec.outlier_score),
                        rec.active_frames(src),
                        rec.active_bouts(src),
                    ),
                )
                it_bkt.setBackground(QtGui.QColor(bg_hex))
                it_bkt.setForeground(QtGui.QColor(fg_hex))
                bfont = it_bkt.font()
                bfont.setBold(rec.primary_bucket != "Normal")
                it_bkt.setFont(bfont)
                it_bkt.setToolTip(
                    "50k tags: " + (", ".join(rec.outlier_tags) if rec.outlier_tags else "Normal")
                )
                self.video_table.setItem(row_idx, 1, it_bkt)

                # Col 2: Frames (%)
                f_cnt = rec.active_frames(src)
                pct = rec.active_pct(src)
                sec_tot = f_cnt / 30.0
                f_txt = f"{f_cnt:,}f ({pct:.1f}%)" if f_cnt > 0 else "0f (0.0%)"
                it_frm = _SortableTableItem(f_txt, sort_key=(int(f_cnt), float(pct)))
                it_frm.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                it_frm.setToolTip(
                    f"{f_cnt:,} frames ({sec_tot:.1f} s) out of {rec.valid_frames:,} valid frames"
                )
                if f_cnt == 0:
                    it_frm.setForeground(QtGui.QColor("#64748B"))
                self.video_table.setItem(row_idx, 2, it_frm)

                # Col 3: Bouts
                b_cnt = rec.active_bouts(src)
                it_bouts = _SortableTableItem(str(b_cnt), sort_key=(int(b_cnt), int(f_cnt)))
                it_bouts.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                it_bouts.setToolTip(
                    f"{_plural(b_cnt, 'bout')} across {_plural(rec.n_animals, 'animal track')} "
                    f"({rec.pred_bouts_per_min:.1f} bouts/min)"
                )
                if b_cnt == 0:
                    it_bouts.setForeground(QtGui.QColor("#64748B"))
                self.video_table.setItem(row_idx, 3, it_bouts)

                # Col 4: Max Bout
                max_b = rec.active_max_bout(src)
                mb_txt = f"{max_b}f ({max_b / 30.0:.1f}s)" if max_b > 0 else "—"
                it_max = _SortableTableItem(mb_txt, sort_key=int(max_b))
                it_max.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.video_table.setItem(row_idx, 4, it_max)

                # Col 5: Low-Conf Behavior (E3a) — clean typography, no noisy cell background
                e3a_b = rec.pred_low_conf_pos_bouts
                if e3a_b > 0:
                    min_p_s = (
                        f" ({rec.pred_min_bout_prob:.2f})"
                        if math.isfinite(rec.pred_min_bout_prob)
                        else ""
                    )
                    e3a_txt = f"{e3a_b}{min_p_s}"
                    e3a_sort = (
                        e3a_b,
                        rec.pred_low_conf_pos_frames,
                        -(
                            rec.pred_min_bout_prob
                            if math.isfinite(rec.pred_min_bout_prob)
                            else 1.0
                        ),
                    )
                elif "E3a_low_conf_pos" in rec.outlier_tags and math.isfinite(
                    rec.pred_min_bout_prob
                ):
                    e3a_txt = f"min {rec.pred_min_bout_prob:.2f}"
                    e3a_sort = (0, 1, -rec.pred_min_bout_prob)
                else:
                    e3a_txt = "—"
                    e3a_sort = (0, 0, -1.0)
                it_e3a = _SortableTableItem(e3a_txt, sort_key=e3a_sort)
                it_e3a.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if e3a_b > 0 or "E3a_low_conf_pos" in rec.outlier_tags:
                    it_e3a.setForeground(QtGui.QColor("#991B1B"))
                    efont = it_e3a.font()
                    efont.setBold(True)
                    it_e3a.setFont(efont)
                    it_e3a.setToolTip(
                        f"E3a: {_plural(e3a_b, 'low-confidence behavior bout')} "
                        f"(min bout confidence {rec.pred_min_bout_prob:.2f})"
                    )
                else:
                    it_e3a.setForeground(QtGui.QColor("#94A3B8"))
                self.video_table.setItem(row_idx, 5, it_e3a)

                # Col 6: Near-Miss Not-Behavior (E3b) — `count (seconds)` avoids `s` unit collision
                e3b_s = rec.pred_low_conf_none_spans
                if e3b_s > 0:
                    e3b_sec = rec.pred_low_conf_none_frames / 30.0
                    e3b_txt = f"{e3b_s} ({e3b_sec:.1f}s)"
                    e3b_sort = (e3b_s, rec.pred_low_conf_none_frames)
                else:
                    e3b_txt = "—"
                    e3b_sort = (0, 0)
                it_e3b = _SortableTableItem(e3b_txt, sort_key=e3b_sort)
                it_e3b.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if e3b_s > 0:
                    it_e3b.setForeground(QtGui.QColor("#92400E"))
                    nfont = it_e3b.font()
                    nfont.setBold(True)
                    it_e3b.setFont(nfont)
                    it_e3b.setToolTip(
                        f"E3b: {_plural(e3b_s, 'standalone near-miss Not-Behavior span')} "
                        f"({rec.pred_low_conf_none_frames} frames = {rec.pred_low_conf_none_frames / 30.0:.1f} s)"
                    )
                else:
                    it_e3b.setForeground(QtGui.QColor("#94A3B8"))
                self.video_table.setItem(row_idx, 6, it_e3b)

                # Col 7: Conf (behavior prob if behavior > 0 else Not-Behavior confidence)
                if rec.has_predictions:
                    if rec.pred_pos_frames > 0 and math.isfinite(rec.pred_mean_prob):
                        conf_val = float(rec.pred_mean_prob)
                        conf_txt = f"{conf_val * 100:.0f}%"
                    elif math.isfinite(rec.pred_mean_none_prob):
                        conf_val = float(rec.pred_mean_none_prob)
                        conf_txt = f"{conf_val * 100:.0f}% None"
                    else:
                        conf_val = -1.0
                        conf_txt = "—"
                else:
                    conf_val = -1.0
                    conf_txt = "No Preds"
                it_conf = _SortableTableItem(conf_txt, sort_key=conf_val)
                it_conf.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.video_table.setItem(row_idx, 7, it_conf)

                # Col 8: Human Labels Summary — clean text, no pastel background mosaic
                if rec.has_labels:
                    lbl_tot = rec.label_pos_frames + rec.label_neg_frames
                    lbl_txt = f"+{rec.label_pos_bouts}/-{rec.label_neg_bouts}"
                    if rec.label_dual_class_tracks > 0:
                        lbl_txt += f" ✓{rec.label_dual_class_tracks}"
                    it_lbl = _SortableTableItem(
                        lbl_txt, sort_key=(2, rec.label_dual_class_tracks, lbl_tot)
                    )
                    it_lbl.setForeground(QtGui.QColor("#166534"))
                    lfont = it_lbl.font()
                    lfont.setBold(True)
                    it_lbl.setFont(lfont)
                    it_lbl.setToolTip(
                        f"Human labels: {rec.label_pos_frames}f Pos ({_plural(rec.label_pos_bouts, 'bout')}), "
                        f"{rec.label_neg_frames}f Neg ({_plural(rec.label_neg_bouts, 'bout')}), "
                        f"{_plural(rec.label_dual_class_tracks, 'dual-class CV track')}"
                    )
                else:
                    it_lbl = _SortableTableItem("Unlabeled", sort_key=(0, 0, 0))
                    it_lbl.setForeground(QtGui.QColor("#64748B"))
                it_lbl.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.video_table.setItem(row_idx, 8, it_lbl)

            if len(recs) > 1:
                vhdr = self.video_table.horizontalHeader()
                vhdr.setSortIndicatorShown(True)
                vhdr.setSortIndicator(self._video_table_sort_col, self._video_table_sort_order)
                self.video_table.sortItems(
                    self._video_table_sort_col, self._video_table_sort_order
                )

            self._fit_video_table_columns()

            reselect_row = 0
            if prev_video is not None and len(recs) > 0:
                found_r = self._find_video_table_row(prev_video)
                if found_r is not None:
                    reselect_row = found_r
        finally:
            self.video_table.setUpdatesEnabled(True)
            self.video_table.blockSignals(False)

        if len(recs) > 0:
            self._suppress_auto_seek = True
            try:
                self.video_table.selectRow(reselect_row)
            finally:
                self._suppress_auto_seek = False
            sel_it = self.video_table.item(reselect_row, 0)
            self._selected_video_name = (
                str(sel_it.data(Qt.ItemDataRole.UserRole) or "")
                if sel_it is not None
                else recs[0].video
            )
            self._pending_video_click_already_sought_row = None
            self._video_scrubbed_frame = None
            self._update_video_inspector_card()
            if redraw_scatter_on_change and self._selected_video_name != prev_video:
                self._update_video_scatter_selection_ring()
        else:
            self._selected_video_name = None
            self._pending_video_click_already_sought_row = None
            self._video_scrubbed_frame = None
            self._update_video_inspector_card()
            if redraw_scatter_on_change and prev_video is not None:
                self._update_video_scatter_selection_ring()

    def _on_video_table_header_clicked(self, col: int) -> None:
        if self._video_report is None or self.video_table.rowCount() == 0:
            return
        if self._video_table_sort_col == col:
            order = (
                Qt.SortOrder.AscendingOrder
                if self._video_table_sort_order == Qt.SortOrder.DescendingOrder
                else Qt.SortOrder.DescendingOrder
            )
        else:
            order = Qt.SortOrder.AscendingOrder if col == 0 else Qt.SortOrder.DescendingOrder
        self._video_table_sort_col = col
        self._video_table_sort_order = order

        vhdr = self.video_table.horizontalHeader()
        vhdr.setSortIndicatorShown(True)
        vhdr.setSortIndicator(col, order)

        prev_vid = self._selected_video_name
        self.video_table.blockSignals(True)
        self.video_table.setUpdatesEnabled(False)
        self._suppress_auto_seek = True
        try:
            self.video_table.sortItems(col, order)
            if prev_vid is not None:
                new_r = self._find_video_table_row(prev_vid)
                if new_r is not None:
                    self.video_table.selectRow(new_r)
        finally:
            self._suppress_auto_seek = False
            self.video_table.setUpdatesEnabled(True)
            self.video_table.blockSignals(False)

    def _update_video_scatter_selection_ring(self) -> None:
        """Move the selected-video highlight ring in O(1) without rebuilding the full 2D scatter."""
        if (
            self._video_scatter_sel_artist is None
            or self._video_scatter_coords is None
            or not self._video_scatter_names
        ):
            self._draw_video_scatter()
            return
        vname = self._selected_video_name
        if vname is not None and vname in self._video_scatter_names:
            s_i = self._video_scatter_names.index(vname)
            self._video_scatter_sel_artist.set_offsets(
                np.array(
                    [
                        [
                            float(self._video_scatter_coords[s_i, 0]),
                            float(self._video_scatter_coords[s_i, 1]),
                        ]
                    ],
                    dtype=np.float64,
                )
            )
            self._video_scatter_sel_artist.set_visible(True)
        else:
            self._video_scatter_sel_artist.set_offsets(np.empty((0, 2), dtype=np.float64))
            self._video_scatter_sel_artist.set_visible(False)
        self.scatter_canvas.draw_idle()

    def _on_video_table_selection_changed(self) -> None:
        if self._video_report is None:
            return
        sel = self.video_table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        it = self.video_table.item(row, 0)
        if it is None:
            return
        vname = str(it.data(Qt.ItemDataRole.UserRole) or "")
        if not vname:
            return
        is_right_click = self._video_table_mouse_press_btn == Qt.MouseButton.RightButton
        self._selected_video_name = vname
        self._video_scrubbed_frame = None
        self._update_video_scatter_selection_ring()
        self._update_video_inspector_card()
        if (
            self.auto_seek_action.isChecked()
            and not self._suppress_auto_seek
            and not is_right_click
        ):
            rec = self._get_video_record(vname)
            if rec is not None:
                from_left_click = self._video_table_mouse_press_btn == Qt.MouseButton.LeftButton
                self._pending_video_click_already_sought_row = row
                self._seek_video_in_main_window(
                    rec, frame_zero=False, bring_to_front=from_left_click
                )
        else:
            self._pending_video_click_already_sought_row = None

    def _on_video_table_cell_clicked(self, row: int, _col: int) -> None:
        """Seek JABS even when left-clicking an already-selected row (e.g. row 0 right after load)."""
        btn = self._video_table_mouse_press_btn
        self._clear_video_table_mouse_press_state()
        if self._video_report is None or btn == Qt.MouseButton.RightButton:
            return
        if self._pending_video_click_already_sought_row == row:
            self._pending_video_click_already_sought_row = None
            if self.auto_seek_action.isChecked():
                self._request_bring_main_window_to_front()
            return
        self._pending_video_click_already_sought_row = None
        it = self.video_table.item(row, 0)
        if it is None:
            return
        vname = str(it.data(Qt.ItemDataRole.UserRole) or "")
        if not vname:
            return
        self._selected_video_name = vname
        self._video_scrubbed_frame = None
        self._update_video_scatter_selection_ring()
        self._update_video_inspector_card()
        if self.auto_seek_action.isChecked():
            rec = self._get_video_record(vname)
            if rec is not None:
                self._seek_video_in_main_window(rec, frame_zero=False, bring_to_front=True)

    def _select_video_by_name(self, video_name: str, *, seek_in_jabs: bool = True) -> None:
        self._clear_video_table_mouse_press_state()
        if not video_name or self._video_report is None:
            return
        self._selected_video_name = video_name
        self._video_scrubbed_frame = None
        seek_epoch_before = self._video_seek_epoch
        row = self._find_video_table_row(video_name)
        if row is not None:
            prev_sup = self._suppress_auto_seek
            if not seek_in_jabs:
                self._suppress_auto_seek = True
            try:
                selected_rows = {it.row() for it in self.video_table.selectedItems()}
                if selected_rows == {row}:
                    if self.video_table.currentRow() != row:
                        self.video_table.blockSignals(True)
                        try:
                            self.video_table.setCurrentCell(row, 0)
                        finally:
                            self.video_table.blockSignals(False)
                    self._on_video_table_selection_changed()
                else:
                    self.video_table.selectRow(row)
            finally:
                self._suppress_auto_seek = prev_sup
            if seek_in_jabs and self.auto_seek_action.isChecked() and not self._suppress_auto_seek:
                if self._video_seek_epoch == seek_epoch_before:
                    rec = self._get_video_record(video_name)
                    if rec is not None:
                        self._seek_video_in_main_window(rec, frame_zero=False, bring_to_front=True)
                else:
                    self._request_bring_main_window_to_front()
            return
        self._update_video_scatter_selection_ring()
        self._update_video_inspector_card()
        if seek_in_jabs and self.auto_seek_action.isChecked():
            rec = self._get_video_record(video_name)
            if rec is not None:
                self._seek_video_in_main_window(rec, frame_zero=False, bring_to_front=True)

    def _relax_video_preset_if_disjoint(self, picked: set[str]) -> bool:
        """If user boxes dimmed points outside the active preset, relax preset to `all` so rows appear."""
        if self._video_report is None or self._video_preset == "all" or not picked:
            return False
        preset_names = {
            r.video
            for r in filter_video_records(
                self._video_report.records,
                preset=self._video_preset,
                source=self._video_source,
                unlabeled_only=self._video_unlabeled_only,
                text_query=self.video_search_input.text(),
            )
        }
        if picked & preset_names:
            return False
        self._video_preset = "all"
        idx = self.videos_preset_combo.findData("all")
        if idx >= 0:
            self.videos_preset_combo.blockSignals(True)
            self.videos_preset_combo.setCurrentIndex(idx)
            self.videos_preset_combo.blockSignals(False)
        self._sync_video_quick_pills_from_preset()
        return True

    def _on_video_scatter_box_selected(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        *,
        additive: bool = False,
    ) -> None:
        if self._video_scatter_coords is None or not self._video_scatter_names:
            return
        coords = self._video_scatter_coords
        picked = {
            self._video_scatter_names[i]
            for i in range(len(self._video_scatter_names))
            if x0 <= float(coords[i, 0]) <= x1 and y0 <= float(coords[i, 1]) <= y1
        }
        if additive:
            picked |= self._video_lasso_selected
        if not picked:
            self.status_summary_label.setText("0 videos inside map selection box.")
            return
        relaxed = self._relax_video_preset_if_disjoint(picked)
        self._video_lasso_selected = picked
        self.video_clear_sel_btn.setText(f"✕ Map Selection ({len(picked)})")
        self.video_clear_sel_btn.setVisible(True)
        self._populate_video_table(redraw_scatter_on_change=False)
        self._draw_video_scatter()
        note = " (switched filter to All Videos)" if relaxed else ""
        self.status_summary_label.setText(
            f"Selected {_plural(len(picked), 'video')} from 2D map{note} — press Esc to clear"
        )

    def _on_video_scatter_lasso_selected(self, poly_xy: Any, *, additive: bool = False) -> None:
        if self._video_scatter_coords is None or not self._video_scatter_names or poly_xy is None:
            return
        poly_arr = np.asarray(poly_xy, dtype=np.float64)
        if poly_arr.ndim != 2 or len(poly_arr) < 3:
            return
        inside = MplPath(poly_arr).contains_points(self._video_scatter_coords)
        picked = {self._video_scatter_names[i] for i in np.flatnonzero(inside)}
        if additive:
            picked |= self._video_lasso_selected
        if not picked:
            self.status_summary_label.setText("0 videos inside lasso selection.")
            return
        relaxed = self._relax_video_preset_if_disjoint(picked)
        self._video_lasso_selected = picked
        self.video_clear_sel_btn.setText(f"✕ Lasso ({len(picked)})")
        self.video_clear_sel_btn.setVisible(True)
        self._populate_video_table(redraw_scatter_on_change=False)
        self._draw_video_scatter()
        note = " (switched filter to All Videos)" if relaxed else ""
        self.status_summary_label.setText(
            f"Selected {_plural(len(picked), 'video')} via lasso{note} — press Esc to clear"
        )

    def _draw_video_scatter(self) -> None:
        ax = self.scatter_ax
        ax.clear()
        self._video_scatter_sel_artist = None

        if self._video_report is None:
            ax.text(
                0.5,
                0.5,
                "Scanning all project videos…",
                ha="center",
                va="center",
                fontsize=11.0,
                color="#475569",
                transform=ax.transAxes,
            )
            self._video_scatter_coords = None
            self._video_scatter_names = []
            self.scatter_canvas.set_pick_points(None)
            self.scatter_canvas.mark_layout_dirty()
            self.scatter_canvas.draw_idle()
            return

        src = self._video_source
        # Plot all scope-eligible videos (unlabeled filter respected) so preset/map selection stands out in context
        base_recs = filter_video_records(
            self._video_report.records,
            preset="all",
            source=src,
            unlabeled_only=self._video_unlabeled_only,
        )
        if not base_recs:
            ax.text(
                0.5,
                0.5,
                "No videos match the active filter.",
                ha="center",
                va="center",
                fontsize=11.0,
                color="#475569",
                transform=ax.transAxes,
            )
            self._video_scatter_coords = None
            self._video_scatter_names = []
            self.scatter_canvas.set_pick_points(None)
            self.scatter_canvas.mark_layout_dirty()
            self.scatter_canvas.draw_idle()
            return

        active_set = {r.video for r in self._filtered_video_records()}
        coords = np.array(
            [[float(r.active_bouts(src)), float(r.active_pct(src))] for r in base_recs],
            dtype=np.float64,
        )
        names = [r.video for r in base_recs]
        self._video_scatter_coords = coords
        self._video_scatter_names = names

        max_x = float(np.max(coords[:, 0])) if len(coords) else 1.0
        max_y = float(np.max(coords[:, 1])) if len(coords) else 1.0
        xlim_lo = -max(0.6, 0.05 * max(max_x, 4.0))
        xlim_hi = max(3.0, max_x * 1.12)
        ylim_lo = -max(0.5, 0.06 * max(max_y, 5.0))
        ylim_hi = max(4.0, max_y * 1.14)

        # Cohort E1 P90 reference lines with subtle direct "P90" labels
        thr = self._video_report.thresholds or {}
        b_thr = float(thr.get("bouts_high_thr", 0.0) or 0.0)
        p_thr = float(thr.get("pct_high_thr", 0.0) or 0.0)
        if float(thr.get("n_active_videos", 0.0) or 0.0) >= 3:
            if b_thr > 0:
                ax.axvline(b_thr, color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=1)
                ax.text(
                    b_thr,
                    ylim_hi * 0.97,
                    " P90",
                    ha="left",
                    va="top",
                    fontsize=7.8,
                    fontweight="bold",
                    color="#64748B",
                    zorder=2,
                )
            if p_thr > 0:
                ax.axhline(p_thr, color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=1)
                ax.text(
                    xlim_hi * 0.99,
                    p_thr,
                    "P90 ",
                    ha="right",
                    va="bottom",
                    fontsize=7.8,
                    fontweight="bold",
                    color="#64748B",
                    zorder=2,
                )

        # Collapse 7 screening tags into 4 restrained journal-grade semantic groups
        def _semantic_group(rec: VideoSummaryRecord) -> str:
            pb = rec.primary_bucket
            if pb.startswith(("E1", "E2")):
                return "Extreme"
            if pb.startswith(("E3", "E4")):
                return "Doubt/Fragile"
            if pb.startswith("E5"):
                return "Zero Beh"
            return "Normal"

        group_specs = (
            ("Normal", "#94A3B8", "o", 36, 0.54, 2),
            ("Zero Beh", "#0F766E", "X", 64, 0.92, 7),
            ("Extreme", "#1E40AF", "o", 64, 0.90, 5),
            ("Doubt/Fragile", "#B91C1C", "D", 62, 0.90, 6),
        )
        n_leg_items = 0
        for grp_name, col_hex, mk, sz, alp, zord in group_specs:
            in_grp = [i for i, r in enumerate(base_recs) if _semantic_group(r) == grp_name]
            if not in_grp:
                continue
            in_act = [i for i in in_grp if base_recs[i].video in active_set]
            in_dim = [i for i in in_grp if base_recs[i].video not in active_set]
            if in_dim:
                ax.scatter(
                    coords[in_dim, 0],
                    coords[in_dim, 1],
                    c=col_hex,
                    marker=mk,
                    s=max(22, sz - 16),
                    alpha=0.15,
                    edgecolors="none",
                    zorder=1,
                )
            if in_act:
                ax.scatter(
                    coords[in_act, 0],
                    coords[in_act, 1],
                    c=col_hex,
                    marker=mk,
                    s=sz,
                    alpha=alp,
                    edgecolors="#FFFFFF",
                    linewidths=0.45,
                    zorder=zord,
                    label=f"{grp_name} ({len(in_act)})",
                )
                n_leg_items += 1

        # Highlight map box/lasso selection
        if self._video_lasso_selected:
            sel_loc = [i for i, v in enumerate(names) if v in self._video_lasso_selected]
            if sel_loc:
                ax.scatter(
                    coords[sel_loc, 0],
                    coords[sel_loc, 1],
                    facecolors="none",
                    edgecolors="#1D4ED8",
                    s=140,
                    linewidths=1.8,
                    zorder=8,
                    label=f"Selected ({len(sel_loc)})",
                )
                n_leg_items += 1

        # Persistent selected-video ring artist for instant O(1) offset updates on row stepping
        if self._selected_video_name is not None and self._selected_video_name in names:
            s_i = names.index(self._selected_video_name)
            sel_xy = np.array([[float(coords[s_i, 0]), float(coords[s_i, 1])]], dtype=np.float64)
            sel_vis = True
        else:
            sel_xy = np.empty((0, 2), dtype=np.float64)
            sel_vis = False
        self._video_scatter_sel_artist = ax.scatter(
            sel_xy[:, 0] if len(sel_xy) else [],
            sel_xy[:, 1] if len(sel_xy) else [],
            facecolors="none",
            edgecolors="#0F172A",
            s=195,
            linewidths=2.4,
            zorder=10,
        )
        self._video_scatter_sel_artist.set_visible(sel_vis)

        ax.set_xlim(xlim_lo, xlim_hi)
        ax.set_ylim(ylim_lo, ylim_hi)

        def_xlim = tuple(float(v) for v in ax.get_xlim())
        def_ylim = tuple(float(v) for v in ax.get_ylim())
        self._scatter_default_limits = (def_xlim, def_ylim)  # type: ignore[assignment]
        if self._scatter_view_limits is not None:
            ax.set_xlim(self._scatter_view_limits[0])
            ax.set_ylim(self._scatter_view_limits[1])
        self._sync_zoom_reset_button()

        zoom_tag = " [Zoom]" if self._scatter_view_limits else ""
        scope_lbl = (
            "All Classes"
            if self._video_report.include_all_multiclass
            else self._video_report.behavior
        )
        fig_w_in = float(self.scatter_canvas.figure.get_figwidth())
        leg_ncol = (
            min(2, n_leg_items) if (n_leg_items >= 3 and fig_w_in < 5.6) else min(4, n_leg_items)
        )
        leg_rows = (n_leg_items + leg_ncol - 1) // max(1, leg_ncol) if n_leg_items > 0 else 0
        ax.set_title(
            f"Video Map • {scope_lbl} ({len(active_set)}/{len(base_recs)}){zoom_tag}",
            fontsize=10.2,
            fontweight="bold",
            color="#0F172A",
            pad=(34.0 if leg_rows >= 2 else 22.0) if n_leg_items > 0 else 8.0,
        )
        ax.set_xlabel("Behavior Bouts / Video", fontsize=9.5, fontweight="bold", color="#1E293B")
        ax.set_ylabel("Behavior Time (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#E2E8F0", linewidth=0.8)
        if n_leg_items > 0:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=leg_ncol,
                fontsize=8.0,
                frameon=False,
                handletextpad=0.25,
                columnspacing=0.80,
                borderaxespad=0.0,
                markerscale=0.85,
            )

        hover_tips = [
            f"{_short_video_name(r.video)} • {r.primary_bucket}\n"
            f"{r.active_frames(src):,}f ({r.active_pct(src):.1f}%) • {_plural(r.active_bouts(src), 'bout')} • "
            f"Max {r.active_max_bout(src)}f\n"
            f"Low-Conf: {r.pred_low_conf_pos_bouts} • Near-Miss: {r.pred_low_conf_none_spans} "
            f"({r.pred_low_conf_none_frames / 30.0:.1f}s) • "
            f"{'Labeled' if r.has_labels else 'Unlabeled'}"
            for r in base_recs
        ]
        self.scatter_canvas.set_pick_points(
            coords, np.arange(len(base_recs), dtype=np.int32), hover_tips
        )
        self.scatter_canvas.mark_layout_dirty()
        self.scatter_canvas.draw_idle()

    def _update_video_inspector_card(self) -> None:
        rec = self._get_video_record(self._selected_video_name)
        if rec is None or self._video_report is None:
            self.card_title_lbl.setText("No videos match current filter")
            self.card_meta_lbl.setText("Choose 'All Videos' or clear the search filter below")
            self.diagnosis_lbl.setText(
                "Zero videos match the active filter. Switch to 'All Videos' "
                "or uncheck 'Unlabeled Only' to inspect the full project."
            )
            self.diagnosis_frame.setStyleSheet(
                "QFrame { background-color: #F8FAFC; border-left: 4px solid #64748B; border-radius: 3px; }"
            )
            self.play_clip_btn.setEnabled(False)
            self.video_open_btn.setEnabled(False)
            self.video_open_start_btn.setEnabled(False)
            self.video_inspect_preds_btn.setEnabled(False)
            self.video_inspect_labels_btn.setEnabled(False)
            self.insp_ax.clear()
            self.insp_ax.set_axis_off()
            self.insp_canvas.mark_layout_dirty()
            self.insp_canvas.draw_idle()
            return

        src = self._video_source
        f_cnt = rec.active_frames(src)
        pct = rec.active_pct(src)
        b_cnt = rec.active_bouts(src)
        max_b = rec.active_max_bout(src)
        focal = rec.focal_event_for(self._video_preset, src)

        self.card_title_lbl.setText(f"{_short_video_name(rec.video)}  •  {rec.primary_bucket}")
        self.card_title_lbl.setToolTip(rec.video)
        self.card_meta_lbl.setText(
            f"{f_cnt:,}f ({pct:.1f}% • {f_cnt / 30.0:.1f}s)  •  {_plural(b_cnt, 'bout')}  •  "
            f"Max {max_b}f ({max_b / 30.0:.1f}s)  •  {_plural(rec.n_animals, 'animal')}"
        )
        self.card_meta_lbl.setToolTip(rec.video)

        # Build crisp 2-line plain-English summary (Line 1: signals; Line 2: labels & focal jump)
        line1_parts: list[str] = []
        if f_cnt == 0 and b_cnt == 0:
            if rec.pred_low_conf_none_spans > 0:
                line1_parts.append(
                    f"Zero behavior • {_plural(rec.pred_low_conf_none_spans, 'near-miss span')} "
                    f"({rec.pred_low_conf_none_frames / 30.0:.1f}s) may hide missed bouts"
                )
            elif rec.has_predictions and math.isfinite(rec.pred_mean_none_prob):
                line1_parts.append(
                    f"Zero behavior • {rec.pred_mean_none_prob * 100:.1f}% Not-Behavior conf "
                    f"({rec.valid_frames:,} valid frames)"
                )
            else:
                line1_parts.append("Zero behavior frames and 0 bouts recorded")
        else:
            if rec.has_predictions and math.isfinite(rec.pred_mean_prob):
                line1_parts.append(
                    f"Pred conf {rec.pred_mean_prob * 100:.0f}% ({rec.pred_bouts_per_min:.1f} bouts/min)"
                )
            if rec.pred_low_conf_pos_bouts > 0:
                line1_parts.append(
                    f"E3a: {rec.pred_low_conf_pos_bouts} low-conf (min {rec.pred_min_bout_prob:.2f})"
                )
            if rec.pred_low_conf_none_spans > 0:
                line1_parts.append(
                    f"E3b: {rec.pred_low_conf_none_spans} near-miss ({rec.pred_low_conf_none_frames / 30.0:.1f}s)"
                )
            if rec.pred_postproc_fragility >= 0.35:
                line1_parts.append(f"E4: {rec.pred_postproc_fragility * 100:.0f}% raw/post shift")
            if math.isfinite(rec.pred_chiral_asymmetry) and rec.pred_chiral_asymmetry >= 0.70:
                line1_parts.append(f"E1 chiral {rec.pred_chiral_asymmetry * 100:.0f}%")

        if rec.has_labels:
            cv_part = (
                f" ({_plural(rec.label_dual_class_tracks, 'CV track')})"
                if rec.label_dual_class_tracks > 0
                else ""
            )
            lbl_summary = f"Labels: +{rec.label_pos_bouts}/-{rec.label_neg_bouts}{cv_part}"
        else:
            lbl_summary = "Unlabeled"
        focal_summary = (
            f"Focal: {self._animal_display_text(rec.video, focal.identity)} "
            f"@ f{focal.start_frame} ({focal.reason})"
        )
        line1 = " • ".join(line1_parts) if line1_parts else "Whole-video summary ready"
        line2 = f"{lbl_summary}  •  {focal_summary}"

        bg_hex, _fg_hex, accent_hex = _VIDEO_BUCKET_STYLE.get(
            rec.primary_bucket, ("#F1F5F9", "#1E293B", "#1E40AF")
        )
        self.diagnosis_frame.setStyleSheet(
            f"QFrame {{ background-color: {bg_hex}; border-left: 4px solid {accent_hex}; border-radius: 3px; }}"
        )
        self.diagnosis_lbl.setText(f"{line1}\n{line2}")

        has_cw = self._central_widget is not None
        self.play_clip_btn.setEnabled(has_cw)
        self.video_open_btn.setEnabled(has_cw)
        self.video_open_start_btn.setEnabled(has_cw)
        self.video_inspect_preds_btn.setEnabled(rec.pred_bout_count > 0)
        self.video_inspect_labels_btn.setEnabled(rec.has_labels)

        self._draw_video_timeline_plot(rec)

    def _draw_video_timeline_plot(self, rec: VideoSummaryRecord) -> None:
        ax = self.insp_ax
        ax.clear()
        ax.set_aspect("auto", adjustable="box")
        ax.set_axis_on()

        idents = sorted(rec.track_spans.keys())
        if not idents:
            idents = list(range(max(1, rec.n_animals)))
        max_f = max(1, rec.max_track_frames or 1800)

        seen_lbls: set[str] = set()

        def _leg(name: str) -> str | None:
            if name in seen_lbls:
                return None
            seen_lbls.add(name)
            return name

        is_mc_all = bool(
            self._video_report is not None and self._video_report.include_all_multiclass
        )
        cmap = self._behavior_color_hex_map() if is_mc_all else {}

        for row_i, ident in enumerate(idents):
            # Subtle neutral track baseline
            ax.plot(
                [0, max_f],
                [row_i, row_i],
                color="#E2E8F0",
                linewidth=4.0,
                solid_capstyle="butt",
                zorder=1,
            )
            sp = rec.track_spans.get(ident, {})
            cls_map = rec.pred_by_class.get(ident, {}) if is_mc_all else {}

            # Predicted positive bouts (Set1 per-class colors in multi-class All Classes mode!)
            if is_mc_all and cls_map:
                for cls_name, raw_spans in cls_map.items():
                    c_segs = [(s, max(2, e - s + 1)) for s, e in raw_spans]
                    if not c_segs:
                        continue
                    ax.broken_barh(
                        c_segs,
                        (row_i - 0.24, 0.48),
                        facecolors=cmap.get(cls_name, "#1E40AF"),
                        edgecolors="none",
                        alpha=0.88,
                        zorder=3,
                        label=_leg(cls_name),
                    )
            else:
                pred_segs = [(s, max(2, e - s + 1)) for s, e in sp.get("pred_pos", [])]
                if pred_segs:
                    ax.broken_barh(
                        pred_segs,
                        (row_i - 0.24, 0.48),
                        facecolors="#1E40AF",
                        edgecolors="none",
                        alpha=0.88,
                        zorder=3,
                        label=_leg("Predicted"),
                    )

            # E3a low-confidence positive bouts overlay
            e3a_segs = [(s, max(4, e - s + 1)) for s, e in sp.get("low_conf_pos", [])]
            if e3a_segs:
                ax.broken_barh(
                    e3a_segs,
                    (row_i - 0.24, 0.48),
                    facecolors="#B91C1C",
                    edgecolors="none",
                    alpha=0.95,
                    zorder=4,
                    label=_leg("E3a Low-Conf"),
                )

            # E3b standalone near-miss Not-Behavior spans anchored cleanly on track band
            e3b_segs = [(s, max(5, e - s + 1)) for s, e in sp.get("low_conf_none", [])]
            if e3b_segs:
                ax.broken_barh(
                    e3b_segs,
                    (row_i - 0.22, 0.20),
                    facecolors="#D97706",
                    edgecolors="none",
                    alpha=0.92,
                    zorder=4,
                    label=_leg("E3b Near-Miss"),
                )

            # Human label strips right along the upper edge of each animal lane
            lpos_segs = [(s, max(3, e - s + 1)) for s, e in sp.get("label_pos", [])]
            if lpos_segs:
                ax.broken_barh(
                    lpos_segs,
                    (row_i + 0.27, 0.12),
                    facecolors="#15803D",
                    edgecolors="none",
                    alpha=0.95,
                    zorder=5,
                    label=_leg("Pos Label"),
                )
            lneg_segs = [(s, max(3, e - s + 1)) for s, e in sp.get("label_neg", [])]
            if lneg_segs:
                ax.broken_barh(
                    lneg_segs,
                    (row_i + 0.27, 0.12),
                    facecolors="#475569",
                    edgecolors="none",
                    alpha=0.85,
                    zorder=5,
                    label=_leg("Neg Label"),
                )

        focal = rec.focal_event_for(self._video_preset, self._video_source)
        if focal.start_frame > 0 or focal.end_frame > 0:
            ax.axvline(
                focal.start_frame,
                color="#0F766E",
                linestyle="--",
                linewidth=1.4,
                zorder=6,
                label=_leg(f"Focal f{focal.start_frame}"),
            )

        if self._video_scrubbed_frame is not None:
            ax.axvline(
                self._video_scrubbed_frame,
                color="#0F172A",
                linewidth=1.8,
                zorder=7,
                label=_leg(f"Scrub f{self._video_scrubbed_frame}"),
            )

        ax.set_xlim(0, max_f)
        ax.set_ylim(-0.52, max(0.55, len(idents) - 0.45))
        ax.set_yticks(range(len(idents)))
        ax.set_yticklabels(
            [self._animal_display_text(rec.video, i) for i in idents],
            fontsize=9.2,
            fontweight="bold",
        )
        ax.set_xlabel(
            f"Frame (0\u2013{max_f} = {max_f / 30.0:.0f}s \u2022 click to scrub)",
            fontsize=9.2,
            fontweight="bold",
            color="#1E293B",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", color="#E2E8F0", linewidth=0.8)
        if seen_lbls:
            tl_w_in = float(self.insp_canvas.figure.get_figwidth())
            tl_ncol = min(5 if tl_w_in >= 5.2 else 4, len(seen_lbls))
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                fontsize=8.0,
                ncol=tl_ncol,
                frameon=False,
                handletextpad=0.30,
                columnspacing=0.80,
                borderaxespad=0.0,
            )
        self.insp_canvas.mark_layout_dirty()
        self.insp_canvas.draw_idle()

    def _on_video_timeline_frame_clicked(self, x_frame: float) -> None:
        rec = self._get_video_record(self._selected_video_name)
        if rec is None:
            return
        max_f = max(1, (rec.max_track_frames or 1800) - 1)
        target_f = max(0, min(round(x_frame), max_f))
        self._video_scrubbed_frame = target_f
        if self._central_widget is not None:
            loaded = getattr(self._central_widget, "loaded_video", None)
            if loaded is None or Path(loaded).name != Path(rec.video).name:
                self._seek_video_in_main_window(rec, frame_zero=True)
            if hasattr(self._central_widget, "seek_to_frame_only"):
                self._central_widget.seek_to_frame_only(target_f)
            self._request_bring_main_window_to_front()
        self._draw_video_timeline_plot(rec)
        self.status_summary_label.setText(
            f"Scrubbed {_short_video_name(rec.video)} to frame {target_f} ({target_f / 30.0:.1f}s)"
        )

    def _seek_video_in_main_window(
        self,
        rec: VideoSummaryRecord,
        *,
        frame_zero: bool = False,
        play_focal: bool = False,
        bring_to_front: bool = False,
    ) -> None:
        if self._central_widget is None:
            return
        self._video_seek_epoch += 1
        focal = rec.focal_event_for(self._video_preset, self._video_source)
        beh_target = self._selected_single_behavior() or (
            self._video_report.behavior if self._video_report is not None else ""
        )
        ident = 0 if frame_zero else int(focal.identity)
        start_f = 0 if frame_zero else int(focal.start_frame)
        end_f = (
            None if (frame_zero or focal.end_frame <= focal.start_frame) else int(focal.end_frame)
        )

        self._ignoring_self_seek_signal = True
        try:
            if play_focal and hasattr(self._central_widget, "play_bout_clip"):
                play_end = (
                    int(focal.end_frame)
                    if focal.end_frame > focal.start_frame
                    else min(max(30, (rec.max_track_frames or 1800) - 1), start_f + 90)
                )
                self._central_widget.play_bout_clip(
                    rec.video,
                    ident,
                    start_f,
                    play_end,
                    beh_target or None,
                    context_frames=10,
                )
            elif hasattr(self._central_widget, "seek_to_bout"):
                try:
                    self._central_widget.seek_to_bout(
                        rec.video,
                        ident,
                        start_f,
                        end_f,
                        beh_target or None,
                        select_span=(end_f is not None),
                    )
                except TypeError:
                    self._central_widget.seek_to_bout(
                        rec.video,
                        ident,
                        start_f,
                        end_f,
                        beh_target or None,
                    )
            elif hasattr(self._central_widget, "load_video"):
                vid_path = Path(self._project.project_paths.project_dir) / rec.video
                self._central_widget.load_video(vid_path)
        finally:
            self._ignoring_self_seek_signal = False
        if bring_to_front:
            self._request_bring_main_window_to_front()

    def _on_video_open_clicked(self) -> None:
        rec = self._get_video_record(self._selected_video_name)
        if rec is None:
            return
        self._seek_video_in_main_window(rec, frame_zero=False, bring_to_front=True)
        focal = rec.focal_event_for(self._video_preset, self._video_source)
        self.status_summary_label.setText(
            f"Opened {_short_video_name(rec.video)} @ frame {focal.start_frame} ({focal.reason})"
        )

    def _on_video_open_start_clicked(self) -> None:
        rec = self._get_video_record(self._selected_video_name)
        if rec is None:
            return
        self._video_scrubbed_frame = 0
        self._seek_video_in_main_window(rec, frame_zero=True, bring_to_front=True)
        self._draw_video_timeline_plot(rec)
        self.status_summary_label.setText(f"Opened {_short_video_name(rec.video)} at frame 0")

    def _on_video_play_focal_clicked(self) -> None:
        rec = self._get_video_record(self._selected_video_name)
        if rec is None:
            return
        self._seek_video_in_main_window(rec, frame_zero=False, play_focal=True)

    def _apply_video_name_cohort_in_bout_mode(self, video_name: str) -> None:
        if self._report is None:
            return
        records = self._report.feature_matrix.records
        eligible = self._mode_eligible_indices()
        target_base = Path(video_name).name
        chosen = sorted(
            (gi for gi in eligible if Path(records[gi].video).name == target_base),
            key=lambda gi: (int(records[gi].identity), records[gi].start_frame),
        )
        if chosen:
            self._apply_custom_selection(
                chosen,
                _short_video_name(video_name),
                preserve_order=True,
            )
        else:
            self._populate_table(auto_fallback_if_clean=True, redraw_scatter_on_change=False)
            self._draw_scatter()
            self.status_summary_label.setText(
                f"0 bouts for {_short_video_name(video_name)} in this mode."
            )

    def _on_video_inspect_bouts_clicked(self, target_mode: Literal["audit", "predict"]) -> None:
        rec = self._get_video_record(self._selected_video_name)
        if rec is None:
            return
        video_name = rec.video
        if target_mode == "predict":
            self.mode_predict_btn.setChecked(True)
            need_fetch = not self._report_includes_predictions or self._report is None
            if need_fetch:
                self._pending_select_video_after_refresh = video_name
                self._on_mode_pill_clicked(1)
                return
            self._on_mode_pill_clicked(1)
            self._apply_video_name_cohort_in_bout_mode(video_name)
        else:
            self.mode_audit_btn.setChecked(True)
            if self._report is None:
                self._pending_select_video_after_refresh = video_name
                self._on_mode_pill_clicked(0)
                return
            self._on_mode_pill_clicked(0)
            self._apply_video_name_cohort_in_bout_mode(video_name)

    def _on_video_table_context_menu(self, local_pos: QtCore.QPoint) -> None:
        self._clear_video_table_mouse_press_state()
        if self._video_report is None:
            return
        row = self.video_table.rowAt(local_pos.y())
        vname = self._selected_video_name
        if row >= 0:
            it = self.video_table.item(row, 0)
            if it is not None:
                vname = str(it.data(Qt.ItemDataRole.UserRole) or "")
                if self.video_table.currentRow() != row:
                    self._suppress_auto_seek = True
                    try:
                        self.video_table.selectRow(row)
                    finally:
                        self._suppress_auto_seek = False
        self._show_video_context_menu(self.video_table.viewport().mapToGlobal(local_pos), vname)

    def _show_video_context_menu(self, global_pos: QtCore.QPoint, video_name: str | None) -> None:
        menu = QtWidgets.QMenu(self)
        rec = self._get_video_record(video_name)
        if rec is not None:
            act_open = menu.addAction(f"▶ Open {_short_video_name(rec.video)} in JABS [Enter]")
            act_open.setEnabled(self._central_widget is not None)
            act_open.triggered.connect(self._on_video_open_clicked)

            act_f0 = menu.addAction("⏮ Open at Frame 0")
            act_f0.setEnabled(self._central_widget is not None)
            act_f0.triggered.connect(self._on_video_open_start_clicked)

            menu.addSeparator()
            act_p = menu.addAction(f"🔍 Inspect Predicted Bouts ({rec.pred_bout_count})")
            act_p.setEnabled(rec.pred_bout_count > 0)
            act_p.triggered.connect(lambda: self._on_video_inspect_bouts_clicked("predict"))

            act_l = menu.addAction(
                f"🔍 Audit Human Labels (+{rec.label_pos_bouts} / -{rec.label_neg_bouts})"
            )
            act_l.setEnabled(rec.has_labels)
            act_l.triggered.connect(lambda: self._on_video_inspect_bouts_clicked("audit"))

        if self._scatter_view_limits is not None or self._video_lasso_selected:
            menu.addSeparator()
        if self._scatter_view_limits is not None:
            act_rz = menu.addAction("⟲ Reset 2D Map Zoom [0]")
            act_rz.triggered.connect(self._reset_scatter_zoom)
        if self._video_lasso_selected:
            act_clr = menu.addAction("✕ Clear Map Selection [Esc]")
            act_clr.triggered.connect(self._on_clear_video_map_selection)

        if menu.actions():
            menu.exec(global_pos)

    def _on_send_videos_to_search_bar(self) -> None:
        if self._video_report is None or self._central_widget is None:
            return
        # Preserve exact displayed table sort order
        ordered_names: list[str] = []
        for r in range(self.video_table.rowCount()):
            it = self.video_table.item(r, 0)
            if it is not None:
                vname = str(it.data(Qt.ItemDataRole.UserRole) or "")
                if vname:
                    ordered_names.append(vname)
        by_name = {r.video: r for r in self._video_report.records}
        beh_lbl = self._selected_single_behavior() or self._video_report.behavior
        hits: list[SearchHit] = []
        for vname in ordered_names:
            rec = by_name.get(vname)
            if rec is None:
                continue
            focal = rec.focal_event_for(self._video_preset, self._video_source)
            end_f = focal.end_frame if focal.end_frame >= focal.start_frame else focal.start_frame
            hits.append(
                SearchHit(
                    file=rec.video,
                    identity=str(focal.identity),
                    behavior=beh_lbl,
                    start_frame=focal.start_frame,
                    end_frame=end_f,
                )
            )
        if not hits:
            MessageDialog.warning(self, message="No videos in the current filter to send.")
            return
        preset_txt = self.videos_preset_combo.currentText()
        desc = f"Video Screener: {preset_txt} ({len(hits)} videos)"
        self._central_widget.load_custom_search_queue(hits, description=desc)
        self.status_summary_label.setText(
            f"Sent {len(hits)} videos to JABS Search Bar (< Prev | Next >)"
        )

    def _on_export_videos_csv_clicked(self) -> None:
        if self._video_report is None:
            return
        default_dir = Path(self._video_report.project_dir) / "jabs" / "training_logs"
        default_dir.mkdir(parents=True, exist_ok=True)
        scope_lbl = (
            "All_Classes"
            if self._video_report.include_all_multiclass
            else self._video_report.behavior
        )
        csv_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Whole-Video Screening Summary (CSV)",
            str(default_dir / f"video_screener_{scope_lbl}.csv"),
            "CSV Files (*.csv)",
        )
        if not csv_path:
            return
        out_csv = Path(csv_path)
        full_df = self._video_report.to_dataframe(source=self._video_source)
        # Export the currently visible rows in their exact displayed table sort order
        visible_names: list[str] = []
        for r in range(self.video_table.rowCount()):
            it = self.video_table.item(r, 0)
            if it is not None:
                vname = str(it.data(Qt.ItemDataRole.UserRole) or "")
                if vname:
                    visible_names.append(vname)
        if visible_names and not full_df.empty:
            indexed = full_df.set_index("video")
            ordered = [v for v in visible_names if v in indexed.index]
            df = indexed.loc[ordered].reset_index()
        else:
            df = full_df.iloc[0:0].copy() if self.video_table.rowCount() == 0 else full_df
        df.to_csv(out_csv, index=False)
        is_filtered = len(df) < len(full_df)
        count_txt = (
            f"{_plural(len(df), 'filtered video')} (of {len(full_df)})"
            if is_filtered
            else _plural(len(df), "video")
        )
        MessageDialog.information(
            self,
            title=f"Video Screener Exported ({scope_lbl})",
            message=f"Saved whole-video screening table ({count_txt}) to:\n- {out_csv}",
        )

    def _open_help_guide(self) -> None:
        """Open the JABS User Guide directly on the BALE (Bout & Label Explorer) page."""
        win = self.parent() or (
            self._central_widget.window() if self._central_widget is not None else None
        )
        if win is not None and hasattr(win, "menu_handlers"):
            win.menu_handlers.open_user_guide("bale.md")
            return
        if self._standalone_help_dialog is None:
            self._standalone_help_dialog = UserGuideDialog(
                "JABS", parent=None, initial_page="bale.md"
            )
        else:
            self._standalone_help_dialog.show_page("bale.md")
        self._standalone_help_dialog.show()
        self._standalone_help_dialog.raise_()
        self._standalone_help_dialog.activateWindow()
