"""Cache format settings group for configuring the project's feature cache format."""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLabel, QSizePolicy

from jabs.core.constants import CACHE_FORMAT_KEY
from jabs.core.enums import CacheFormat

from .settings_group import SettingsGroup

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_FORMAT = CacheFormat.PARQUET

_DISPLAY_NAMES: dict[CacheFormat, str] = {
    CacheFormat.HDF5: "HDF5",
    CacheFormat.PARQUET: "Parquet",
}


class CacheFormatSettingsGroup(SettingsGroup):
    """Settings group for feature cache format configuration.

    Controls whether per-identity feature caches are written as HDF5 files or
    as Parquet files. The new format takes effect on the next cache miss; use
    *File > Clear Feature Cache* to force regeneration in the new format.
    """

    def __init__(self, parent=None):
        """Initialize the cache format settings group."""
        super().__init__("Feature Cache Format", parent)

    def _create_controls(self) -> None:
        """Create the cache format combo box control."""
        self._format_combo = QComboBox()
        for fmt in CacheFormat:
            self._format_combo.addItem(_DISPLAY_NAMES[fmt], fmt)
        self._format_combo.setCurrentIndex(self._format_combo.findData(_DEFAULT_CACHE_FORMAT))
        self._format_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.add_control_row("Cache Format:", self._format_combo)

    def _create_documentation(self) -> QLabel:
        """Create help text for the cache format setting."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>What is the Feature Cache Format?</h3>
            <p>JABS pre-computes features and stores them on disk so that
            subsequent training and classification runs can skip the computation step.
            This setting controls the file format used for that cache.</p>

            <ul>
              <li><b>HDF5:</b> The original cache format. Compatible with all versions
              of JABS.</li>
              <li><b>Parquet:</b> A columnar storage format that offers faster reads
              and smaller file sizes.</li>
            </ul>

            <p><b>Note:</b> Changing this setting takes effect on the next cache miss.
            Existing cache files are not automatically converted. To regenerate the
            cache in the new format, use <i>File &gt; Clear Feature Cache&hellip;</i>
            before training or classifying.</p>
            """
        )
        help_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        return help_label

    def get_values(self) -> dict:
        """Return the currently selected cache format.

        Returns:
            Dict with ``CACHE_FORMAT_KEY`` mapped to the selected ``CacheFormat``.
        """
        return {CACHE_FORMAT_KEY: self._format_combo.currentData()}

    def set_values(self, values: dict) -> None:
        """Populate the combo box from a settings dict.

        Args:
            values: Dict that may contain ``CACHE_FORMAT_KEY``. Unrecognized
                values fall back to ``_DEFAULT_CACHE_FORMAT``.
        """
        raw = values.get(CACHE_FORMAT_KEY, _DEFAULT_CACHE_FORMAT)
        try:
            enum_val = CacheFormat(raw) if not isinstance(raw, CacheFormat) else raw
            index = self._format_combo.findData(enum_val)
        except ValueError:
            index = -1

        if index >= 0:
            self._format_combo.setCurrentIndex(index)
        else:
            logger.error(
                "Unrecognized cache_format value %r; defaulting to %s",
                raw,
                _DEFAULT_CACHE_FORMAT.value,
            )
            self._format_combo.setCurrentIndex(self._format_combo.findData(_DEFAULT_CACHE_FORMAT))
