from PySide6.QtCore import Qt
from jabs.postprocess import AVAILABLE_FILTERS, Postprocesser, BaseFilter

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem,
    QComboBox, QLabel, QWidget, QMessageBox
)


class SettingsListItem(QListWidgetItem):
    def __init__(self, setting_name: str, settings: dict):
        super().__init__(setting_name)
        self.setting_name = setting_name
        self.settings = settings

class PostprocessFilterListDialog(QDialog):
    def __init__(self, parent=None, project=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Postprocessing Filters")
        self.resize(400, 300)

        self.layout = QVBoxLayout(self)

        self.filter_list = QListWidget()
        self.layout.addWidget(QLabel("Sequential Filters:"))
        self.layout.addWidget(self.filter_list)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Filter")
        self.remove_btn = QPushButton("Remove Selected")
        self.up_btn = QPushButton("Move Up")
        self.down_btn = QPushButton("Move Down")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.up_btn)
        btn_layout.addWidget(self.down_btn)
        self.layout.addLayout(btn_layout)

        self.button_box = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        self.button_box.addWidget(self.ok_btn)
        self.button_box.addWidget(self.cancel_btn)
        self.layout.addLayout(self.button_box)

        self.add_btn.clicked.connect(self.add_filter)
        self.remove_btn.clicked.connect(self.remove_filter)
        self.up_btn.clicked.connect(self.move_up)
        self.down_btn.clicked.connect(self.move_down)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        # TODO: Initialize the filters from the project
        # project_filters = project.get_postprocessing_definition()
        # for post_filter, filter_settings in project_filters:
        #     self.add_filter_item(post_filter, filter_settings)

    def add_filter(self):
        dialog = FilterSelectDialog(self)
        if dialog.exec() == QDialog.Accepted:
            filter_name = dialog.selected_filter
            filter_settings = dialog.filter_settings
            self.add_filter_item(filter_name, filter_settings)

    def add_filter_item(self, filter_name, filter_settings):
        item = SettingsListItem(filter_name, filter_settings)
        self.filter_list.addItem(item)

    def remove_filter(self):
        row = self.filter_list.currentRow()
        if row >= 0:
            self.filter_list.takeItem(row)

    def move_up(self):
        row = self.filter_list.currentRow()
        if row > 0:
            item = self.filter_list.takeItem(row)
            self.filter_list.insertItem(row - 1, item)
            self.filter_list.setCurrentRow(row - 1)

    def move_down(self):
        row = self.filter_list.currentRow()
        if row < self.filter_list.count() - 1 and row >= 0:
            item = self.filter_list.takeItem(row)
            self.filter_list.insertItem(row + 1, item)
            self.filter_list.setCurrentRow(row + 1)

    def get_filters(self):
        return [self.filter_list.item(i).text() for i in range(self.filter_list.count())]

    def get_postprocessor(self):
        postprocessor = Postprocessor()
        for i in range(self.filter_list.count()):
            new_filter = AVAILABLE_FILTERS[self.filter_list.item(i).text()](self.filter_list.item(i).filter_settings)
            postprocessor.add_filter(new_filter)
        return postprocessor

class FilterSelectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._filters = {cur_filter().get_filter_name(): cur_filter for cur_filter in AVAILABLE_FILTERS}
        self.setWindowTitle("Select Filter")
        self.layout = QVBoxLayout(self)
        self.combo = QComboBox()
        self.combo.addItems(self._filters.keys())
        self.layout.addWidget(QLabel("Choose a filter:"))
        self.layout.addWidget(self.combo)

        self.settings_widgets = {}
        self.settings_layout = QVBoxLayout()
        self.layout.addLayout(self.settings_layout)
        self.combo.currentTextChanged.connect(self.update_settings_widgets)
        self.update_settings_widgets(self.combo.currentText())

        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        self.layout.addLayout(btn_layout)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def update_settings_widgets(self, filter_name):
        # Remove old widgets
        while self.settings_layout.count():
            item = self.settings_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.settings_widgets.clear()
        # Add new widgets for the selected filter
        filter_info = self._filters.get(filter_name, BaseFilter)
        default_kwargs = filter_info().get_kwargs()
        for key, value in default_kwargs.items():
            label = QLabel(f"{key}:")
            if isinstance(value, bool):
                from PySide6.QtWidgets import QCheckBox
                widget = QCheckBox()
                widget.setChecked(value)
            elif isinstance(value, (int, float)):
                from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox
                if isinstance(value, int):
                    widget = QSpinBox()
                    widget.setValue(value)
                else:
                    widget = QDoubleSpinBox()
                    widget.setValue(value)
            else:
                from PySide6.QtWidgets import QLineEdit
                widget = QLineEdit(str(value))
            self.settings_layout.addWidget(label)
            self.settings_layout.addWidget(widget)
            self.settings_widgets[key] = widget

    @property
    def selected_filter(self):
        return self.combo.currentText()

    @property
    def filter_settings(self):
        settings = {}
        filter_name = self.selected_filter
        filter_info = self._filters.get(filter_name, BaseFilter)
        default_kwargs = filter_info().get_kwargs()
        for key, default in default_kwargs.items():
            widget = self.settings_widgets.get(key)
            if widget is None:
                settings[key] = default
            elif isinstance(widget, QLabel):
                continue
            elif hasattr(widget, 'isChecked'):
                settings[key] = widget.isChecked()
            elif hasattr(widget, 'value'):
                settings[key] = widget.value()
            elif hasattr(widget, 'text'):
                text = widget.text()
                # Try to cast to original type
                if isinstance(default, int):
                    try:
                        settings[key] = int(text)
                    except Exception:
                        settings[key] = default
                elif isinstance(default, float):
                    try:
                        settings[key] = float(text)
                    except Exception:
                        settings[key] = default
                elif isinstance(default, bool):
                    settings[key] = text.lower() in ('true', '1', 'yes')
                else:
                    settings[key] = text
            else:
                settings[key] = default
        return settings
