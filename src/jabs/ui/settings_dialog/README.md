# Settings Dialog Package

This package contains the JABS project settings dialogs and related components.

## Overview

The settings dialogs are built using a modular architecture where different groups of related settings can be added as separate `SettingsGroup` subclasses. Each dialog type is specialized for a particular context:

- **`BaseSettingsDialog`**: Abstract base class for settings dialogs.
- **`ProjectSettingsDialog`**: Dialog for editing project-specific settings.
- **`JabsSettingsDialog`**: Dialog for editing global JABS application settings.

Each settings dialog can host multiple settings groups, and each group has:

- A form-style grid layout for controls
- An optional collapsible documentation section
- Methods to get/set values from the relevant settings manager

## Architecture

### Key Components

- **`BaseSettingsDialog`** - Abstract base dialog that provides the common structure for settings dialogs
- **`ProjectSettingsDialog`** - Dialog for project-level settings
- **`JabsSettingsDialog`** - Dialog for global JABS settings
- **`SettingsGroup`** - Base class for creating settings groups with controls and documentation
- **`CollapsibleSection`** - Reusable widget for collapsible help/documentation sections

### How It Works

1. The settings dialog creates a scrollable page that hosts multiple `SettingsGroup` instances
2. Each `SettingsGroup` manages its own controls and documentation
3. When the dialog opens, it loads current values from the appropriate settings manager (project or global)
4. When the user clicks "Save", all groups' values are collected and saved

## Creating a New Settings Group

To add a new group of settings:

### 1. Create a new class inheriting from `SettingsGroup`

```python
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QSpinBox, QSizePolicy
from .settings_group import SettingsGroup

class MySettingsGroup(SettingsGroup):
    """Settings group for my feature."""
    
    def __init__(self, parent=None):
        super().__init__("My Feature Settings", parent)
```

### 2. Override `_create_controls()` to add your widgets

Use `add_control_row()` to add labeled controls:

```python
def _create_controls(self) -> None:
    """Create the settings controls."""
    # Add a checkbox
    self._enable_feature = QCheckBox("Enable this feature")
    self.add_control_row("Enable feature:", self._enable_feature)
    
    # Add a combo box
    self._method_selection = QComboBox()
    self._method_selection.addItems(["Method A", "Method B", "Method C"])
    self.add_control_row("Method:", self._method_selection)
    
    # Add a spin box
    self._iterations = QSpinBox()
    self._iterations.setRange(1, 100)
    self.add_control_row("Iterations:", self._iterations)
```

Or use `add_widget_row()` for widgets that don't fit the label/control pattern:

```python
def _create_controls(self) -> None:
    # Add a full-width checkbox (no label)
    self._advanced_mode = QCheckBox("Enable advanced mode")
    self.add_widget_row(self._advanced_mode)
```

### 3. Override `_create_documentation()` to add help text (optional)

```python
def _create_documentation(self):
    """Create help documentation for these settings."""
    help_label = QLabel(self)
    help_label.setTextFormat(Qt.TextFormat.RichText)
    help_label.setWordWrap(True)
    help_label.setText(
        """
        <h3>What do these settings do?</h3>
        <p>Detailed explanation of what these settings control...</p>
        
        <ul>
          <li><b>Enable feature:</b> Description here</li>
          <li><b>Method:</b> Description of methods</li>
          <li><b>Iterations:</b> How many times to iterate</li>
        </ul>
        """
    )
    help_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return help_label
```

### 4. Implement `get_values()` and `set_values()`

```python
def get_values(self) -> dict:
    """Get current settings values."""
    return {
        "my_enable_feature": self._enable_feature.isChecked(),
        "my_method": self._method_selection.currentText(),
        "my_iterations": self._iterations.value(),
    }

def set_values(self, values: dict) -> None:
    """Set settings values from a dictionary."""
    self._enable_feature.setChecked(values.get("my_enable_feature", False))
    
    method = values.get("my_method", "Method A")
    index = self._method_selection.findText(method)
    if index >= 0:
        self._method_selection.setCurrentIndex(index)
    
    self._iterations.setValue(values.get("my_iterations", 10))
```

### 5. Add your group to the appropriate settings dialog

Edit `project_settings_dialog.py` or `jabs_settings_dialog.py` as appropriate:

```python
from .my_settings_group import MySettingsGroup

class ProjectSettingsDialog(BaseSettingsDialog):
    def __init__(self, settings_manager: SettingsManager, parent: QWidget | None = None):
        # ... existing code ...
        
        # Add your settings group
        my_group = MySettingsGroup(page)
        self._settings_groups.append(my_group)
        page_layout.addWidget(my_group)
        page_layout.setAlignment(my_group, Qt.AlignmentFlag.AlignTop)
```

## Tips

- **Setting names**: Use descriptive names and consider prefixing with the group name to avoid conflicts (e.g., `"calibration_method"` instead of just `"method"`)
- **Default values**: Always provide sensible defaults in `set_values()` using `.get(key, default)`
- **Widget sizing**: Controls are kept compact by default. The third column of the grid expands to fill extra space, keeping controls left-aligned
- **Documentation**: Rich text is supported in help sections. Use `<h3>`, `<p>`, `<ul>`, `<li>`, `<b>`, etc.
- **Validation**: Add validation in `get_values()` or connect to widget signals if needed
- **Persistence**: Setting values are persisted via the `SettingsManager` when the dialog is saved. They will appear in the jabs/project.json file under the top-level "settings" key.

## File Organization

```
settings_dialog/
├── __init__.py                    # Exports SettingsDialog and SettingsGroup
├── settings_dialog.py             # Main dialog
├── settings_group.py              # Base class for settings groups
├── collapsible_section.py         # Collapsible widget for documentation
└── README.md                      # This file
```

Add new settings groups as separate files in this directory, then import and instantiate them in the appropriate settings dialog.
