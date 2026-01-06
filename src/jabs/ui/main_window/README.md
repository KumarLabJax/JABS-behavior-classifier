# MainWindow Package

This package contains the refactored MainWindow implementation, split into focused modules for better organization and maintainability.

## Package Structure

```
main_window/
├── __init__.py              # Package interface, re-exports MainWindow
├── constants.py             # Configuration constants and settings keys
├── menu_builder.py          # Menu construction logic
├── menu_handlers.py         # Menu action callbacks
├── main_window.py           # Core MainWindow class
└── README.md               # This file
```

## Module Responsibilities

### `constants.py`
- Application settings keys (e.g., `RECENT_PROJECTS_KEY`, `SESSION_TRACKING_ENABLED_KEY`)
- Default values (e.g., `DEFAULT_WINDOW_WIDTH`, `DEFAULT_WINDOW_HEIGHT`)
- Feature flags (e.g., `USE_NATIVE_FILE_DIALOG`)

**When to add here:** Any hardcoded string keys, default values, or configuration constants used by multiple files.

### `menu_builder.py`
- `MenuBuilder` class: Creates and configures all menus
- `MenuReferences` dataclass: Container for menu/action references
- Menu layout and structure
- Menu item creation and organization

**When to add here:** New menu items, submenus, or changes to menu structure.

### `menu_handlers.py`
- `MenuHandlers` class: Contains all menu action callback methods
- Business logic for menu actions
- Organized by menu section (File, View, Features, etc.)

**When to add here:** New menu action handlers or logic triggered by menu items.

### `main_window.py`
- `MainWindow` class: Core window implementation
- Window lifecycle management
- Project loading/unloading
- Widget initialization and layout
- Event handling (keyboard shortcuts, window events)
- Methods that can't be in MenuHandlers (e.g., keyboard shortcuts, Qt sender() limitations)

**When to add here:** Core window functionality, widget initialization, keyboard shortcuts, or methods that need access to Qt's object model.

## Design Principles

### 1. Tight Coupling is Intentional

`MenuBuilder` and `MenuHandlers` are **internal helper classes** that exist solely to organize MainWindow's code. They intentionally access MainWindow's protected members:

```python
# This is BY DESIGN - marked with # noinspection PyProtectedMember
self.window._central_widget.load_video(...)
self.window._project.clear_cache()
```

**Why?** These classes are part of MainWindow's implementation, not independent components. The module boundary (not class boundary) is the real encapsulation.

### 2. No Qt sender() Pattern

**❌ DON'T use `sender()`:**
```python
def my_handler(self, checked: bool):
    action = self.window.sender()  # BAD - tight coupling to Qt internals
    name = action.text().split(" ")[1]
```

**✅ DO use explicit parameters with lambdas:**
```python
# In menu_builder.py
action.triggered.connect(
    lambda checked, name=item_name: self.handlers.my_handler(checked, name)
)

# In menu_handlers.py
def my_handler(self, checked: bool, name: str):
    # Clear, testable, explicit dependencies
```

**Why?** `sender()` creates hidden dependencies, breaks abstraction, and makes code harder to test and understand.

## Adding New Functionality

### Adding a New Menu Item

1. **Add the menu item in `menu_builder.py`:**
   ```python
   def _build_file_menu(self, menu: QtWidgets.QMenu) -> dict:
       new_action = QtGui.QAction("My New Action", self.main_window)
       new_action.setShortcut(QtGui.QKeySequence("Ctrl+N"))
       new_action.triggered.connect(self.handlers.handle_new_action)
       menu.addAction(new_action)

       return {"new_action": new_action}  # Return for reference
   ```

2. **Add the handler in `menu_handlers.py`:**
   ```python
   def handle_new_action(self) -> None:
       """Handle the new action."""
       # Implementation here
       self.window._project.do_something()
       self.window.display_status_message("Action completed")
   ```

3. **Update `MenuReferences` if needed:**
   ```python
   @dataclass
   class MenuReferences:
       # ... existing fields ...
       new_action: QtGui.QAction  # Add if MainWindow needs direct access
   ```

### Adding a New Feature Toggle

1. **Add to feature menu in `menu_builder.py`:**
   ```python
   enable_my_feature = QtGui.QAction("Enable My Feature", self.main_window)
   enable_my_feature.setCheckable(True)
   enable_my_feature.triggered.connect(self.handlers.set_my_feature_enabled)
   menu.addAction(enable_my_feature)
   ```

2. **Add handler in `menu_handlers.py`:**
   ```python
   def set_my_feature_enabled(self, checked: bool) -> None:
       """Toggle my feature on/off."""
       behavior = self.window._central_widget.behavior
       self.window._project.settings_manager.save_behavior(
           behavior, {"my_feature": checked}
       )
   ```

### Adding a Keyboard Shortcut

If the shortcut is tied to a menu item, add it in `MenuBuilder`:
```python
action.setShortcut(QtGui.QKeySequence("Ctrl+K"))
```

If it's a standalone shortcut (no menu item), add it to `MainWindow.keyPressEvent()`:
```python
def keyPressEvent(self, event: QEvent) -> None:
    match event.key():
        case Qt.Key.Key_K if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._my_custom_action()
```

## Common Pitfalls

### ❌ Pitfall 1: Nested Dict Settings

**Problem:** Using `save_behavior()` with nested dicts overwrites the entire dict:

```python
# BAD - This will ERASE other static object settings!
self.window._project.settings_manager.save_behavior(
    behavior, {"static_objects": {landmark_name: checked}}
)
```

**Solution:** Get current dict, update one key, save all:

```python
# GOOD - Preserves other settings
all_settings = self.window._project.settings_manager.get_behavior(
    behavior
).get("static_objects", {})
all_settings[landmark_name] = checked
self.window._project.settings_manager.save_behavior(
    behavior, {"static_objects": all_settings}
)
```

### ❌ Pitfall 2: Forgetting to Return Menu References

**Problem:** Creating actions but not returning them for MainWindow:

```python
def _build_file_menu(self, menu: QtWidgets.QMenu) -> dict:
    important_action = QtGui.QAction("Important", self.main_window)
    menu.addAction(important_action)
    return {}  # BAD - MainWindow can't access important_action!
```

**Solution:** Return actions that MainWindow needs to access:

```python
def _build_file_menu(self, menu: QtWidgets.QMenu) -> dict:
    important_action = QtGui.QAction("Important", self.main_window)
    menu.addAction(important_action)
    return {"important_action": important_action}  # GOOD
```

## Examples

### Example 1: Adding a "Refresh Project" Menu Item

**Step 1:** Add to menu_builder.py:
```python
def _build_file_menu(self, menu: QtWidgets.QMenu) -> dict:
    # ... existing actions ...

    refresh_action = QtGui.QAction("Refresh Project", self.main_window)
    refresh_action.setShortcut(QtGui.QKeySequence("F5"))
    refresh_action.setStatusTip("Reload project from disk")
    refresh_action.setEnabled(False)  # Enabled when project loaded
    refresh_action.triggered.connect(self.handlers.refresh_project)
    menu.addAction(refresh_action)

    return {
        # ... existing returns ...
        "refresh_action": refresh_action,
    }
```

**Step 2:** Add to menu_handlers.py:
```python
def refresh_project(self) -> None:
    """Reload the current project from disk."""
    if not self.window._project:
        return

    project_path = self.window._project.project_dir
    self.window.open_project(project_path)
    self.window.display_status_message("Project refreshed", duration=3000)
```

**Step 3:** Update MenuReferences in menu_builder.py:
```python
@dataclass
class MenuReferences:
    # ... existing fields ...
    refresh_action: QtGui.QAction
```

**Step 4:** Enable/disable in main_window.py when project loads:
```python
def _on_project_opened(self):
    # ... existing code ...
    self._refresh_action.setEnabled(True)

def _on_project_closed(self):
    # ... existing code ...
    self._refresh_action.setEnabled(False)
```

## Testing

When testing menu functionality:
1. Test handlers independently by calling them directly
2. Mock `self.window` and its dependencies
3. Verify correct methods are called with correct parameters
4. Test that lambdas pass parameters correctly

## Questions?

If you're unsure where to add functionality:
- **Menu item/structure?** → `menu_builder.py`
- **Menu action logic?** → `menu_handlers.py`
- **Window/widget setup?** → `main_window.py`
- **Settings keys/constants?** → `constants.py`

When in doubt, follow existing patterns in the codebase.