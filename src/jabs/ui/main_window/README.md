# MainWindow Package

This package contains the refactored MainWindow implementation, split into focused modules for better organization and maintainability.

## Package Structure

```
main_window/
├── __init__.py              # Package interface, re-exports MainWindow
├── central_widget.py        # Widget managing the main content area of JABS, set as MainWindow's central widget
├── constants.py             # Configuration constants and settings keys
├── menu_builder.py          # Menu construction logic
├── menu_handlers.py         # Menu action callbacks
├── main_window.py           # Core MainWindow class
├── README.md                # This file
└── video_list_widget.py     # Video List Widget, implemented as a dockable widget that can be attached to MainWindow
```

## Module Responsibilities

### `central_widget.py`
- `CentralWidget` class: Manages the main content area of the JABS Window

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

### `video_list_widget.py`
- `VideoListWidget` class: Dockable widget for displaying the list of videos in the project. By default, it is docked to the left side of MainWindow. It can dragged to the top or right sides, or floated as a separate window. It can be shown/hidden via the View menu.

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

## Questions?

If you're unsure where to add functionality:
- **Menu item/structure?** → `menu_builder.py`
- **Menu action logic?** → `menu_handlers.py`
- **Window/widget setup?** → `main_window.py`
- **Settings keys/constants?** → `constants.py`

When in doubt, follow existing patterns in the codebase.