"""Constants for the MainWindow module."""

from jabs.core.utils import get_bool_env_var

# maximum number of recent projects to show in the File->Recent Projects menu
RECENT_PROJECTS_MAX = 10

# Environment variable flags
USE_NATIVE_FILE_DIALOG = get_bool_env_var("JABS_NATIVE_FILE_DIALOG", True)

# Default window dimensions
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
