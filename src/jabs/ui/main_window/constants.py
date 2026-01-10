"""Constants for the MainWindow module."""

from jabs.utils import get_bool_env_var

# Environment variable flags
USE_NATIVE_FILE_DIALOG = get_bool_env_var("JABS_NATIVE_FILE_DIALOG", True)

# Settings keys
RECENT_PROJECTS_KEY = "recent_projects"
LICENSE_ACCEPTED_KEY = "license_accepted"
LICENSE_VERSION_KEY = "license_version"
WINDOW_SIZE_KEY = "main_window_size"
SESSION_TRACKING_ENABLED_KEY = "session_tracking_enabled"

# Default window dimensions
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
