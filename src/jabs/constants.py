ORG_NAME = "JAX"
APP_NAME = "JABS"
APP_NAME_LONG = f"{ORG_NAME} Animal Behavior System"

# a hard coded random seed used for the final training done with all
# training data before saving the classifier
# the choice of random seed is arbitrary
FINAL_TRAIN_SEED = 0xAB3BDB

# some defaults for compressing hdf5 output
COMPRESSION = "gzip"
COMPRESSION_OPTS_DEFAULT = 6

# settings keys for project settings stored in the project.json file
CV_GROUPING_KEY = "cv_grouping"

# these are settings keys stored in QSettings for the JABS GUI
LICENSE_ACCEPTED_KEY = "license_accepted"
LICENSE_VERSION_KEY = "license_version"
RECENT_PROJECTS_KEY = "recent_projects"
SESSION_TRACKING_ENABLED_KEY = "session_tracking_enabled"
TORCH_BACKEND_KEY = "torch_backend"
WINDOW_SIZE_KEY = "main_window_size"
