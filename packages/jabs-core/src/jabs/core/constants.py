ORG_NAME = "JAX"
APP_NAME = "JABS"
APP_NAME_LONG = f"{ORG_NAME} Animal Behavior System"

# a hard coded random seed used for the final training
# This is not used during cross-validation, but to ensure that final classifier is reproducible
# we use this fixed seed when training the final model after cross validation.
FINAL_TRAIN_SEED = 0xAB3BDB

# some defaults for compressing hdf5 output
COMPRESSION = "gzip"
COMPRESSION_OPTS_DEFAULT = 6

# settings keys for project settings stored in the project.json file
CV_GROUPING_KEY = "cv_grouping"
# regex used when CV_GROUPING_KEY is the "Filename Pattern" strategy
CV_GROUPING_REGEX_KEY = "cv_grouping_regex"
CLASSIFIER_MODE_KEY = "classifier_mode"
CACHE_FORMAT_KEY = "cache_format"

# behavior-scoped settings keys stored under the "behavior" section of project.json
# ordered list of prediction postprocessing stage configurations
POSTPROCESSING_KEY = "postprocessing"
# when true, cross-validation also reports metrics with the postprocessing pipeline applied
EVALUATE_POSTPROCESSING_IN_CV_KEY = "evaluate_postprocessing_in_cv"

# reserved behavior name used in multi-class mode to store explicit negative labels
MULTICLASS_NONE_BEHAVIOR = "None"
