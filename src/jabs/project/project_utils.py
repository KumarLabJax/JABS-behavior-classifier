import re

import h5py


def to_safe_name(behavior: str) -> str:
    """
    Create a version of the given behavior name that
    should be safe to use in filenames.
    :param behavior: string behavior name
    """
    safe_behavior = re.sub("[^0-9a-zA-Z]+", "_", behavior).rstrip("_")
    # get rid of consecutive underscores
    safe_behavior = re.sub("_{2,}", "_", safe_behavior)
    return safe_behavior
