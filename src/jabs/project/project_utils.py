import re

import h5py


def to_safe_name(behavior: str) -> str:
    """
    Create a version of the given behavior name that should be safe to use in filenames.
    :param behavior: string behavior name
    :returns: sanitized behavior name
    :raises ValueError: if the behavior name is empty after sanitization
    """

    safe_behavior = re.sub(r"[^\w.-]+", "_", behavior, flags=re.UNICODE)
    # get rid of consecutive underscores
    safe_behavior = re.sub("_{2,}", "_", safe_behavior)

    # Remove leading and trailing underscores
    safe_behavior = safe_behavior.lstrip("_").rstrip("_")

    if safe_behavior == "":
        raise ValueError("Behavior name is empty after sanitization.")
    return safe_behavior
