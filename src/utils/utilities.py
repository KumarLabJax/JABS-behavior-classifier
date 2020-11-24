import os
import sys
from contextlib import contextmanager


@contextmanager
def hide_stderr():
    fd = sys.stderr.fileno()

    # copy fd before it is overwritten
    with os.fdopen(os.dup(fd), 'wb') as copied:
        sys.stderr.flush()

        # open destination
        with open(os.devnull, 'wb') as fout:
            os.dup2(fout.fileno(), fd)
        try:
            yield fd
        finally:
            # restore stderr to its previous value
            sys.stderr.flush()
            os.dup2(copied.fileno(), fd)
