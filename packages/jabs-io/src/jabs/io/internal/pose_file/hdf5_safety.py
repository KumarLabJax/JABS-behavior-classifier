"""Refuse HDF5 files whose data is not all inside the file.

HDF5 can keep a dataset's values somewhere else: an external link names
another file, a dataset with external storage reads raw bytes from other files,
and a virtual dataset maps other files' datasets into it. h5py follows each
one as soon as the data is read, so a crafted upload converted, validated or
read on a server would copy that server's files into the result. A soft link
stays inside the file, but it can dangle or loop, and nothing JABS writes uses
one.

ADR 0002 states no rule about links or external storage. This check is a
safety rule for untrusted input, not a format rule: every file the writer
produces passes it.

The walk reads links, never what they point at: ``visititems`` follows soft
links, and can loop on them.
"""

import logging

import h5py

logger = logging.getLogger(__name__)


def check_self_contained(h5: h5py.File) -> list[str]:
    """Return every link or dataset in the file that refers to data outside it.

    Every link from the root down is examined without being followed. Only
    hard links are accepted; a hard link to a group already seen (a second
    name, or a cycle) is not walked again.

    Args:
        h5: The open file.

    Returns:
        One problem per offending path, such as
        ``"/poseest/bbox is an external link to other.h5:/data"``; empty when
        the file is self-contained.
    """
    problems: list[str] = []
    root = h5["/"]
    seen = {root.id}
    pending: list[h5py.Group] = [root]
    while pending:
        group = pending.pop()
        prefix = group.name.rstrip("/")
        for name in group:
            path = f"{prefix}/{name}"
            try:
                link = group.get(name, getlink=True)
            except (TypeError, KeyError, OSError) as error:
                # h5py raises TypeError for a user-defined link class.
                problems.append(f"{path} is a link h5py cannot read: {error}")
                continue
            if isinstance(link, h5py.ExternalLink):
                problems.append(f"{path} is an external link to {link.filename}:{link.path}")
                continue
            if isinstance(link, h5py.SoftLink):
                problems.append(f"{path} is a soft link to {link.path}")
                continue
            if not isinstance(link, h5py.HardLink):
                problems.append(f"{path} is a link of an unknown kind: {type(link).__name__}")
                continue
            try:
                obj = group[name]
                if isinstance(obj, h5py.Group):
                    if obj.id not in seen:
                        seen.add(obj.id)
                        pending.append(obj)
                elif isinstance(obj, h5py.Dataset):
                    problems.extend(_dataset_problems(path, obj))
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                # A damaged object whose storage cannot be inspected cannot be
                # shown to be inside the file.
                problems.append(f"{path} cannot be opened to check its storage: {error}")
    if problems:
        logger.debug("%s refers to data outside it: %s", h5.filename, "; ".join(problems))
    return sorted(problems)


def _dataset_problems(path: str, dataset: h5py.Dataset) -> list[str]:
    """Return a dataset's storage outside the file: external raw files, or a virtual layout."""
    problems = []
    if dataset.is_virtual:
        problems.append(f"{path} is a virtual dataset, mapping other datasets into it")
    external = dataset.external
    if external:
        files = ", ".join(sorted({_file_name(entry[0]) for entry in external}))
        problems.append(f"{path} is a dataset with external storage in {files}")
    return problems


def _file_name(value: object) -> str:
    """Return an external storage file name as text, whether h5py gave bytes or str."""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
