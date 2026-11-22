"""The keyed conversion cache and open_unified."""

import logging
import os
import shutil
import stat
import sys
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest

from jabs.core.utils import hash_file
from jabs.pose_estimation import PoseOpenError, conversion_cache
from jabs.pose_estimation.conversion_cache import (
    cache_root,
    converted_path,
    ensure_converted,
    open_unified,
)
from jabs.pose_estimation.pose import Pose

from .conftest import build_pose_file
from .contract import sample_path

_LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)


@pytest.fixture
def legacy_v4(tmp_path: Path) -> Path:
    """A copy of the v4 sample named like a user's pose file."""
    path = tmp_path / "videos" / "vid_pose_est_v4.h5"
    path.parent.mkdir()
    shutil.copyfile(sample_path(4), path)
    return path


@pytest.fixture
def convert_spy(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Count convert_legacy calls while still converting."""
    spy = mock.Mock(wraps=conversion_cache.convert_legacy)
    monkeypatch.setattr(conversion_cache, "convert_legacy", spy)
    return spy


def test_cache_root_is_the_cache_dir(tmp_path: Path) -> None:
    """A project's cache_dir is the cache root."""
    assert cache_root(tmp_path) == tmp_path


def test_cache_root_defaults_to_user_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a cache_dir the per-user cache directory for jabs is used."""
    calls: list[str] = []

    def fake_user_cache_dir(appname: str) -> str:
        calls.append(appname)
        return str(tmp_path / "user-cache")

    monkeypatch.setattr(conversion_cache, "user_cache_dir", fake_user_cache_dir)
    assert cache_root(None) == tmp_path / "user-cache"
    assert calls == ["jabs"]


def test_converted_path_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``<root>/converted/<video stem>.<legacy hash>-r<converter revision>.h5``."""
    monkeypatch.setattr(conversion_cache, "_converter_revision", lambda: 7)
    path = converted_path(Path("/videos/vid_pose_est_v6.h5"), "abc", tmp_path)
    assert path == tmp_path / "converted" / "vid.abc-r7.h5"


def test_key_changes_with_legacy_bytes(legacy_v4: Path, tmp_path: Path) -> None:
    """Changing the legacy file's bytes changes the key."""
    before = converted_path(legacy_v4, hash_file(legacy_v4), tmp_path)
    with legacy_v4.open("ab") as f:
        f.write(b"\0")
    after = converted_path(legacy_v4, hash_file(legacy_v4), tmp_path)
    assert before != after


def test_key_changes_with_converter_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new converter revision (converter output changed) changes the key."""
    source = Path("vid_pose_est_v6.h5")
    monkeypatch.setattr(conversion_cache, "_converter_revision", lambda: 1)
    old = converted_path(source, "abc", tmp_path)
    monkeypatch.setattr(conversion_cache, "_converter_revision", lambda: 2)
    assert converted_path(source, "abc", tmp_path) != old


def test_converter_revision_is_jabs_io_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key uses jabs-io's CONVERTER_REVISION, not the package version."""
    from jabs.io.internal.pose_file import convert

    assert conversion_cache._converter_revision() == convert.CONVERTER_REVISION
    monkeypatch.setattr(convert, "CONVERTER_REVISION", 99)
    assert conversion_cache._converter_revision() == 99
    assert converted_path(Path("vid_pose_est_v6.h5"), "abc", Path("/c")).name == "vid.abc-r99.h5"


def test_revision_bump_reconverts(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, convert_spy: mock.Mock
) -> None:
    """Bumping the revision converts again and removes the old revision's copy."""
    cache = tmp_path / "cache"
    digest = hash_file(legacy_v4)
    old = converted_path(legacy_v4, digest, cache)
    with open_unified(legacy_v4, cache) as pose:
        assert pose.converter_revision == conversion_cache._converter_revision()
    assert old.exists()
    bumped = conversion_cache._converter_revision() + 1
    monkeypatch.setattr(conversion_cache, "_converter_revision", lambda: bumped)
    with open_unified(legacy_v4, cache) as pose:
        assert pose.converter_revision == bumped
    assert convert_spy.call_count == 2
    assert converted_path(legacy_v4, digest, cache).exists()
    assert not old.exists()


def test_unified_file_has_no_converter_revision(tmp_path: Path) -> None:
    """A unified file was not converted, so it has no converter revision."""
    path = build_pose_file(tmp_path / "vid_pose.h5")
    with open_unified(path, tmp_path / "cache") as pose:
        assert pose.converter_revision is None


def test_second_open_converts_zero_times(
    legacy_v4: Path, tmp_path: Path, convert_spy: mock.Mock
) -> None:
    """The second open of an unchanged file reuses the conversion."""
    cache = tmp_path / "cache"
    with open_unified(legacy_v4, cache) as first:
        assert isinstance(first, Pose)
    assert convert_spy.call_count == 1
    with open_unified(legacy_v4, cache) as second:
        assert second.hash == hash_file(legacy_v4)
    assert convert_spy.call_count == 1


def test_open_unified_hashes_the_source_once(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open of a cached conversion hashes the user's file once; the hash is Pose.hash.

    The open that converts hashes it a second time, after converting, to catch
    a file that changed while it was being converted.
    """
    spy = mock.Mock(wraps=conversion_cache.hash_file)
    monkeypatch.setattr(conversion_cache, "hash_file", spy)
    with open_unified(legacy_v4, tmp_path / "cache"):
        pass
    assert spy.call_args_list == [mock.call(legacy_v4), mock.call(legacy_v4)]

    spy.reset_mock()
    with open_unified(legacy_v4, tmp_path / "cache") as pose:
        assert pose.hash == hash_file(legacy_v4)
        assert pose.pose_file == legacy_v4
    spy.assert_called_once_with(legacy_v4)


def test_other_version_copy_removed(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversion deletes same-stem, same-hash copies of other converter versions."""
    cache = tmp_path / "cache"
    digest = hash_file(legacy_v4)
    converted = cache / "converted"
    converted.mkdir(parents=True)
    stale = converted / f"vid.{digest}-0.0.1.h5"
    stale.touch()
    other_hash = converted / "vid.0000-0.0.1.h5"
    other_hash.touch()
    other_stem = converted / f"vid2.{digest}-0.0.1.h5"
    other_stem.touch()
    monkeypatch.setattr(conversion_cache, "_converter_revision", lambda: 9)
    destination = ensure_converted(legacy_v4, digest, cache)
    assert destination == converted / f"vid.{digest}-r9.h5"
    assert destination.exists()
    assert not stale.exists()
    assert other_hash.exists()
    assert other_stem.exists()


def test_cleanup_tolerates_os_error(
    legacy_v4: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A copy that cannot be deleted (open on Windows) is logged and left."""
    cache = tmp_path / "cache"
    digest = hash_file(legacy_v4)
    (cache / "converted").mkdir(parents=True)
    stale = cache / "converted" / f"vid.{digest}-0.0.1.h5"
    stale.touch()

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "unlink", refuse)
    with caplog.at_level(logging.WARNING, logger=conversion_cache.__name__):
        destination = ensure_converted(legacy_v4, digest, cache)
    assert destination.exists()
    assert stale.exists()
    assert any(r.levelno == logging.WARNING and stale.name in r.message for r in caplog.records)


def test_replace_error_uses_existing_destination(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When another worker's copy blocks the final replace, its copy is used."""
    cache = tmp_path / "cache"
    digest = hash_file(legacy_v4)
    real_convert = conversion_cache.convert_legacy

    def racing_convert(source: Path, destination: Path) -> None:
        real_convert(source, destination)
        raise PermissionError("destination is open in another process")

    monkeypatch.setattr(conversion_cache, "convert_legacy", racing_convert)
    destination = ensure_converted(legacy_v4, digest, cache)
    assert destination == converted_path(legacy_v4, digest, cache)
    assert destination.exists()


def test_os_error_without_destination_is_pose_open_error(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError that leaves no conversion behind is a PoseOpenError."""

    def failing_convert(source: Path, destination: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(conversion_cache, "convert_legacy", failing_convert)
    with pytest.raises(PoseOpenError, match="disk full") as info:
        ensure_converted(legacy_v4, "abc", tmp_path / "cache")
    assert info.value.path == legacy_v4


def test_converter_refusal_is_pose_open_error(tmp_path: Path) -> None:
    """A converter refusal names the file and the reason."""
    source = tmp_path / "vid_pose_est_v8.h5"
    shutil.copyfile(_LEGACY_CORPUS / "v8-duplicate-embed.h5", source)
    with pytest.raises(PoseOpenError) as info:
        open_unified(source, tmp_path / "cache")
    error = info.value
    assert error.path == source
    assert error.reason
    assert str(source) in str(error)
    assert error.reason in str(error)
    assert "identity" in error.reason
    # the path is named once, not repeated from the converter's message
    assert str(error).count(str(source)) == 1
    assert not list((tmp_path / "cache" / "converted").glob("*.h5"))


def test_unified_file_opens_without_conversion(tmp_path: Path, convert_spy: mock.Mock) -> None:
    """A ``*_pose.h5`` file is read in place, never converted."""
    path = build_pose_file(tmp_path / "vid_pose.h5")
    cache = tmp_path / "cache"
    with open_unified(path, cache) as pose:
        assert pose.pose_file == path
        assert pose.hash == hash_file(path)
        assert pose.source_format == "jabs.pose-file"
    convert_spy.assert_not_called()
    assert not (cache / "converted").exists()


def test_unreadable_unified_file_is_pose_open_error(tmp_path: Path) -> None:
    """A unified-named file that is not a pose file is a PoseOpenError."""
    path = tmp_path / "vid_pose.h5"
    path.write_bytes(b"not hdf5")
    with pytest.raises(PoseOpenError) as info:
        open_unified(path, tmp_path / "cache")
    assert info.value.path == path


def test_missing_file_is_pose_open_error(tmp_path: Path) -> None:
    """A missing file is a PoseOpenError, not a bare OSError."""
    path = tmp_path / "vid_pose_est_v4.h5"
    with pytest.raises(PoseOpenError) as info:
        open_unified(path, tmp_path / "cache")
    assert info.value.path == path


def test_unknown_name_is_pose_open_error(tmp_path: Path) -> None:
    """A name following neither pattern is refused, naming both patterns."""
    path = build_pose_file(tmp_path / "vid.h5")
    with pytest.raises(PoseOpenError, match=r"_pose\.h5"):
        open_unified(path, tmp_path / "cache")


def test_user_cache_used_without_cache_dir(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_unified with no cache_dir converts into the per-user cache."""
    user_cache = tmp_path / "user-cache"
    monkeypatch.setattr(conversion_cache, "user_cache_dir", lambda appname: str(user_cache))
    with open_unified(legacy_v4) as pose:
        assert pose.num_frames > 0
    assert len(list((user_cache / "converted").glob("vid.*.h5"))) == 1


def test_corrupt_cached_conversion_is_reconverted(
    legacy_v4: Path,
    tmp_path: Path,
    convert_spy: mock.Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A truncated cached conversion is deleted and converted again, once."""
    cache = tmp_path / "cache"
    with open_unified(legacy_v4, cache) as pose:
        expected_frames = pose.num_frames
    cached = converted_path(legacy_v4, hash_file(legacy_v4), cache)
    data = cached.read_bytes()
    cached.write_bytes(data[: len(data) // 2])

    with (
        caplog.at_level(logging.WARNING, logger=conversion_cache.__name__),
        open_unified(legacy_v4, cache) as pose,
    ):
        assert pose.num_frames == expected_frames
    assert convert_spy.call_count == 2
    assert cached.stat().st_size > len(data) // 2
    assert any(r.levelno == logging.WARNING and cached.name in r.message for r in caplog.records)


def test_conversion_that_cannot_be_opened_twice_is_pose_open_error(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the reconversion cannot be opened either, PoseOpenError; no copy is left."""
    convert = mock.Mock(side_effect=lambda source, destination: destination.write_bytes(b"junk"))
    monkeypatch.setattr(conversion_cache, "convert_legacy", convert)
    cache = tmp_path / "cache"
    with pytest.raises(PoseOpenError) as info:
        open_unified(legacy_v4, cache)
    assert info.value.path == legacy_v4
    assert convert.call_count == 2
    assert not converted_path(legacy_v4, hash_file(legacy_v4), cache).exists()


def test_broken_unified_file_is_never_deleted(tmp_path: Path) -> None:
    """A unified file the user has that cannot be opened is reported, not deleted."""
    path = tmp_path / "vid_pose.h5"
    path.write_bytes(b"not hdf5")
    with pytest.raises(PoseOpenError):
        open_unified(path, tmp_path / "cache")
    assert path.read_bytes() == b"not hdf5"


def test_replaced_legacy_file_is_reconverted(
    legacy_v4: Path, tmp_path: Path, convert_spy: mock.Mock
) -> None:
    """New legacy bytes give a new key: a new conversion, a new hash, the new data."""
    cache = tmp_path / "cache"
    with open_unified(legacy_v4, cache) as pose:
        old_hash = pose.hash
        identity = pose.identities[0]
        old_points, _ = pose.get_identity_poses(identity)
        old_points = old_points.copy()
    old_copy = converted_path(legacy_v4, old_hash, cache)

    # replace the file with different valid content: every keypoint moves
    replacement = tmp_path / "replacement_pose_est_v4.h5"
    shutil.copyfile(legacy_v4, replacement)
    with h5py.File(replacement, "a") as h5:
        points = h5["poseest/points"]
        points[...] = points[...] // 2
    shutil.copyfile(replacement, legacy_v4)

    with open_unified(legacy_v4, cache) as pose:
        assert pose.hash != old_hash
        assert pose.hash == hash_file(legacy_v4)
        new_points, _ = pose.get_identity_poses(identity)
        assert not np.array_equal(new_points, old_points, equal_nan=True)
    assert convert_spy.call_count == 2
    new_copy = converted_path(legacy_v4, hash_file(legacy_v4), cache)
    assert convert_spy.call_args.args[1] == new_copy
    assert new_copy != old_copy


def test_source_changed_during_conversion_is_discarded(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source rewritten while it converts is refused; nothing is kept under the stale key."""
    real_convert = conversion_cache.convert_legacy

    def convert_then_change_source(source: Path, destination: Path) -> None:
        real_convert(source, destination)
        with source.open("ab") as f:
            f.write(b"\0")

    monkeypatch.setattr(conversion_cache, "convert_legacy", convert_then_change_source)
    cache = tmp_path / "cache"
    digest = hash_file(legacy_v4)
    with pytest.raises(PoseOpenError, match="changed") as info:
        ensure_converted(legacy_v4, digest, cache)
    assert info.value.path == legacy_v4
    assert not converted_path(legacy_v4, digest, cache).exists()
    assert not list((cache / "converted").glob("*.h5"))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_conversion_mode_follows_the_umask(legacy_v4: Path, tmp_path: Path) -> None:
    """A conversion is readable as widely as the umask allows, like any created file."""
    previous = os.umask(0o022)
    try:
        converted = ensure_converted(legacy_v4, hash_file(legacy_v4), tmp_path / "cache")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(converted.stat().st_mode) == 0o644


def test_cache_root_honors_jabs_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """JABS_CACHE_DIR, when set, wins over the per-user cache directory."""
    monkeypatch.setenv("JABS_CACHE_DIR", str(tmp_path / "env-cache"))
    assert cache_root(None) == tmp_path / "env-cache"
    # a project's cache directory still wins
    assert cache_root(tmp_path / "project") == tmp_path / "project"
    monkeypatch.setenv("JABS_CACHE_DIR", "")
    monkeypatch.setattr(conversion_cache, "user_cache_dir", lambda _app: str(tmp_path / "user"))
    assert cache_root(None) == tmp_path / "user"


def test_jabs_cache_dir_receives_conversions(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open without a cache_dir converts into JABS_CACHE_DIR."""
    monkeypatch.setenv("JABS_CACHE_DIR", str(tmp_path / "env-cache"))
    with open_unified(legacy_v4):
        pass
    assert converted_path(legacy_v4, hash_file(legacy_v4), None).parent == (
        tmp_path / "env-cache" / "converted"
    )
    assert converted_path(legacy_v4, hash_file(legacy_v4), None).exists()


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions; root ignores them",
)
def test_unwritable_user_cache_converts_into_a_temporary_directory(
    legacy_v4: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without a writable per-user cache, one-off opens still work and leave nothing behind."""
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    monkeypatch.setenv("JABS_CACHE_DIR", str(read_only))
    temporaries: list[Path] = []
    real_mkdtemp = conversion_cache.tempfile.mkdtemp

    def recording_mkdtemp(*args: object, **kwargs: object) -> str:
        temporaries.append(Path(real_mkdtemp(*args, **kwargs)))
        return str(temporaries[-1])

    monkeypatch.setattr(conversion_cache.tempfile, "mkdtemp", recording_mkdtemp)
    read_only.chmod(0o500)
    try:
        with caplog.at_level(logging.WARNING, logger=conversion_cache.__name__):
            for _ in range(2):
                with open_unified(legacy_v4) as pose:
                    assert pose.num_frames > 0
                    assert temporaries[-1].is_dir()
                assert not temporaries[-1].exists()
    finally:
        read_only.chmod(0o700)
    assert list(read_only.iterdir()) == []
    assert len(temporaries) == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(read_only) in warnings[0].getMessage()


def test_newer_revision_copy_is_kept(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversion deletes only older revisions' copies, never a newer JABS's.

    Two JABS versions sharing a project would otherwise delete each other's
    conversions on every open.
    """
    cache = tmp_path / "cache"
    digest = hash_file(legacy_v4)
    converted = cache / "converted"
    converted.mkdir(parents=True)
    older = converted / f"vid.{digest}-r2.h5"
    newer = converted / f"vid.{digest}-r4.h5"
    much_newer = converted / f"vid.{digest}-r10.h5"
    version_named = converted / f"vid.{digest}-0.4.0.h5"
    unrecognized = converted / f"vid.{digest}-something.h5"
    for path in (older, newer, much_newer, version_named, unrecognized):
        path.touch()
    monkeypatch.setattr(conversion_cache, "_converter_revision", lambda: 3)
    destination = ensure_converted(legacy_v4, digest, cache)
    assert destination == converted / f"vid.{digest}-r3.h5"
    assert not older.exists()
    assert not version_named.exists()
    assert newer.exists()
    assert much_newer.exists()
    assert unrecognized.exists()


@pytest.fixture
def unsearchable_converted(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Make ``Path.exists`` raise for anything in a ``converted`` directory listed here.

    Before Python 3.14, ``Path.exists`` re-raises EACCES, as for a shared
    cache whose ``converted/`` another user created 0o700.
    """
    blocked: list[Path] = []
    real_exists = Path.exists

    def exists(self: Path, *args: object, **kwargs: object) -> bool:
        if any(self.parent == directory for directory in blocked):
            raise PermissionError(13, "Permission denied", str(self))
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)
    return blocked


def test_unsearchable_project_cache_is_not_a_raw_permission_error(
    legacy_v4: Path, tmp_path: Path, unsearchable_converted: list[Path]
) -> None:
    """A conversion directory that cannot be searched never raises a raw PermissionError."""
    cache = tmp_path / "cache"
    unsearchable_converted.append(cache / "converted")
    try:
        with open_unified(legacy_v4, cache) as pose:
            assert pose.num_frames > 0
    except PoseOpenError:
        pass


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions; root ignores them",
)
def test_unsearchable_conversion_directory_is_pose_open_error(
    legacy_v4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real ``converted/`` another user made 0o700: an error for a project, a fallback without."""
    cache = tmp_path / "cache"
    converted = cache / "converted"
    converted.mkdir(parents=True)
    converted.chmod(0)
    monkeypatch.setenv("JABS_CACHE_DIR", str(cache))
    try:
        with pytest.raises(PoseOpenError):
            open_unified(legacy_v4, cache)
        with open_unified(legacy_v4) as pose:
            assert pose.num_frames > 0
    finally:
        converted.chmod(0o700)
