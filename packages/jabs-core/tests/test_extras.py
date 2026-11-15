"""Tests for reporting which optional extras of a distribution are installed."""

import importlib.metadata

import pytest

from jabs.core.utils import extras as extras_module
from jabs.core.utils.extras import ExtraStatus, extras_status


class FakeMetadata:
    """Stand-in for a distribution's metadata message, holding only Provides-Extra."""

    def __init__(self, extras: list[str]) -> None:
        self._extras = extras

    def get_all(self, key: str) -> list[str] | None:
        """Return the declared extras for ``Provides-Extra``, None for anything else."""
        return self._extras if key == "Provides-Extra" else None


@pytest.fixture
def fake_environment(monkeypatch):
    """Install a fake set of distributions, returning a function that defines them.

    The function takes ``{name: (version, provides_extra, requires_dist)}``.
    """

    def install(distributions: dict[str, tuple[str, list[str], list[str]]]) -> None:
        def lookup(name: str):
            for dist, info in distributions.items():
                if dist.replace("_", "-").lower() == name.replace("_", "-").lower():
                    return info
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(
            extras_module.importlib.metadata, "metadata", lambda n: FakeMetadata(lookup(n)[1])
        )
        monkeypatch.setattr(extras_module.importlib.metadata, "requires", lambda n: lookup(n)[2])
        monkeypatch.setattr(extras_module.importlib.metadata, "version", lambda n: lookup(n)[0])

    return install


def _by_name(statuses: list[ExtraStatus]) -> dict[str, ExtraStatus]:
    return {status.name: status for status in statuses}


def test_installed_missing_and_partial_extras(fake_environment) -> None:
    """Each extra reports its applicable packages and which of them are missing."""
    fake_environment(
        {
            "app": (
                "1.0",
                ["full", "none", "partial"],
                [
                    "core-dep>=1.0",
                    "alpha>=1.0 ; extra == 'full'",
                    "beta ; extra == 'none'",
                    "alpha ; extra == 'partial'",
                    "gamma ; extra == 'partial'",
                ],
            ),
            "core-dep": ("1.0", [], []),
            "alpha": ("2.0", [], []),
        }
    )

    statuses = _by_name(extras_status("app"))

    assert list(statuses) == ["full", "none", "partial"]
    assert statuses["full"] == ExtraStatus("full", ("alpha",), ())
    assert statuses["full"].installed
    assert statuses["none"].missing == ("beta",)
    assert not statuses["none"].installed and not statuses["none"].partially_installed
    assert statuses["partial"].missing == ("gamma",)
    assert statuses["partial"].partially_installed


def test_core_requirements_with_markers_are_not_part_of_any_extra(fake_environment) -> None:
    """A core dependency with a platform marker is not mistaken for an extra's package."""
    fake_environment(
        {
            "app": (
                "1.0",
                ["x"],
                ["core-dep ; python_version >= '3'", "alpha ; extra == 'x'"],
            ),
            "alpha": ("1.0", [], []),
        }
    )

    assert extras_status("app") == [ExtraStatus("x", ("alpha",), ())]


def test_packages_excluded_by_their_marker_are_not_required(fake_environment) -> None:
    """A package the extra skips on this Python version does not count as missing."""
    fake_environment(
        {
            "app": (
                "1.0",
                ["classifiers"],
                [
                    "catboost ; python_version < '3.0' and extra == 'classifiers'",
                    "lightgbm ; extra == 'classifiers'",
                ],
            ),
            "lightgbm": ("4.7.0", [], []),
        }
    )

    assert extras_status("app") == [ExtraStatus("classifiers", ("lightgbm",), ())]


def test_wrong_version_counts_as_missing(fake_environment) -> None:
    """A package installed at a version the extra does not accept is missing."""
    fake_environment(
        {
            "app": ("1.0", ["x"], ["alpha>=2.0 ; extra == 'x'"]),
            "alpha": ("1.5", [], []),
        }
    )

    assert extras_status("app")[0].missing == ("alpha",)


def test_nested_extras_are_followed(fake_environment) -> None:
    """An extra that needs another package's extra also needs that extra's packages."""
    fake_environment(
        {
            "app": ("1.0", ["nwb"], ["lib[nwb] ; extra == 'nwb'"]),
            "lib": ("1.0", ["nwb"], ["pynwb ; extra == 'nwb'", "ndx-pose ; extra == 'nwb'"]),
            "ndx-pose": ("1.0", [], []),
        }
    )

    status = extras_status("app")[0]

    assert status.required == ("lib", "pynwb", "ndx-pose")
    assert status.missing == ("pynwb",)


def test_extras_that_refer_to_each_other_do_not_recurse_forever(fake_environment) -> None:
    """A distribution whose extras include each other is still reported."""
    fake_environment(
        {
            "app": (
                "1.0",
                ["a", "b"],
                ["app[b] ; extra == 'a'", "app[a] ; extra == 'b'", "alpha ; extra == 'a'"],
            ),
            "alpha": ("1.0", [], []),
        }
    )

    statuses = _by_name(extras_status("app"))

    assert statuses["a"].installed
    assert "alpha" in statuses["b"].required


def test_unknown_distribution_has_no_extras(fake_environment) -> None:
    """A distribution that is not installed reports no extras rather than raising."""
    fake_environment({})

    assert extras_status("not-installed") == []


def test_real_distribution_reports_its_declared_extras() -> None:
    """Against the real environment, every extra the package declares is reported."""
    declared = importlib.metadata.metadata("jabs-core").get_all("Provides-Extra") or []

    assert [status.name for status in extras_status("jabs-core")] == sorted(declared)
