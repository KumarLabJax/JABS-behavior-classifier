"""Unit tests for jabs.utils.update_checker module."""

import json
from importlib import metadata
from unittest.mock import MagicMock, Mock, patch

from jabs.utils.update_checker import check_for_update, is_pypi_install


class TestCheckForUpdate:
    """Test suite for check_for_update function."""

    def test_update_available(self):
        """Test when a newer version is available on PyPI."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {"version": "1.0.0"}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("jabs.utils.update_checker.version_str", return_value="0.9.0"),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            has_update, latest, current = check_for_update()

        assert has_update is True
        assert latest == "1.0.0"
        assert current == "0.9.0"

    def test_no_update_available(self):
        """Test when current version is up to date."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {"version": "1.0.0"}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("jabs.utils.update_checker.version_str", return_value="1.0.0"),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            has_update, latest, current = check_for_update()

        assert has_update is False
        assert latest == "1.0.0"
        assert current == "1.0.0"

    def test_newer_local_version(self):
        """Test when local version is newer than PyPI (e.g., dev version)."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {"version": "1.0.0"}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("jabs.utils.update_checker.version_str", return_value="2.0.0"),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            has_update, latest, current = check_for_update()

        assert has_update is False
        assert latest == "1.0.0"
        assert current == "2.0.0"

    def test_network_error(self):
        """Test handling of network errors when checking for updates."""
        with (
            patch("jabs.utils.update_checker.version_str", return_value="1.0.0"),
            patch(
                "urllib.request.urlopen",
                side_effect=Exception("Network error"),
            ),
        ):
            has_update, latest, current = check_for_update()

        assert has_update is False
        assert latest is None
        assert current == "1.0.0"

    def test_invalid_json_response(self):
        """Test handling of invalid JSON response from PyPI."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"invalid json"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("jabs.utils.update_checker.version_str", return_value="1.0.0"),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            has_update, latest, current = check_for_update()

        assert has_update is False
        assert latest is None
        assert current == "1.0.0"

    def test_missing_version_in_response(self):
        """Test handling of response missing version information."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("jabs.utils.update_checker.version_str", return_value="1.0.0"),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            has_update, latest, current = check_for_update()

        assert has_update is False
        assert latest is None
        assert current == "1.0.0"

    def test_timeout_handling(self):
        """Test that timeout is properly set in the request."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {"version": "1.0.0"}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("jabs.utils.update_checker.version_str", return_value="1.0.0"),
            patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen,
        ):
            check_for_update()

        # Verify timeout parameter was passed
        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        assert call_args[1]["timeout"] == 5

    def test_version_comparison_with_prereleases(self):
        """Test version comparison with pre-release versions."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {"version": "1.0.0rc1"}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("jabs.utils.update_checker.version_str", return_value="0.9.0"),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            has_update, latest, current = check_for_update()

        # 1.0.0rc1 > 0.9.0
        assert has_update is True
        assert latest == "1.0.0rc1"

    def test_pypi_api_url(self):
        """Test that the correct PyPI API URL is used."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"info": {"version": "1.0.0"}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with (
            patch("jabs.utils.update_checker.version_str", return_value="1.0.0"),
            patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen,
        ):
            check_for_update()

        # Verify the correct PyPI API URL was used
        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        assert call_args[0][0] == "https://pypi.org/pypi/jabs-behavior-classifier/json"


class TestIsPypiInstall:
    """Test suite for is_pypi_install function."""

    def test_pip_installation(self):
        """Test detection of pip installation."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = "pip\n"

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        assert result is True
        mock_dist.read_text.assert_called_once_with("INSTALLER")

    def test_uv_installation(self):
        """Test detection of uv installation."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = "uv\n"

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        assert result is True
        mock_dist.read_text.assert_called_once_with("INSTALLER")

    def test_pip_installation_no_whitespace(self):
        """Test pip detection without trailing whitespace."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = "pip"

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        assert result is True

    def test_other_installer(self):
        """Test detection of other installer (not pip or uv)."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = "conda\n"

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        assert result is False

    def test_package_not_found(self):
        """Test when package metadata is not found."""
        with patch(
            "importlib.metadata.distribution",
            side_effect=metadata.PackageNotFoundError("jabs-behavior-classifier"),
        ):
            result = is_pypi_install()

        assert result is False

    def test_installer_file_missing(self):
        """Test when INSTALLER file is missing."""
        mock_dist = MagicMock()
        mock_dist.read_text.side_effect = FileNotFoundError()

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        assert result is False

    def test_installer_file_none(self):
        """Test when INSTALLER file returns None."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = None

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        assert result is False

    def test_empty_installer_file(self):
        """Test when INSTALLER file is empty."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = ""

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        assert result is False

    def test_whitespace_only_installer(self):
        """Test when INSTALLER file contains only whitespace."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = "   \n  "

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        assert result is False

    def test_case_sensitive_installer_name(self):
        """Test that installer name is case-sensitive."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = "PIP\n"

        with patch("importlib.metadata.distribution", return_value=mock_dist):
            result = is_pypi_install()

        # Should be False because we check for lowercase "pip", not "PIP"
        assert result is False

    def test_generic_exception_handling(self):
        """Test handling of unexpected exceptions."""
        with patch(
            "importlib.metadata.distribution",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = is_pypi_install()

        assert result is False
