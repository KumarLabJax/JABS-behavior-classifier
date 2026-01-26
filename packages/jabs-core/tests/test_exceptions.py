"""Unit tests for jabs.core.exceptions module."""

import pytest

from jabs.core.exceptions import (
    DistanceScaleException,
    FeatureVersionException,
    MissingBehaviorError,
    PoseHashException,
    PoseIdEmbeddingException,
)


class TestPoseHashException:
    """Tests for PoseHashException."""

    def test_inherits_from_exception(self):
        """Test that PoseHashException inherits from Exception."""
        assert issubclass(PoseHashException, Exception)

    def test_can_be_raised(self):
        """Test that PoseHashException can be raised."""
        with pytest.raises(PoseHashException):
            raise PoseHashException("Hash mismatch")

    def test_message_preserved(self):
        """Test that the exception message is preserved."""
        message = "Expected hash abc123, got def456"
        with pytest.raises(PoseHashException, match=message):
            raise PoseHashException(message)

    def test_can_be_caught_as_exception(self):
        """Test that PoseHashException can be caught as generic Exception."""
        with pytest.raises(Exception):  # noqa: B017
            raise PoseHashException("test")


class TestPoseIdEmbeddingException:
    """Tests for PoseIdEmbeddingException."""

    def test_inherits_from_exception(self):
        """Test that PoseIdEmbeddingException inherits from Exception."""
        assert issubclass(PoseIdEmbeddingException, Exception)

    def test_can_be_raised(self):
        """Test that PoseIdEmbeddingException can be raised."""
        with pytest.raises(PoseIdEmbeddingException):
            raise PoseIdEmbeddingException("Invalid embedding")

    def test_message_preserved(self):
        """Test that the exception message is preserved."""
        message = "Invalid instance_embed_id value: -1"
        with pytest.raises(PoseIdEmbeddingException, match=message):
            raise PoseIdEmbeddingException(message)


class TestMissingBehaviorError:
    """Tests for MissingBehaviorError."""

    def test_inherits_from_exception(self):
        """Test that MissingBehaviorError inherits from Exception."""
        assert issubclass(MissingBehaviorError, Exception)

    def test_can_be_raised(self):
        """Test that MissingBehaviorError can be raised."""
        with pytest.raises(MissingBehaviorError):
            raise MissingBehaviorError("Behavior not found")

    def test_message_preserved(self):
        """Test that the exception message is preserved."""
        message = "Behavior 'grooming' not found in prediction file"
        with pytest.raises(MissingBehaviorError, match=message):
            raise MissingBehaviorError(message)


class TestFeatureVersionException:
    """Tests for FeatureVersionException."""

    def test_inherits_from_exception(self):
        """Test that FeatureVersionException inherits from Exception."""
        assert issubclass(FeatureVersionException, Exception)

    def test_can_be_raised(self):
        """Test that FeatureVersionException can be raised."""
        with pytest.raises(FeatureVersionException):
            raise FeatureVersionException("Version mismatch")

    def test_message_preserved(self):
        """Test that the exception message is preserved."""
        message = "Feature version 1 not compatible with JABS version 2"
        with pytest.raises(FeatureVersionException, match=message):
            raise FeatureVersionException(message)


class TestDistanceScaleException:
    """Tests for DistanceScaleException."""

    def test_inherits_from_exception(self):
        """Test that DistanceScaleException inherits from Exception."""
        assert issubclass(DistanceScaleException, Exception)

    def test_can_be_raised(self):
        """Test that DistanceScaleException can be raised."""
        with pytest.raises(DistanceScaleException):
            raise DistanceScaleException("Scale mismatch")

    def test_message_preserved(self):
        """Test that the exception message is preserved."""
        message = "Distance scale 0.05 does not match classifier scale 0.04"
        with pytest.raises(DistanceScaleException, match=message):
            raise DistanceScaleException(message)


class TestExceptionDistinctness:
    """Tests to ensure exceptions are distinct and don't accidentally catch each other."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            PoseHashException,
            PoseIdEmbeddingException,
            MissingBehaviorError,
            FeatureVersionException,
            DistanceScaleException,
        ],
    )
    def test_exception_is_unique_class(self, exception_class):
        """Test that each exception is its own unique class."""
        # All should be subclasses of Exception
        assert issubclass(exception_class, Exception)

        # Should not be subclasses of each other (except through Exception)
        other_exceptions = [
            PoseHashException,
            PoseIdEmbeddingException,
            MissingBehaviorError,
            FeatureVersionException,
            DistanceScaleException,
        ]
        for other in other_exceptions:
            if other is not exception_class:
                assert not issubclass(exception_class, other)
                assert not issubclass(other, exception_class)

    def test_catch_specific_exception_only(self):
        """Test that catching one exception doesn't catch another."""
        try:
            raise PoseHashException("test")
        except PoseIdEmbeddingException:
            pytest.fail("PoseIdEmbeddingException should not catch PoseHashException")
        except PoseHashException:
            pass  # Expected

    @pytest.mark.parametrize(
        "exception_class,message",
        [
            (PoseHashException, "hash error"),
            (PoseIdEmbeddingException, "embedding error"),
            (MissingBehaviorError, "behavior error"),
            (FeatureVersionException, "version error"),
            (DistanceScaleException, "scale error"),
        ],
    )
    def test_exception_with_empty_message(self, exception_class, message):
        """Test that exceptions work with various messages."""
        exc = exception_class(message)
        assert str(exc) == message

    @pytest.mark.parametrize(
        "exception_class",
        [
            PoseHashException,
            PoseIdEmbeddingException,
            MissingBehaviorError,
            FeatureVersionException,
            DistanceScaleException,
        ],
    )
    def test_exception_no_message(self, exception_class):
        """Test that exceptions work without a message."""
        exc = exception_class()
        assert str(exc) == ""
