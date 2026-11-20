"""User-facing summaries of unusable videos."""

from jabs.project.unusable_report import (
    UnusableNotice,
    skipped_training_notice,
    unusable_videos_notice,
)


def test_no_unusable_videos_no_notice() -> None:
    """Nothing to report gives no notice."""
    assert unusable_videos_notice({}, set()) is None
    assert skipped_training_notice([], {}) is None


def test_unusable_videos_notice_lists_videos_reasons_and_labels() -> None:
    """Every unusable video is listed with its reason; labeled ones are flagged."""
    notice = unusable_videos_notice(
        {"b.avi": "refused", "a.avi": "cannot read"}, annotated={"b.avi", "c.avi"}
    )

    assert isinstance(notice, UnusableNotice)
    assert "2 videos cannot be used" in notice.message
    assert "1 video with labels will not be trained on" in notice.message
    assert notice.details.splitlines() == ["a.avi: cannot read", "b.avi (labeled): refused"]
    assert notice.as_text() == f"{notice.message}\n{notice.details}"


def test_skipped_training_notice_names_the_videos() -> None:
    """A training run that skipped labeled videos names them and why."""
    notice = skipped_training_notice(["b.avi"], {"b.avi": "refused"})

    assert notice is not None
    assert notice.message.startswith("1 video with labels for this training run was left out")
    assert notice.details == "b.avi: refused"
