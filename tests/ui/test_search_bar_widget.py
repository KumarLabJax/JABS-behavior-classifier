import pytest

from jabs.behavior_search import SearchHit

# Try to import the functions; if Qt/EGL is not available (e.g., headless CI),
# mark all tests in this module to be skipped
try:
    from jabs.ui.search_bar_widget import (
        _binary_search_with_comparator,
        _compare_file_frame_vs_search_hit,
    )

    SKIP_UI_TESTS = False
except ImportError as e:
    # Qt/PySide6 not available (likely headless environment)
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def test_compare_file_frame_vs_search_hit_before():
    """Test that a frame before the hit returns -1."""
    hit = SearchHit(file="video1", identity="0", behavior=None, start_frame=10, end_frame=20)
    key = {"video_name": "video1", "frame_index": 5}
    assert _compare_file_frame_vs_search_hit(key, hit) == -1


def test_compare_file_frame_vs_search_hit_after():
    """Test that a frame after the hit returns 1."""
    hit = SearchHit(file="video1", identity="0", behavior=None, start_frame=10, end_frame=20)
    key = {"video_name": "video1", "frame_index": 25}
    assert _compare_file_frame_vs_search_hit(key, hit) == 1


def test_compare_file_frame_vs_search_hit_overlap():
    """Test that a frame within the hit returns 0."""
    hit = SearchHit(file="video1", identity="0", behavior=None, start_frame=10, end_frame=20)
    key = {"video_name": "video1", "frame_index": 15}
    assert _compare_file_frame_vs_search_hit(key, hit) == 0


def test_compare_file_frame_vs_search_hit_different_video():
    """Test that a frame from a different video returns -1 or 1 as appropriate."""
    hit = SearchHit(file="video2", identity="0", behavior=None, start_frame=10, end_frame=20)
    key = {"video_name": "video1", "frame_index": 15}
    assert _compare_file_frame_vs_search_hit(key, hit) == -1
    key = {"video_name": "video3", "frame_index": 15}
    assert _compare_file_frame_vs_search_hit(key, hit) == 1


def test_binary_search_with_comparator_found():
    """Test that binary search finds an existing element."""
    arr = [10, 20, 30, 40]

    def cmp(key, elem):
        return key - elem

    found, idx = _binary_search_with_comparator(arr, 20, cmp)
    assert found == 20
    assert idx == 1


def test_binary_search_with_comparator_not_found():
    """Test that binary search returns correct index for missing element."""
    arr = [10, 20, 30, 40]

    def cmp(key, elem):
        return key - elem

    found, idx = _binary_search_with_comparator(arr, 25, cmp)
    assert found is None
    assert idx == 2


def test_binary_search_with_comparator_empty():
    """Test that binary search on empty list returns index 0."""
    arr = []

    def cmp(key, elem):
        return key - elem

    found, idx = _binary_search_with_comparator(arr, 10, cmp)
    assert found is None
    assert idx == 0
