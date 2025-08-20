# tests/project/test_project_util.py

import pytest

from src.jabs.project.project_utils import to_safe_name


def test_to_safe_name():
    """Test that to_safe_name converts strings to safe file names."""
    assert to_safe_name("My Project") == "My_Project"
    assert to_safe_name("Project@123") == "Project_123"
    assert to_safe_name("File/Name") == "File_Name"
    assert to_safe_name("  Leading and Trailing  ") == "Leading_and_Trailing"


def test_to_safe_name_special_characters():
    """Test that to_safe_name handles special characters correctly."""
    assert to_safe_name("Name_with-dashes") == "Name_with-dashes"
    assert to_safe_name("Name.with.dots") == "Name.with.dots"
    assert to_safe_name("Name@#with$%^special&*chars") == "Name_with_special_chars"


def test_to_safe_name_unicode():
    """Test that to_safe_name handles Unicode characters."""
    assert to_safe_name("Привет") == "Привет"
    assert to_safe_name("你好") == "你好"
    assert to_safe_name("こんにちは") == "こんにちは"


def test_to_safe_name_consecutive_underscores():
    """Test that to_safe_name reduces consecutive underscores to a single underscore."""
    assert to_safe_name("Name__with___underscores") == "Name_with_underscores"
    assert to_safe_name("Multiple   Spaces") == "Multiple_Spaces"
    assert to_safe_name("Special!!@@##$$") == "Special"


def test_to_safe_name_empty_behavior():
    """Test that to_safe_name raises ValueError for empty or invalid behavior names."""
    with pytest.raises(ValueError, match="Behavior name is empty after sanitization."):
        to_safe_name("")
    with pytest.raises(ValueError, match="Behavior name is empty after sanitization."):
        to_safe_name("!@#$%^&*()")
