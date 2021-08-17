import joblib
from pathlib import Path

from src.classifier.classifier import Classifier

_VERSION = 2


class ClassifierSerializer:
    """
    This class is for serializing a Classifier object along with some other
    attributes not contained in the Classifier object itself.

    Currently this is primarily used by the classify.py script to implement
    the 'train' command.
    """

    def __init__(self, classifier: Classifier, window_size: int,
                 behavior_name: str, use_social: bool):
        self._version = _VERSION
        self._classifier = classifier
        self._window_size = window_size
        self._behavior_name = behavior_name
        self._use_social = use_social

    @property
    def classifier(self):
        return self._classifier

    @property
    def window_size(self):
        return self._window_size

    @property
    def behavior_name(self):
        return self._behavior_name

    @property
    def version(self):
        return self._version

    def save(self, path: Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        c = joblib.load(path)
        if not isinstance(c, cls):
            raise ValueError(
                f"{path} is not instance of ClassifierSerializer")

        if c.version != _VERSION:
            raise ValueError(f"Error deserializing classifier. "
                             f"File version {c.version}, expected {_VERSION}.")

        return c.classifier, c.behavior_name, c.window_size
