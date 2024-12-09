import json
from ast import literal_eval
import unittest 

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from src.jabs.classifier import HYPERPARAMETER_PATH
from src.jabs.classifier.classifier import load_hyperparameters, ClassifierType


# Hyperparameter Tests
class TestHyperparameters(unittest.TestCase):
    """
    This test will attempt to build classifiers with a json file.
    """
    
    _hyper_file = HYPERPARAMETER_PATH

    @classmethod
    def setUpClass(cls) -> None:
        print("hyper-parameter file location:", cls._hyper_file)

        with open(cls._hyper_file, "rb") as j:
            data = json.loads(j.read())

        parameters = data["parameters"]

        for classifier in parameters:
            for key in parameters[classifier]:
                try:
                    parameters[classifier][key] = literal_eval(parameters[classifier][key])
                except Exception as e:
                    continue # print(e, classifier, key)
        
        cls.parameters = parameters

    def test_build_classifiers(self):
        
        RandomForestClassifier(**self.parameters["random_forest"])
        GradientBoostingClassifier(**self.parameters["gradient_boost"])
        RandomForestClassifier(**self.parameters["random_forest"])
        assert True
    
    def test_load_hyperparameters(self):

        test_classifier = ClassifierType.RANDOM_FOREST
        classifier_hyperparameters = load_hyperparameters()
        self.assertIn("criterion", classifier_hyperparameters[test_classifier])
    


