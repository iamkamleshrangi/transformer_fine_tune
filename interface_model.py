"Interfaces for various Transformer Model"
from abc import ABCMeta, abstractmethod

class IModel(metaclass=ABCMeta):
    "Interface for Transformer Model"
    @staticmethod
    @abstractmethod
    def load():
        "NLP model loading function"

    @staticmethod
    @abstractmethod
    def train_model():
        "An abstract interface method"
