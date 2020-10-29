from abc import abstractmethod, ABCMeta
import tensorflow as tf


class Dataset(metaclass=ABCMeta):

    @abstractmethod
    def train_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def validation_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def test_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def loss_fn(self, predictions, labels=None):
        pass

    @abstractmethod
    def accuracy_fn(self, predictions, labels=None):
        pass
