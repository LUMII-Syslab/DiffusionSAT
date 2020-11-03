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
    def loss(self, predictions, step_data) -> tf.Tensor:
        pass

    @abstractmethod
    def accuracy(self, predictions, step_data) -> tf.Tensor:
        pass

    @abstractmethod
    def filter_model_inputs(self, step_data) -> dict:
        pass
