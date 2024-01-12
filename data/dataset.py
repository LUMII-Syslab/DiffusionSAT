from abc import abstractmethod, ABCMeta

import tensorflow as tf


class Dataset(metaclass=ABCMeta):
    """ Base dataset that other datasets must implement to be compliant
    with training framework.
    """

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
    def args_for_train_step(self, step_data) -> dict:
        pass

    @abstractmethod
    def metrics(self, initial=False) -> list:
        pass

class SatInstances(metclass=ABCMeta):
    """ Base dataset for generating SAT instances.
    """
    
    @abstractmethod
    def train_generator(self) -> tuple:
        """ Generator function (instead of return use yield), that returns single instance to be writen in DIMACS file.
        This generator should be finite (in the size of dataset)
        :return: tuple(variable_count: int, clauses: list of tuples)
        """
        pass

    @abstractmethod
    def test_generator(self) -> tuple:
        """ Generator function (instead of return use yield), that returns single instance to be writen in DIMACS file.
        This generator should be finite (in the size of dataset)
        :return: tuple(variable_count: int, clauses: list of tuples)
        """
        pass

class SatSpecifics(metclass=ABCMeta):
    
    @abstractmethod
    def prepare_dataset(self, batched_dataset: tf.data.Dataset):
        """ Prepare task specifics for dataset.
        :param dataset: tf.data.Dataset with attributes 
                        clauses, solutions, batched_clauses, adj_indices_pos, adj_indices_neg, variable_count, clauses_in_formula, cells_in_formula
        :return: tf.data.Dataset
        """
        pass
    
    @abstractmethod
    def args_for_train_step(self, step_data) -> dict:
        """ Converts the given batch (from previously prepared datased via prepare_dataset) 
            to a dictionary of arguments to be passed to the neural network train_step method.
        """
        pass
    
    
    @abstractmethod
    def metrics(self, initial=False) -> list:
        pass