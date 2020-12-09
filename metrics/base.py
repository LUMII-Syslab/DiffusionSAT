from abc import ABCMeta, abstractmethod


class Metric(metaclass=ABCMeta):

    @abstractmethod
    def update_state(self, model_output, step_data):
        pass

    @abstractmethod
    def log_in_tensorboard(self, step: int = None, reset_state=True):
        pass

    @abstractmethod
    def log_in_stdout(self, step: int = None, reset_state=True):
        pass

    @abstractmethod
    def reset_state(self):
        pass
