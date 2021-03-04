import abc

import tensorflow as tf
from tensorflow_addons.utils import types
from typeguard import typechecked


class GradientAccumulatorWrapper(tf.keras.optimizers.Optimizer, metaclass=abc.ABCMeta):
    @typechecked
    def __init__(
            self,
            optimizer: types.Optimizer,
            aggregation_steps: int = 1,
            name: str = "GradientAccumulator",
            **kwargs
    ):
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        if not isinstance(aggregation_steps, int):
            raise TypeError("aggregation_steps must be of int type")

        self._optimizer = optimizer
        self._set_hyper("aggregation_steps", float(aggregation_steps))

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "gradient_accumulator")  # Slot for accumulating gradients over steps

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def apply_gradients(self, grads_and_vars, name=None):
        aggregation_steps = self._get_hyper("aggregation_steps")

        if self.iterations % int(aggregation_steps) == 0:
            for grad, var in grads_and_vars:
                slot = self.get_slot(var, "gradient_accumulator")
                slot.assign(slot / aggregation_steps)  # Take mean over accumulated gradient

            acc_grad_and_vars = [(self.get_slot(var, "gradient_accumulator"), var) for grad, var in grads_and_vars]
            result = super().apply_gradients(acc_grad_and_vars, name)

            for grad, var in grads_and_vars:
                self.get_slot(var, "gradient_accumulator").assign(grad)  # reset gradient accumulator

            return result
        else:
            for grad, var in grads_and_vars:
                self.get_slot(var, "gradient_accumulator").assign_add(grad)

            return tf.no_op()  # update weights evert aggregation_steps

    def _resource_apply_dense(self, grad, handle, apply_state):
        return self._optimizer._resource_apply_dense(grad, handle, apply_state)

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        return self._optimizer._resource_apply_sparse(grad, handle, indices, apply_state)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices, **kwargs):
        return self._optimizer._resource_apply_sparse_duplicate_indices(grad, handle, indices, **kwargs)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects,
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    def set_weights(self, weights):
        return self._optimizer.set_weights(weights)

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)
