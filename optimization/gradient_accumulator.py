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

    def set_weights(self, weights):
        self._optimizer.set_weights(weights)

    def apply_gradients(self, grads_and_vars, name=None):
        op = super(GradientAccumulatorWrapper, self).apply_gradients(grads_and_vars, name)

        aggregation_steps = self._get_hyper("aggregation_steps")
        if (self.iterations - 1) % int(aggregation_steps) == 0:
            self._optimizer._iterations.assign_add(1)

        return op

    def _resource_apply_dense(self, grad, handle, apply_state):
        aggregation_steps = self._get_hyper("aggregation_steps")

        if self.iterations % int(aggregation_steps) == 0:
            slot = self.get_slot(handle, "gradient_accumulator")
            slot.assign(slot / aggregation_steps)  # Take mean over accumulated gradient
            result = self._optimizer._resource_apply_dense(slot, handle)
            slot.assign(grad)  # reset gradient accumulator
            return result
        else:
            self.get_slot(handle, "gradient_accumulator").assign_add(grad)
            return tf.no_op()  # update weights evert aggregation_steps

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        aggregation_steps = self._get_hyper("aggregation_steps")

        if self.iterations % int(aggregation_steps) == 0:
            slot = self.get_slot(handle, "gradient_accumulator")
            slot.assign(slot / aggregation_steps)  # Take mean over accumulated gradient
            result = self._optimizer._resource_apply_sparse(slot, handle, indices)
            slot.assign(grad)  # reset gradient accumulator
            return result
        else:
            self.get_slot(handle, "gradient_accumulator").assign_add(grad)
            return tf.no_op()  # update weights evert aggregation_steps

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}
