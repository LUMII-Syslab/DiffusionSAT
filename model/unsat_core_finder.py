import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.normalization import PairNorm
from loss.sat import softplus_loss_adj, softplus_mixed_loss_adj
from model.mlp import MLP
from utils.parameters_log import *
from utils.sat import is_batch_sat


class SATSolver(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=64, train_rounds=16, test_rounds=8,
                 query_maps=64, supervised=False, **kwargs):
        super().__init__(**kwargs, name="SATSolver")
        self.supervised = supervised
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds
        self.optimizer = optimizer
        self.logit_maps = 1

        self.variables_norm = PairNorm(subtract_mean=True)
        self.clauses_norm = PairNorm(subtract_mean=True)

        self.update_gate = MLP(3, feature_maps * 2, feature_maps, name="update_gate")
        self.variables_output = MLP(2, feature_maps, self.logit_maps, name="variables_output")
        self.variables_query = MLP(2, query_maps, query_maps, name="variables_query")
        self.clause_mlp = MLP(2, feature_maps * 2, feature_maps + 1 * query_maps, name="clause_update")

        self.feature_maps = feature_maps
        self.query_maps = query_maps

    def call(self, adj_matrix, clauses_graph=None, variables_graph=None, training=None, mask=None):
        shape = tf.shape(adj_matrix)
        n_vars = shape[0] // 2
        n_clauses = shape[1]

        variables = tf.ones([n_vars, self.feature_maps])
        clause_state = tf.ones([n_clauses, self.feature_maps])

        rounds = self.train_rounds if training else self.test_rounds

        last_logits, step, unsupervised_loss, clause_state, variables = self.loop(adj_matrix,
                                                                                  clause_state,
                                                                                  clauses_graph,
                                                                                  rounds,
                                                                                  training,
                                                                                  variables,
                                                                                  variables_graph)

        if training:
            last_clauses = softplus_loss_adj(last_logits, adj_matrix=tf.sparse.transpose(adj_matrix))
            tf.summary.histogram("clauses", last_clauses)
            last_layer_loss = tf.reduce_sum(
                softplus_mixed_loss_adj(last_logits, adj_matrix=tf.sparse.transpose(adj_matrix)))
            tf.summary.histogram("logits", last_logits)
            tf.summary.scalar("last_layer_loss", last_layer_loss)

            tf.summary.scalar("steps_taken", step)

        return last_logits, unsupervised_loss, step

    def loop(self, adj_matrix, clause_state, clauses_graph, rounds, training, variables, variables_graph):
        step_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        cl_adj_matrix = tf.sparse.transpose(adj_matrix)
        shape = tf.shape(adj_matrix)
        n_clauses = shape[1]
        n_vars = shape[0] // 2

        last_logits = tf.zeros([n_vars, self.logit_maps])
        lit_degree = tf.reshape(tf.sparse.reduce_sum(adj_matrix, axis=1), [n_vars * 2, 1])
        degree_weight = tf.math.rsqrt(tf.maximum(lit_degree, 1))
        var_degree_weight = 4 * tf.math.rsqrt(tf.maximum(lit_degree[:n_vars, :] + lit_degree[n_vars:, :], 1))

        variables_graph_norm = variables_graph / tf.sparse.reduce_sum(variables_graph, axis=-1, keepdims=True)
        clauses_graph_norm = clauses_graph / tf.sparse.reduce_sum(clauses_graph, axis=-1, keepdims=True)

        for step in tf.range(rounds):
            # make a query for solution, get its value and gradient
            with tf.GradientTape() as grad_tape:
                # add some randomness to avoid zero collapse in normalization
                v1 = tf.concat([variables, tf.random.normal([n_vars, 4])], axis=-1)
                query = self.variables_query(v1)
                clauses_loss = softplus_loss_adj(query, cl_adj_matrix)
                step_loss = tf.reduce_sum(clauses_loss)

            variables_grad = tf.convert_to_tensor(grad_tape.gradient(step_loss, query))
            variables_grad = variables_grad * var_degree_weight
            # calculate new clause state
            clauses_loss *= 4

            clause_unit = tf.concat([clause_state, clauses_loss], axis=-1)
            clause_data = self.clause_mlp(clause_unit, training=training)

            variables_loss_all = clause_data[:, 0:self.query_maps]
            new_clause_value = clause_data[:, self.query_maps:]
            new_clause_value = self.clauses_norm(new_clause_value, clauses_graph_norm, training=training) * 0.25
            clause_state = new_clause_value + 0.1 * clause_state

            # Aggregate loss over edges
            variables_loss = tf.sparse.sparse_dense_matmul(adj_matrix, variables_loss_all)
            variables_loss *= degree_weight
            variables_loss_pos, variables_loss_neg = tf.split(variables_loss, 2, axis=0)

            # calculate new variable state
            unit = tf.concat([variables_grad, variables, variables_loss_pos, variables_loss_neg], axis=-1)
            new_variables = self.update_gate(unit)
            new_variables = self.variables_norm(new_variables, variables_graph_norm, training=training) * 0.25
            variables = new_variables + 0.1 * variables

            # calculate logits and loss
            logits = self.variables_output(variables)
            per_clause_loss = softplus_mixed_loss_adj(logits, cl_adj_matrix)

            logit_loss = per_clause_loss
            step_losses = step_losses.write(step, logit_loss)

            is_sat = is_batch_sat(logits, cl_adj_matrix)
            last_logits = logits

            if is_sat == 1:
                break

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8
            clause_state = tf.stop_gradient(clause_state) * 0.2 + clause_state * 0.8

        unsupervised_loss = tf.reduce_sum(step_losses.stack(), axis=0) / tf.cast(rounds, tf.float32)
        return last_logits, step, unsupervised_loss, clause_state, variables


class UNSATMinimizer(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=64, train_rounds=8, test_rounds=8,
                 query_maps=64, supervised=False, **kwargs):
        super().__init__(**kwargs, name="UNSATMinimizer")
        self.supervised = supervised
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds
        self.optimizer = optimizer
        self.logit_maps = 1

        self.variables_norm = PairNorm(subtract_mean=True)
        self.clauses_norm = PairNorm(subtract_mean=True)

        self.update_gate = MLP(3, feature_maps * 2, feature_maps, name="update_gate")
        self.clauses_output = MLP(2, feature_maps, self.logit_maps, name="clauses_mask_output")
        self.variables_query = MLP(2, query_maps, query_maps, name="variables_query")
        self.clause_mlp = MLP(2, feature_maps * 2, feature_maps + 1 * query_maps, name="clause_update")

        self.feature_maps = feature_maps
        self.query_maps = query_maps

    def call(self, adj_matrix, clauses_graph=None, variables_graph=None, training=None, mask=None):
        shape = tf.shape(adj_matrix)
        n_vars = shape[0] // 2
        n_clauses = shape[1]

        variables = tf.ones([n_vars, self.feature_maps])
        clauses = tf.ones([n_clauses, self.feature_maps])

        rounds = self.train_rounds if training else self.test_rounds

        clauses_mask = self.loop(adj_matrix,
                                 clauses,
                                 clauses_graph,
                                 rounds,
                                 training,
                                 variables,
                                 variables_graph)

        return clauses_mask

    def loop(self, adj_matrix, clauses, clauses_graph, rounds, training, variables, variables_graph):
        cl_adj_matrix = tf.sparse.transpose(adj_matrix)
        shape = tf.shape(adj_matrix)
        n_clauses = shape[1]
        n_vars = shape[0] // 2

        last_logits = tf.zeros([n_vars, self.logit_maps])
        lit_degree = tf.reshape(tf.sparse.reduce_sum(adj_matrix, axis=1), [n_vars * 2, 1])
        degree_weight = tf.math.rsqrt(tf.maximum(lit_degree, 1))
        var_degree_weight = 4 * tf.math.rsqrt(tf.maximum(lit_degree[:n_vars, :] + lit_degree[n_vars:, :], 1))

        variables_graph_norm = variables_graph / tf.sparse.reduce_sum(variables_graph, axis=-1, keepdims=True)
        clauses_graph_norm = clauses_graph / tf.sparse.reduce_sum(clauses_graph, axis=-1, keepdims=True)

        for _ in tf.range(rounds):
            # make a query for solution, get its value and gradient
            with tf.GradientTape() as grad_tape:
                # add some randomness to avoid zero collapse in normalization
                v1 = tf.concat([variables, tf.random.normal([n_vars, 4])], axis=-1)
                query = self.variables_query(v1)
                clauses_loss = softplus_loss_adj(query, cl_adj_matrix)
                step_loss = tf.reduce_sum(clauses_loss)

            variables_grad = tf.convert_to_tensor(grad_tape.gradient(step_loss, query))
            variables_grad = variables_grad * var_degree_weight
            # calculate new clause state
            clauses_loss *= 4

            clause_unit = tf.concat([clauses, clauses_loss], axis=-1)
            clause_data = self.clause_mlp(clause_unit, training=training)

            variables_loss_all = clause_data[:, 0:self.query_maps]
            new_clause_value = clause_data[:, self.query_maps:]
            new_clause_value = self.clauses_norm(new_clause_value, clauses_graph_norm, training=training) * 0.25
            clauses = new_clause_value + 0.1 * clauses

            # Aggregate loss over edges
            variables_loss = tf.sparse.sparse_dense_matmul(adj_matrix, variables_loss_all)
            variables_loss *= degree_weight
            variables_loss_pos, variables_loss_neg = tf.split(variables_loss, 2, axis=0)

            # calculate new variable state
            unit = tf.concat([variables_grad, variables, variables_loss_pos, variables_loss_neg], axis=-1)
            new_variables = self.update_gate(unit)
            new_variables = self.variables_norm(new_variables, variables_graph_norm, training=training) * 0.25
            variables = new_variables + 0.1 * variables

            # calculate logits and loss
            logits = self.clauses_output(clauses)
            last_logits = logits

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8
            clauses = tf.stop_gradient(clauses) * 0.2 + clauses * 0.

        clauses_mask = tf.sigmoid(last_logits + tf.random.normal(tf.shape(last_logits), 0, 1))  # TODO: Better noise

        return tf.squeeze(clauses_mask, axis=-1)


class CoreFinder(Model):

    def __init__(self, optimizer_minimizer, optimizer_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unsat_minimizer = UNSATMinimizer(optimizer_minimizer)
        self.solver = SATSolver(optimizer_solver)

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),  # TODO: Also SAT instances
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def train_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        with tf.GradientTape() as minimizer_tape, tf.GradientTape() as solver_tape:
            clauses_mask = self.unsat_minimizer(adj_matrix, clauses_graph, variables_graph, training=True)
            _, solver_loss, step = self.solver(adj_matrix * clauses_mask, clauses_graph, variables_graph, training=True)

            count_loss = tf.sparse.reduce_sum(clauses_graph * clauses_mask, axis=1)
            minimizer_loss = -solver_loss + count_loss * 0.0001  # TODO: solver_loss should be per graph
            minimizer_loss = tf.reduce_mean(minimizer_loss)
            solver_loss = tf.reduce_mean(solver_loss)

        discretization_level = tf.abs(tf.round(clauses_mask) - clauses_mask)
        discretization_level = tf.reduce_max(discretization_level)

        count_set_clauses = tf.reduce_sum(tf.round(clauses_mask))

        maker_vars = self.unsat_minimizer.trainable_variables
        solver_vars = self.solver.trainable_variables

        minimizer_gradients = minimizer_tape.gradient(minimizer_loss, maker_vars)
        solver_gradients = solver_tape.gradient(solver_loss, solver_vars)

        self.unsat_minimizer.optimizer.apply_gradients(zip(minimizer_gradients, maker_vars))
        self.solver.optimizer.apply_gradients(zip(solver_gradients, solver_vars))

        return {
            "steps_taken": step,
            "count_loss": count_loss,
            "minimizer_loss": minimizer_loss,
            "solver_loss": solver_loss,
            "minimizer_gradients": minimizer_gradients,
            "solver_gradients": solver_gradients,
            "discretization_level": discretization_level,
            "set_clauses": count_set_clauses
        }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def predict_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        clauses_mask, count_loss = self.unsat_minimizer(adj_matrix, clauses_graph, variables_graph, training=True)
        sat_prediction, solver_loss, step = self.solver(adj_matrix * clauses_mask, clauses_graph * clauses_mask,
                                                        variables_graph, training=True)

        return {
            "steps_taken": step,
            "loss": solver_loss,
            "sat_predictions": tf.squeeze(sat_prediction, axis=-1),
            "unsat_core": clauses_mask
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__}
