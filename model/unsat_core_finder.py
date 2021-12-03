import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.normalization import PairNorm
from loss.sat import softplus_loss, softplus_mixed_loss, unsat_cnf_loss, unsat_cnf_clauses_loss
from model.mlp import MLP
from utils.parameters_log import *
from utils.sat import is_batch_sat


class SATSolver(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=64, train_rounds=16, test_rounds=16,
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

    @staticmethod
    def mask_variables_graph(adj_matrix, variables_graph, clauses_mask_softplus):
        masked_adj_matrix = adj_matrix * clauses_mask_softplus
        var_sum = tf.sparse.reduce_sum(masked_adj_matrix, axis=1)
        v, not_v = tf.split(var_sum, 2)
        variables_mask = 1 - tf.exp(-(v + not_v))
        return variables_graph * variables_mask

    def call(self, adj_matrix, clauses_mask_sigmoid=None, clauses_mask_softplus=None, clauses_graph=None,
             variables_graph=None, training=None, mask=None):

        shape = tf.shape(adj_matrix)
        n_vars = shape[0] // 2
        n_clauses = shape[1]

        adj_matrix = adj_matrix * clauses_mask_sigmoid
        clauses_graph = clauses_graph * clauses_mask_sigmoid
        variables_graph = self.mask_variables_graph(adj_matrix, variables_graph, clauses_mask_softplus)

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
            last_clauses = softplus_loss(last_logits, tf.sparse.transpose(adj_matrix))
            tf.summary.histogram("clauses", last_clauses)
            last_layer_loss = tf.reduce_sum(softplus_mixed_loss(last_logits, tf.sparse.transpose(adj_matrix)))
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

        clauses_in_graph = tf.sparse.reduce_sum(clauses_graph, axis=-1, keepdims=True)

        variables_graph_norm = variables_graph * tf.math.reciprocal_no_nan(
            tf.sparse.reduce_sum(variables_graph, axis=-1, keepdims=True))
        clauses_graph_norm = clauses_graph * tf.math.reciprocal_no_nan(clauses_in_graph)

        for step in tf.range(rounds):
            # make a query for solution, get its value and gradient
            with tf.GradientTape() as grad_tape:
                # add some randomness to avoid zero collapse in normalization
                v1 = tf.concat([variables, tf.random.normal([n_vars, 4])], axis=-1)
                query = self.variables_query(v1)
                clauses_loss = softplus_loss(query, cl_adj_matrix)
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
            per_clause_loss = softplus_mixed_loss(logits, cl_adj_matrix)
            per_graph_loss = tf.sparse.sparse_dense_matmul(clauses_graph, per_clause_loss)
            per_graph_loss = tf.sqrt(per_graph_loss + 1e-6) - tf.sqrt(1e-6)

            step_losses = step_losses.write(step, per_graph_loss)

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


def sample_logistic(shape, eps=1e-5):
    minval = eps
    maxval = 1 - eps
    sample = (minval - maxval) * tf.random.uniform(shape) + maxval
    return tf.math.log(sample / (1 - sample))


class UNSATMinimizer(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=64, train_rounds=16, test_rounds=16,
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
        step_outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)

        cl_adj_matrix = tf.sparse.transpose(adj_matrix)
        shape = tf.shape(adj_matrix)
        n_clauses = shape[1]
        n_vars = shape[0] // 2

        last_logits = tf.zeros([n_vars, self.logit_maps])
        lit_degree = tf.reshape(tf.sparse.reduce_sum(adj_matrix, axis=1), [n_vars * 2, 1])
        degree_weight = tf.math.rsqrt(tf.maximum(lit_degree, 1))
        var_degree_weight = 4 * tf.math.rsqrt(tf.maximum(lit_degree[:n_vars, :] + lit_degree[n_vars:, :], 1))

        variables_graph_norm = variables_graph * tf.math.reciprocal_no_nan(
            tf.sparse.reduce_sum(variables_graph, axis=-1, keepdims=True))
        clauses_graph_norm = clauses_graph * tf.math.reciprocal_no_nan(
            tf.sparse.reduce_sum(clauses_graph, axis=-1, keepdims=True))

        tf.debugging.check_numerics(variables_graph_norm.values, "variables_graph_norm")
        tf.debugging.check_numerics(clauses_graph_norm.values, "clauses_graph_norm")

        for step in tf.range(rounds):
            # make a query for solution, get its value and gradient
            with tf.GradientTape() as grad_tape:
                # add some randomness to avoid zero collapse in normalization
                tf.debugging.check_numerics(variables, "variables2")

                v1 = tf.concat([variables, tf.random.normal([n_vars, 4])], axis=-1)
                tf.debugging.check_numerics(v1, "v1")
                query = self.variables_query(v1)
                tf.debugging.check_numerics(query, "query")
                clauses_loss = unsat_cnf_clauses_loss(query, cl_adj_matrix)
                step_loss = tf.reduce_sum(clauses_loss)

                tf.debugging.check_numerics(clauses_loss, "clauses_loss")

            variables_grad = tf.convert_to_tensor(grad_tape.gradient(step_loss, query))
            variables_grad = variables_grad * var_degree_weight
            tf.debugging.check_numerics(variables_grad, "variables_grad")
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

            tf.debugging.check_numerics(variables, "variables")

            # calculate logits and loss
            logits = self.clauses_output(clauses)
            step_outputs = step_outputs.write(step, logits)

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8
            clauses = tf.stop_gradient(clauses) * 0.2 + clauses * 0.8

        if training:
            last_logits = tf.reduce_mean(step_outputs.stack(), axis=0)
        else:
            last_logits = step_outputs.stack()[-1, :, :]

        if training:
            last_logits += sample_logistic(tf.shape(last_logits))

        clauses_mask_sigmoid = tf.sigmoid(last_logits)
        clauses_mask_softplus = tf.nn.softplus(last_logits)

        return tf.squeeze(clauses_mask_sigmoid, axis=-1), tf.squeeze(clauses_mask_softplus, axis=-1)


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
            clauses_mask_sigmoid, clauses_mask_softplus = self.unsat_minimizer(adj_matrix, clauses_graph,
                                                                               variables_graph,
                                                                               training=True)
            tf.debugging.check_numerics(clauses_mask_sigmoid, "clauses_mask_sigmoid")
            tf.debugging.check_numerics(clauses_mask_softplus, "clauses_mask_softplus")

            core_logits, core_loss, step = self.solver(adj_matrix, clauses_mask_sigmoid, clauses_mask_softplus,
                                                       clauses_graph, variables_graph, training=True)

            tf.debugging.check_numerics(core_logits, "core_logits")

            # Sub-formulas of UNSAT core should be satisfiable
            subcore_total_loss = 0
            for i in range(4):
                clause_mask = self.generate_clause_mask(clauses_graph, clauses_mask_sigmoid)
                subformula_mask_sigmoid = clauses_mask_sigmoid * clause_mask
                subformula_mask_softplus = clauses_mask_softplus * clause_mask
                subcore_logits, subcore_loss, step = self.solver(adj_matrix, subformula_mask_sigmoid,
                                                                 subformula_mask_softplus, clauses_graph,
                                                                 variables_graph, training=True)

                subcore_total_loss += subcore_loss

            # We want to find minimum (or minimal) UNSAT core
            count_loss = tf.sparse.reduce_sum(clauses_graph * clauses_mask_sigmoid, axis=1)

            masked_adj_matrix = tf.sparse.transpose(adj_matrix * clauses_mask_sigmoid)
            masked_clauses_graph = clauses_graph * clauses_mask_sigmoid

            # Loss for both networks
            minimizer_loss = unsat_cnf_loss(core_logits, masked_adj_matrix, masked_clauses_graph)
            minimizer_loss = tf.reduce_mean(minimizer_loss + subcore_total_loss)
            solver_loss = tf.reduce_mean(core_loss + subcore_total_loss)

        # Update weights of each network
        minimizer_gradients = minimizer_tape.gradient(minimizer_loss, self.unsat_minimizer.trainable_variables)
        solver_gradients = solver_tape.gradient(solver_loss, self.solver.trainable_variables)

        self.unsat_minimizer.optimizer.apply_gradients(
            zip(minimizer_gradients, self.unsat_minimizer.trainable_variables))
        self.solver.optimizer.apply_gradients(zip(solver_gradients, self.solver.trainable_variables))

        # Additional metrics
        discr_level = self.discretization_level(clauses_mask_sigmoid)
        count_set_clauses = tf.reduce_mean(tf.sparse.reduce_sum(clauses_graph * tf.round(clauses_mask_sigmoid), axis=1))

        return {
            "steps_taken": step,
            "count_loss": count_loss,
            "minimizer_loss": minimizer_loss,
            "solver_loss": solver_loss,
            "minimizer_gradients": minimizer_gradients,
            "solver_gradients": solver_gradients,
            "discretization_level": discr_level,
            "set_clauses": count_set_clauses
        }

    @staticmethod
    def discretization_level(clauses_mask):
        discretization_level = tf.abs(tf.round(clauses_mask) - clauses_mask)
        discretization_level = tf.reduce_mean(discretization_level)
        return discretization_level

    @staticmethod
    def generate_clause_mask(clauses_graph, clauses_mask):
        """ Generates binary mask that masks single non-zero clause in each graph of the batch """
        int_mask = tf.round(clauses_mask)
        noise = tf.random.uniform(tf.shape(int_mask)) * int_mask
        max_val = tf.sparse.reduce_max(clauses_graph * noise, axis=1)
        max_val = tf.sparse.reduce_max(tf.sparse.transpose(clauses_graph) * max_val, axis=1)
        mask = -(noise - max_val) * int_mask
        mask = tf.not_equal(mask, 0)
        return tf.cast(mask, tf.float32)

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def predict_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        clauses_mask, _ = self.unsat_minimizer(adj_matrix, clauses_graph, variables_graph, training=False)

        return {
            "steps_taken": tf.zeros([1]),
            "loss": tf.zeros([1]),
            "unsat_core": clauses_mask
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__}
