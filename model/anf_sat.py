import optuna
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.normalization import PairNorm, LayerNormalization
from loss.anf import anf_loss, anf_value_cplx, anf_value_cplx_adj
from model.mlp import MLP
from utils.parameters_log import *


class ANFSAT(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=128, msg_layers=3,
                 vote_layers=3, train_rounds=16, test_rounds=32,
                 query_maps=128, supervised=True, trial: optuna.Trial = None, **kwargs):
        super().__init__(**kwargs, name="QuerySAT")
        self.supervised = supervised
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds
        self.optimizer = optimizer
        self.use_message_passing = True
        self.use_query = True
        self.skip_first_rounds = 0
        self.prediction_tries = 1

        update_layers = trial.suggest_int("variables_update_layers", 2, 4) if trial else msg_layers
        output_layers = trial.suggest_int("output_layers", 2, 4) if trial else vote_layers
        query_layers = trial.suggest_int("query_layers", 2, 4) if trial else vote_layers
        clauses_layers = trial.suggest_int("clauses_update_layers", 2, 4) if trial else msg_layers

        feature_maps = trial.suggest_categorical("feature_maps", [16, 32, 64]) if trial else feature_maps
        query_maps = trial.suggest_categorical("query_maps", [16, 32, 64]) if trial else query_maps

        update_scale = trial.suggest_discrete_uniform("update_scale", 0.2, 2., 0.2) if trial else 2
        output_scale = trial.suggest_discrete_uniform("output_scale", 0.2, 2., 0.2) if trial else 1
        clauses_scale = trial.suggest_discrete_uniform("clauses_scale", 0.2, 2., 0.2) if trial else 2
        query_scale = trial.suggest_discrete_uniform("query_scale", 0.2, 2., 0.2) if trial else 3

        self.variables_norm = PairNorm(subtract_mean=True)
        self.clauses_norm = PairNorm(subtract_mean=True)

        self.update_gate = MLP(update_layers, int(feature_maps * update_scale), feature_maps, name="update_gate", do_layer_norm=False)
        self.variables_output = MLP(output_layers, int(feature_maps * output_scale), 1, name="variables_output", do_layer_norm=False)
        self.variables_query = MLP(query_layers, int(query_maps * query_scale), query_maps, name="variables_query", do_layer_norm=False)
        self.clause_mlp = MLP(clauses_layers, int(feature_maps * clauses_scale), feature_maps + 1 * query_maps, name="clause_update", do_layer_norm=False)
        self.grad_mlp = MLP(clauses_layers, int(feature_maps * clauses_scale), query_maps, name="grad_update", do_layer_norm=False)
        self.ands_mlp = MLP(clauses_layers, int(feature_maps * clauses_scale), query_maps, name="ands_update", do_layer_norm=False)
        self.var2ands_mlp = MLP(clauses_layers, int(feature_maps * clauses_scale), query_maps, name="var2ands_mlp", do_layer_norm=False)
        self.var2clause_mlp = MLP(clauses_layers, int(feature_maps * clauses_scale), query_maps, name="var2clause_mlp", do_layer_norm=False)

        #self.lit_mlp = MLP(msg_layers, query_maps * 4, query_maps * 2, name="lit_query", do_layer_norm=False)

        self.feature_maps = feature_maps
        self.query_maps = query_maps
        self.vote_layers = vote_layers
        self.zero_var = self.add_weight("zero_var", [1,query_maps], initializer=tf.constant_initializer(0.02))

    def zero_state(self, n_units, n_features, stddev=0.25):
        onehot = tf.one_hot(tf.zeros([n_units], dtype=tf.int64), n_features)
        onehot -= 1 / n_features
        return onehot * tf.sqrt(tf.cast(n_features, tf.float32)) * stddev

    def call(self, ands_index1:tf.Tensor,ands_index2:tf.Tensor=None,clauses_adj:tf.SparseTensor=None,n_vars=0, n_clauses=0, training=None, labels=None,clauses_graph=None, variables_graph=None):
        variables = self.zero_state(n_vars, self.feature_maps)
        clause_state = self.zero_state(n_clauses, self.feature_maps)
        rounds = self.train_rounds if training else self.test_rounds

        last_logits, step, unsupervised_loss, supervised_loss, clause_state, variables = self.loop(ands_index1,ands_index2,clauses_adj,
                                                                                                   clause_state,
                                                                                                   labels,
                                                                                                   rounds,
                                                                                                   training,
                                                                                                   variables,
                                                                                                   n_clauses,
                                                                                                   clauses_graph,
                                                                                                   variables_graph)
        if not self.supervised: last_logits=-last_logits # meaning of logits is reversed in unsupervised
        if training:
            tf.summary.histogram("logits", -last_logits)
            #per_clause_loss = anf_loss(-last_logits, ands_index1, ands_index2, clauses_adj, n_clauses)
            #tf.summary.histogram("clauses", per_clause_loss)
            #tf.summary.scalar("last_layer_loss", tf.reduce_sum(tf.square(1-per_clause_loss)))


        return last_logits, unsupervised_loss + supervised_loss, step

    def loop(self, ands_index1:tf.Tensor,ands_index2:tf.Tensor,clauses_adj:tf.SparseTensor, clause_state, labels, rounds, training, variables, n_clauses,clauses_graph, variables_graph):
        step_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        n_vars = tf.shape(variables)[0]
        var_degree = tf.reshape(tf.sparse.reduce_sum(clauses_adj, axis=1), [-1, 1])
        degree_weight = tf.math.rsqrt(tf.maximum(var_degree, 1))
        clause_degree = tf.reshape(tf.sparse.reduce_sum(clauses_adj, axis=0), [-1, 1])
        clause_degree_weight = tf.math.rsqrt(tf.maximum(clause_degree, 1))

        variables_graph_norm = variables_graph / tf.sparse.reduce_sum(variables_graph, axis=-1, keepdims=True)
        clauses_graph_norm = clauses_graph / tf.sparse.reduce_sum(clauses_graph, axis=-1, keepdims=True)

        last_logits = tf.zeros([n_vars, 1])
        v_grad = tf.zeros([n_vars, self.query_maps])
        query_msg = tf.zeros([n_vars, self.query_maps])
        cl_msg = tf.zeros([n_clauses, self.query_maps])
        query_value = tf.zeros([n_vars, self.query_maps])

        for step in tf.range(rounds):
            # make a query for solution, get its value and gradient
            if self.use_query:
                with tf.GradientTape() as grad_tape:
                    # add some randomness to avoid zero collapse in normalization
                    v1 = tf.concat([variables, tf.random.normal([n_vars, 4])], axis=-1)
                    query = self.variables_query(v1)
                    clauses_real, clauses_im, clause_ands1,clause_ands2 = anf_value_cplx_adj(query, ands_index1, ands_index2, clauses_adj)
                    query_value = query
                    query_msg = tf.concat([clauses_real, clauses_im], axis=-1)
                    #step_loss = tf.reduce_sum(1 - clauses_real)
                #variables_grad = tf.convert_to_tensor(grad_tape.gradient(step_loss, query))
                #v_grad = variables_grad

            # calculate new clause state
            #clauses_loss *= 4
            ands1 = tf.gather(variables, ands_index1-1, axis=0) #get variables corresponding to and1
            ands2 = tf.gather(variables, ands_index2-1, axis=0) #get variables corresponding to and2

            if self.use_message_passing:
                ands_data = self.var2ands_mlp(tf.concat([ands1, ands2], axis=-1), training=training)
                var_data = self.var2clause_mlp(variables, training=training)
                var_data = tf.concat([self.zero_var, var_data, ands_data], axis=0)
                clause_messages = tf.sparse.sparse_dense_matmul(clauses_adj, var_data, adjoint_a=True)
                clause_messages *= clause_degree_weight
                cl_msg = clause_messages
                if self.use_query:
                    clause_unit = tf.concat([clause_state, clause_messages, query_msg], axis=-1)
                else:
                    clause_unit = tf.concat([clause_state, clause_messages], axis=-1)
            else:
                clause_unit = tf.concat([clause_state, query_msg], axis=-1)
            clause_data = self.clause_mlp(clause_unit, training=training)

            loss_to_vars = clause_data[:, 0:self.query_maps]
            new_clause_value = clause_data[:, self.query_maps:]
            new_clause_value = self.clauses_norm(new_clause_value, clauses_graph_norm, training=training) * 0.25
            clause_state = new_clause_value + 0.1 * clause_state

            # Aggregate loss over edges
            variables_ands_loss = tf.sparse.sparse_dense_matmul(clauses_adj, loss_to_vars) # sum over vars and and_data
            variables_ands_loss *= degree_weight
            variables_loss = variables_ands_loss[1:n_vars+1,:] #take variable part
            ands_data = tf.concat([ands1, ands2, variables_ands_loss[n_vars+1:,:]], axis=-1) #concat anded variables and clause values corespond to ands
            ands_value = self.ands_mlp(ands_data) #apply mlp to and data

            # add and_value to respective variable data
            variables_loss = tf.tensor_scatter_nd_add(variables_loss, tf.expand_dims(ands_index1 - 1, -1), ands_value)
            variables_loss = tf.tensor_scatter_nd_add(variables_loss, tf.expand_dims(ands_index2 - 1, -1), ands_value)
            #variables_loss *= degree_weight

            # calculate new variable state
            unit = tf.concat([variables, variables_loss], axis=-1)
            new_variables = self.update_gate(unit)
            new_variables = self.variables_norm(new_variables, variables_graph_norm, training=training) * 0.25
            variables = new_variables + 0.1 * variables

            # calculate logits and loss
            logits = self.variables_output(variables)
            if self.supervised:
                if labels is not None:
                    smoothed_labels = 0.5 * 0.1 + tf.expand_dims(tf.cast(labels, tf.float32), -1) * 0.9
                    logit_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=smoothed_labels))
                else:
                    logit_loss = 0.
            else:
                per_clause_loss = anf_loss(logits, ands_index1, ands_index2, clauses_index, n_clauses)
                #per_graph_loss = tf.sparse.sparse_dense_matmul(clauses_graph, per_clause_loss)
                #logit_loss = tf.reduce_sum(tf.sqrt(per_graph_loss + 1e-6))
                logit_loss = tf.reduce_sum(tf.square(1-per_clause_loss))

            step_losses = step_losses.write(step, logit_loss)

            # n_unsat_clauses = unsat_clause_count(logits, clauses)
            # if n_unsat_clauses == 0:

            # is_batch_sat = self.is_batch_sat(logits, cl_adj_matrix)
            # if is_batch_sat == 1:
            #     if not self.supervised:
            #         # now we know the answer, we can use it for supervised training
            #         labels_got = tf.round(tf.sigmoid(logits))
            #         supervised_loss = tf.reduce_mean(
            #             tf.nn.sigmoid_cross_entropy_with_logits(logits=last_logits, labels=labels_got))
            #     last_logits = logits
            #     break
            last_logits = logits

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8
            clause_state = tf.stop_gradient(clause_state) * 0.2 + clause_state * 0.8

        if training:
             tf.summary.histogram("query_msg", query_msg)
             tf.summary.histogram("clause_msg", cl_msg)
             tf.summary.histogram("var_grad", v_grad)
        #     tf.summary.histogram("var_loss_msg", var_loss_msg)
             tf.summary.histogram("query", query_value)

        unsupervised_loss = tf.reduce_sum(step_losses.stack()) / tf.cast(rounds, tf.float32)
        return last_logits, step, unsupervised_loss, 0.0, clause_state, variables

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.int64),
                                  tf.TensorSpec(shape=(), dtype=tf.int64),
                                  tf.TensorSpec(shape=(), dtype=tf.int64),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.SparseTensorSpec(shape = [None, None], dtype = tf.float32),
                                  tf.SparseTensorSpec(shape = [None, None], dtype = tf.float32),
                                  tf.SparseTensorSpec(shape = [None, None], dtype = tf.float32)])
    def train_step(self, ands_index1,ands_index2,n_vars, n_clauses, n_ands, solution, clauses_adj,clauses_graph, variables_graph):
        with tf.GradientTape() as tape:
            _, loss, step = self.call(ands_index1 = ands_index1,ands_index2 = ands_index2,
                                      clauses_adj = clauses_adj,
                                      n_vars=n_vars, n_clauses=n_clauses, training=True, labels=solution, clauses_graph=clauses_graph, variables_graph=variables_graph)
            train_vars = self.trainable_variables
            gradients = tape.gradient(loss, train_vars)
            # for g in gradients:
            #     if tf.math.is_nan(tf.reduce_sum(g)):
            #         print("g", g)
            self.optimizer.apply_gradients(zip(gradients, train_vars))

        return {
            "steps_taken": step,
            "loss": loss,
            "gradients": gradients
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.int64),
                                  tf.TensorSpec(shape=(), dtype=tf.int64),
                                  tf.TensorSpec(shape=(), dtype=tf.int64),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32)])
    def predict_step(self, ands_index1,ands_index2,n_vars, n_clauses, n_ands, solution, clauses_adj,clauses_graph, variables_graph):
        predictions, loss, step = self.call(ands_index1 = ands_index1,ands_index2 = ands_index2,
                                      clauses_adj = clauses_adj,
                                      n_vars=n_vars, n_clauses=n_clauses, training=False, labels=None, clauses_graph=clauses_graph, variables_graph=variables_graph)

        return {
            "steps_taken": step,
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__,
                HP_FEATURE_MAPS: self.feature_maps,
                HP_QUERY_MAPS: self.query_maps,
                HP_TRAIN_ROUNDS: self.train_rounds,
                HP_TEST_ROUNDS: self.test_rounds,
                HP_MLP_LAYERS: self.vote_layers,
                }
