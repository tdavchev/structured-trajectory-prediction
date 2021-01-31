import json

import numpy as np
import tensorflow as tf
import utils.distributions as distributions


class NeuralNetwork(object):
    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.batch_size = args.batch_size
        self.sequence_length = args.sequence_length
        self.num_units = args.num_units
        self.embedding_size = args.embedding_size
        self.learning_rate = args.learning_rate
        self.max_num_agents = args.max_num_agents
        self.mode = args.mode
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.grad_clip = args.grad_clip
        self.model_type = args.model_type
        # Those two shouldn't be here ..
        self.action_include = args.action_include if 'action_include' in args._fields else False
        self.social_grid_include = args.social_grid_include if 'social_grid_include' in args._fields else False
        self.label_include = args.label_include if 'label_include' in args._fields else False
        self.extra_linear_layer = args.extra_linear_layer if 'extra_linear_layer' in args._fields else False
        # self.oned_vae = args.oned_vae if 'oned_vae' in args._fields else False

    def init_session(self):
        return self._init_session()

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def build_graph(self):
        return self._build_graph()

    def _build_graph(self):
        pass

    def _close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def close(self):
        self._close_sess()

    def getCoef(self, output, output_size):
        with self.g.as_default():
            # eq 20 -> 22 of Graves (2013)
            z = output
            z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, output_size, axis=1)

            # The output must be exponentiated for the std devs
            z_sx = tf.exp(z_sx)
            z_sy = tf.exp(z_sy)
            # Tanh applied to keep it in the range [-1, 1]
            z_corr = tf.tanh(z_corr)

            return [z_mux, z_muy, z_sx, z_sy, z_corr]

    def embedInputs(self, inputs, embedding_w, embedding_b):
        # embed the inputs
        with self.g.as_default():
            with tf.name_scope("Embed_inputs"):
                embedded_inputs = []
                for x in inputs:
                    # Each x is a 2D tensor of size numPoints x 2
                    embedded_x = tf.nn.relu(tf.add(tf.matmul(x, embedding_w), embedding_b))
                    embedded_inputs.append(embedded_x)

                return embedded_inputs

    def get1DmixtureCoef(self, output):
        with self.g.as_default():
            out_pi, out_mu, out_sigma = tf.split(output, 3, 1)
            out_pi = tf.subtract(out_pi, tf.reduce_logsumexp(out_pi, 1, keep_dims=True))

            out_sigma = tf.exp(out_sigma)

            return out_pi, out_mu, out_sigma

    def finalLayer(self, graph, outputs, output_w, output_b):
        with self.g.as_default():
            with tf.name_scope("Final_layer"):
                # Apply the linear layer. Output would be a 
                # tensor of shape 1 x output_size
                output = tf.reshape(tf.concat(outputs, 1), [-1, self.num_units])
                output = tf.nn.xw_plus_b(output, output_w, output_b)
                return output

    def lstmAdvance(self, embedded_inputs, cell, state, reuse=False, scope_name="LSTM"):
        with self.g.as_default():
            # advance the lstm cell state with one for each entry
            with tf.variable_scope(scope_name) as scope:
                if reuse:
                    scope.reuse_variables()
                output, last_state = cell(embedded_inputs, state)

                return output, last_state

    def buildEmbeddings(self, input_dim, output_dim, sub_action_size=2, reduced_act=1, num_classes=0, action_cond=False):
        with self.g.as_default(): # Think this is unncessary
            embeddings_w = {}
            embeddings_b = {}
            # if self.oned_vae:
            #     with tf.variable_scope("oned_embedding"):
            #         embeddings_w["fc_1"] = tf.get_variable("one_d_embedding_w", [sub_action_size*2, reduced_act])
            #         embeddings_b["fc_1"] = tf.get_variable("one_d_embedding_b", [reduced_act])
            #         # tf.summary.histogram('action_embedding', embeddings_w["action_combine"])
            #         # tf.summary.histogram('action_embedding', embeddings_b["action_combine"])

            if self.action_include:
                with tf.variable_scope("action_embedding"):
                    embeddings_w["action_combine"] = tf.get_variable("first_embedding_w", [sub_action_size, reduced_act], initializer=tf.glorot_normal_initializer())
                    embeddings_b["action_combine"] = tf.get_variable("first_embedding_b", [reduced_act], initializer=tf.constant_initializer(0.01))
                    # tf.summary.histogram('action_embedding', embeddings_w["action_combine"])
                    # tf.summary.histogram('action_embedding', embeddings_b["action_combine"])

            if self.extra_linear_layer:
                with tf.variable_scope("extra_embedding"):
                    embeddings_w["complete_embedding"] = tf.get_variable("complete_embedding_w", [2*reduced_act, self.num_units], initializer=tf.glorot_normal_initializer())
                    embeddings_b["complete_embedding"] = tf.get_variable("complete_embedding_b", [self.num_units], initializer=tf.constant_initializer(0.01))

            if self.social_grid_include:
                with tf.variable_scope("social_embedding"):
                    embeddings_w["social_embed"] = tf.get_variable("social_embedding_w", [self.grid_size*self.grid_size*self.num_units, self.embedding_size], initializer=tf.glorot_normal_initializer())
                    embeddings_b["social_embed"] = tf.get_variable("social_embedding_b", [self.embedding_size], initializer=tf.constant_initializer(0.01))
                    # tf.summary.histogram('social_embedding', embeddings_w["social_embed"])
                    # tf.summary.histogram('social_embedding', embeddings_b["social_embed"])

            if self.label_include:
                with tf.variable_scope("labels_embedding"):
                    embeddings_w["labels_embed"] = tf.get_variable("labels_embedding_w", [self.num_units, num_classes], initializer=tf.glorot_normal_initializer())
                    embeddings_b["labels_embed"] = tf.get_variable("labels_embedding_b", [num_classes], initializer=tf.constant_initializer(0.01))
                    # tf.summary.histogram('label_embedding', embeddings_w["label_embed"])
                    # tf.summary.histogram('label_embedding', embeddings_b["label_embed"])

            # Define variables for embedding the input
            with tf.variable_scope("coordinate_embedding"):
                embeddings_w["input_embed"] = tf.get_variable("embedding_w", [input_dim, self.embedding_size], initializer=tf.glorot_normal_initializer())
                embeddings_b["input_embed"] = tf.get_variable("embedding_b", [self.embedding_size], initializer=tf.constant_initializer(0.01))

            # Define variables for the output linear layer
            with tf.variable_scope("output_embeddings"):
                embeddings_w["output_embed"] = tf.get_variable("output_w", [self.num_units, output_dim], initializer=tf.glorot_normal_initializer())
                embeddings_b["output_embed"] = tf.get_variable("output_b", [output_dim], initializer=tf.constant_initializer(0.01))

            # # # # Used for tensorboard summary # # #
            # tf.summary.histogram('embedding_w', embeddings_w["input_embed"])
            # tf.summary.histogram('embedding_b', embeddings_b["input_embed"])
            # tf.summary.histogram('output_w', embeddings_w["output_embed"])
            # tf.summary.histogram('output_b', embeddings_b["output_embed"])
            # # # # # # # # # # # # # # # # # # # # #

            return embeddings_w, embeddings_b

    def get2DGaussianLossFunc(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        with self.g.as_default():
            # Calculate the PDF of the data w.r.t to the distribution
            result0 = distributions.tf_2d_normal(self.g, x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            # For numerical stability purposes as in Vemula (2018)
            epsilon = 1e-20
            # Numerical stability
            result1 = -tf.log(tf.maximum(result0, epsilon))

            return tf.reduce_sum(result1)

    def getMixture1DGaussianLossFunc(self, z_pi, z_mu, z_sx, x_data):
        with self.g.as_default():
            result0 = distributions.tf_1d_lognormal(self.g, x_data, z_mu, z_sx)
            result0 = tf.add(z_pi, result0)
            result0 = tf.reduce_logsumexp(result0, 1, keep_dims=True)
            return -tf.reduce_mean(result0)

    # Thanks D. Ha!
    # modified from https://github.com/hardmaru/WorldModelsExperiments
    def getModelParams(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p*10000).astype(np.int).tolist() # ..?!
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                pshape = self.sess.run(var).shape
                p = np.array(params[idx])
                print(pshape, p.shape)
                assert pshape == p.shape, "inconsistent shape"
                assign_op = var.assign(p.astype(np.float)/10000.)
                self.sess.run(assign_op)
                idx += 1

    def save_json(self, jsonfile='lstm.json'):
        model_params, model_shapes, model_names = self.getModelParams()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def load_json(self, jsonfile='lstm.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
