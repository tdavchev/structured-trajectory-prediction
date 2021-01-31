import tensorflow as tf
from numpy import zeros
from utils import DEFAULT_SEED
from utils.optimiser import Optimiser

from models.network import NeuralNetwork


class DynamicMDNLSTM(Optimiser, NeuralNetwork):
    def __init__(self, args, rnd, reuse=False):
        super(DynamicMDNLSTM, self).__init__(args, rnd)
        # input is [batch_size, sequence, im_dim+act_dim] --> [100, 20, 96+60]
        # output is [batch_size, sequence, im_dim] --> [100, 20, 96]
        self.k_mix = args.num_mixtures
        self.action_include = args.action_include
        self.vae_z_size = args.input_dim
        # to train using D only uncomment this
        # self.extra_dim = args.num_units
        # End
        # for normal RDB training uncomment this
        self.extra_dim = args.input_dim + args.num_units
        # End
        with tf.variable_scope('mdn_rnn', reuse=reuse):
            self.g = tf.Graph()
            with self.g.as_default():
                tf.set_random_seed(DEFAULT_SEED)
                self.build_graph()

        self.init_session()

    def _build_graph(self):
        with tf.name_scope('Feed_tensors'):
            self.inputs = tf.placeholder(shape=[self.batch_size, self.sequence_length, self.input_dim], dtype=tf.float32, name='inputs')
            # self.action_inputs = tf.placeholder(shape=[self.batch_size, self.sequence_length, self.max_num_agents * 2], dtype=tf.float32, name='action_inputs')
            self.action_inputs = tf.placeholder(shape=[self.batch_size, self.sequence_length, self.max_num_agents, 2], dtype=tf.float32, name='action_inputs')
            self.targets = tf.placeholder(shape=[self.batch_size, self.sequence_length, self.output_dim], dtype=tf.float32, name='targets')

        self.output_size = self.output_dim * self.k_mix * 3

        # build embeddings
        # embeddings_w, embeddings_b = self.buildEmbeddings(input_dim=self.max_num_agents, output_dim=self.output_size, action_cond=True)
        embeddings_w, embeddings_b = self.buildEmbeddings(input_dim=self.max_num_agents, output_dim=self.output_size)

        # self.inputs_acts = self.inputs if not self.action_include else self.attachActions(embeddings_w, embeddings_b)
        self.inputs_acts = self.inputs if not self.action_include else self.attachActions(embeddings_w, embeddings_b)

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.histogram('global_step', self.global_step)

        with tf.name_scope('Cell'):
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_units, layer_norm=False)
            
        self.initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        output, last_state = tf.nn.dynamic_rnn(
            cell, 
            self.inputs_acts, 
            initial_state=self.initial_state, 
            time_major=False, 
            swap_memory=True, 
            dtype=tf.float32, 
            scope="RNN")

        def prepare_output(output, embbedings_w, embbedings_b):
            output = tf.reshape(output, [-1, self.num_units])
            output = tf.nn.xw_plus_b(output, embbedings_w["output_embed"], embbedings_b["output_embed"])

            return tf.reshape(output, [-1, self.k_mix * 3])

        self.final_state = last_state
        # let's do it the distribution way... output --> mu, sigma and corr
        output = prepare_output(output, embeddings_w, embeddings_b)
        [o_pi, o_mu, o_sx] = self.get1DmixtureCoef(output)

        self.o_mu = o_mu
        self.o_sx = o_sx
        self.o_pi = o_pi

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.name_scope("Optimization"):
                flat_targets = tf.reshape(self.targets, [-1, 1])
                self.flat = flat_targets

                lossfunc = self.getMixture1DGaussianLossFunc(self.g, o_pi, o_mu, o_sx, flat_targets)
                
                self.cost = tf.reduce_mean(lossfunc)
                tf.summary.scalar('cost', self.cost)

                self.lr = tf.Variable(self.learning_rate, trainable=False)
                optimizer = tf.train.AdamOptimizer(self.lr)

                gvs = optimizer.compute_gradients(self.cost)
                capped_gvs = [(tf.clip_by_value(grad, -self.grad_clip, self.grad_clip), var) for grad, var in gvs if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

        self.summary = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()

    # def attachActions(self, embbedings_w, embbedings_b):
    #     def prepare_actons():
    #         a_inputs = tf.split(self.action_inputs, self.sequence_length, 1)
    #         a_inputs = [tf.squeeze(input_, [1]) for input_ in a_inputs]

    #         a_inputs = [tf.split(_act, self.batch_size, 0) for _act in a_inputs]
    #         return [[tf.squeeze(input_, [0]) for input_ in _act] for _act in a_inputs]

    #     # Prepare inputs ..
    #     a_inputs = prepare_actons()
    #     first_embed = self.embed_first_inputs(a_inputs, embbedings_w["action_combine"], embbedings_b["action_combine"])
    #     embedded_act_inputs = self.embed_final_inputs(first_embed, embbedings_w["action_embed"], embbedings_b["action_embed"])

    #     return tf.concat([self.inputs, tf.transpose(tf.stack(embedded_act_inputs), [1, 0, 2])], 2) 

    # def buildEmbeddings(self, output_dim):
    #     embbedings_w = {}
    #     embbedings_b = {}
    #     print("OVERRIDE!!!!!")
    #     with tf.variable_scope("coordinate_embedding"):
    #         embbedings_w["action_combine"] = tf.get_variable("first_embedding_w", [2, 1])
    #         embbedings_b["action_combine"] = tf.get_variable("first_embedding_b", [1])
    #         embbedings_w["action_embed"] = tf.get_variable("embedding_w", [self.max_num_agents, self.embedding_size])
    #         embbedings_b["action_embed"] = tf.get_variable("embedding_b", [self.embedding_size]) 

    #     with tf.variable_scope('output_embeddings'):
    #         embbedings_w["output_embed"] = tf.get_variable("output_w", [self.num_units, output_dim])
    #         embbedings_b["output_embed"] = tf.get_variable("output_b", [output_dim])

    #     return embbedings_w, embbedings_b

    # def attachActions(self):
    #     return tf.concat([self.inputs, self.action_inputs], 2)  

    def attachActions(self, embbedings_w, embbedings_b):
        def prepare_actons():
            a_inputs = tf.split(self.action_inputs, self.sequence_length, 1)
            a_inputs = [tf.squeeze(input_, [1]) for input_ in a_inputs]

            a_inputs = [tf.split(_act, self.batch_size, 0) for _act in a_inputs]
            return [[tf.squeeze(input_, [0]) for input_ in _act] for _act in a_inputs]

        # Prepare inputs ..
        a_inputs = prepare_actons()
        first_embed = self.embed_first_inputs(a_inputs, embbedings_w["action_combine"], embbedings_b["action_combine"])
        embedded_act_inputs = self.embed_final_inputs(first_embed, embbedings_w["input_embed"], embbedings_b["input_embed"])

        return tf.concat([self.inputs, tf.transpose(tf.stack(embedded_act_inputs), [1, 0, 2])], 2)    

    def embed_first_inputs(self, inputs, embedding_w, embedding_b):
        # embed the inputs
        with tf.name_scope("Embed_inputs"):
            embedded_inputs = []
            for x in inputs:
                embedded_sub_inputs = []
                for entry in x:
                    # Each x is a 2D tensor of size numPoints x 2
                    embedded_x = tf.nn.relu(tf.add(tf.matmul(entry, embedding_w), embedding_b))
                    embedded_sub_inputs.append(embedded_x)

                embedded_inputs.append(tf.squeeze(tf.stack(embedded_sub_inputs), [2]))
            return embedded_inputs

    def embed_final_inputs(self, inputs, embedding_w, embedding_b):
        # embed the inputs
        with tf.name_scope("Embed_inputs"):
            embedded_inputs = []
            for x in inputs:
                # Each x is a 2D tensor of size numPoints x 2
                embedded_x = tf.nn.relu(tf.add(tf.matmul(x, embedding_w), embedding_b))
                embedded_inputs.append(embedded_x)

            return embedded_inputs

    def step(self, contents, **kwargs):
        '''This step is specific to all lstms that make a single 
           step per agent ... perhaps I can redefine this in model_tools 
           once I have a better idea of what I am doing ...
        '''
        latent_sequence = zeros((contents['inputs'].shape[0], contents['inputs'].shape[1], self.extra_dim))
        for batch in range(contents['inputs'].shape[0]): # control's batch size is important now..
            x_batch, y_batch, a_batch = \
                contents['inputs'][batch], contents['targets'][batch], contents['actions'][batch]

            latent_sequence[batch] = self.inferHa(
                x_batch.reshape((1, x_batch.shape[0], x_batch.shape[1])),
                a_batch.reshape((1, a_batch.shape[0], a_batch.shape[1], a_batch.shape[2])),
                # a_batch.reshape((1, a_batch.shape[0], a_batch.shape[1])),
                initial_states=kwargs['initial_states'], 
                last_axis=self.extra_dim,
                which=kwargs['which'])

        return latent_sequence
