import tensorflow as tf
from tensorflow.python.ops import rnn_cell

from models.network import NeuralNetwork
from utils import DEFAULT_SEED
from utils.optimiser import Optimiser


class MapCond3LSTM(Optimiser, NeuralNetwork):
    def __init__(self, args, rnd, reuse=False):
        super(MapCond3LSTM, self).__init__(args, rnd)
        self.extra_dim = args.extra_dim
        self.vae_z_size = args.l_size
        # ensure all ops are from the same graph
        with tf.variable_scope("mapcond_lstm", reuse=reuse):
            self.g = tf.Graph()
            with self.g.as_default():
                tf.set_random_seed(DEFAULT_SEED)
                self.build_graph()

        self.init_session()

    def _build_graph(self):
        '''Method that builds the graph as per our blog post.'''

        cell = rnn_cell.BasicLSTMCell(self.num_units, state_is_tuple=False)
        self.input_data = tf.placeholder(tf.float32, [self.sequence_length, self.max_num_agents, self.input_dim])
        self.extra_data = tf.placeholder(tf.float32, [self.sequence_length, self.extra_dim])
        self.target_data = tf.placeholder(tf.float32, [self.sequence_length, self.max_num_agents, self.output_dim])

        self.lr = tf.Variable(self.learning_rate, trainable=False, name="learning_rate")
        self.output_size = 5
        # at the moment consider a single initial state, then try with one per agent
        # then try the one per intent!
        self.LSTM_states = tf.zeros([self.max_num_agents, cell.state_size], name="LSTM_states") # state size includes both h and z ..
        self.initial_states = tf.split(self.LSTM_states, self.max_num_agents, 0)
        # self.initial_states = lstm_cell.zero_state(batch_size=self.max_num_agents, dtype=tf.float32)
        # self.initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        self.output_states = tf.split(tf.zeros([self.max_num_agents, cell.output_size]), self.max_num_agents, 0)

        frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, self.sequence_length, 0)]
        extra_frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.extra_data, self.sequence_length, 0)]
        frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, self.sequence_length, 0)]

        nonexistent_agent = tf.constant(0.0, name="zero_agent")
        self.cost = tf.constant(0.0, name="cost")
        self.counter = tf.constant(0.0, name="counter")
        self.increment = tf.constant(1.0, name="increment")
        # input dimensionality is the x and y position at every step
        # the output is comprised of two means, two std and 1 corr variable

        embeddings_w, embeddings_b = self.buildEmbeddings(
                input_dim=self.input_dim-1,
                output_dim=self.output_size,
                sub_action_size=self.extra_dim,
                reduced_act=self.embedding_size)

        self.initial_output = tf.split(tf.zeros([self.max_num_agents, self.output_size]), self.max_num_agents, 0)
        for seq, frame in enumerate(frame_data):
            print("Frame number", seq)
            current_frame_data = frame  # max_num_agents x 3 tensor
            current_extra_frame_data = extra_frame_data[seq] # max_num_agents x extra_dim tensor
            for agent in range(self.max_num_agents):
                agent_id = current_frame_data[agent, 0]
                self.spatial_input = tf.slice(current_frame_data, [agent, 1], [1, 2])
                self.extra_input = tf.reshape(current_extra_frame_data, [1, self.extra_dim])
                # embedded_spatial_input = tf.nn.leaky_relu(tf.nn.xw_plus_b(self.spatial_input, embeddings_w["input_embed"], embeddings_b["input_embed"]))
                # embedded_extra_input = tf.nn.leaky_relu(tf.nn.xw_plus_b(self.extra_input, embeddings_w["action_combine"], embeddings_b["action_combine"]))

                embedded_spatial_input = tf.nn.relu(tf.nn.xw_plus_b(self.spatial_input, embeddings_w["input_embed"], embeddings_b["input_embed"]))
                embedded_extra_input = tf.nn.relu(tf.nn.xw_plus_b(self.extra_input, embeddings_w["action_combine"], embeddings_b["action_combine"]))

                complete_input = tf.concat([embedded_spatial_input, embedded_extra_input], 1)
                # complete_input = tf.nn.xw_plus_b(complete_input, embeddings_w["complete_embedding"], embeddings_b["complete_embedding"])

                # NOTE: This last state is the last agent's last state ...
                reuse = True if seq > 0 or agent > 0 else False
                self.output_states[agent], self.initial_states[agent] = self.lstmAdvance(complete_input, cell, self.initial_states[agent], reuse)
                self.initial_output[agent] = tf.nn.xw_plus_b(self.output_states[agent], embeddings_w["output_embed"], embeddings_b["output_embed"])

                [x_data, y_data] = tf.split(tf.slice(frame_target_data[seq], [agent, 1], [1, 2]), 2, 1)
                target_agent_id = frame_target_data[seq][agent, 0]

                [o_mux, o_muy, o_sx, o_sy, o_corr] = self.getCoef(self.initial_output[agent], self.output_size)
                lossfunc = self.get2DGaussianLossFunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
                self.cost = tf.where(tf.logical_or(tf.equal(agent_id, nonexistent_agent), tf.equal(target_agent_id, nonexistent_agent)), self.cost, tf.add(self.cost, lossfunc))
                self.counter = tf.where(tf.logical_or(tf.equal(agent_id, nonexistent_agent), tf.equal(target_agent_id, nonexistent_agent)), self.counter, tf.add(self.counter, self.increment))

        self.final_states = tf.concat(self.initial_states, 0)
        self.final_output = self.initial_output
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.name_scope("Optimization"):
                self.cost = tf.divide(self.cost, self.counter)
                tvars = tf.trainable_variables()
                l2 = 0.0005 * sum(tf.nn.l2_loss(t_param) for t_param in tvars)
                self.cost = self.cost + l2
                tf.summary.scalar('cost', self.cost)
                self.gradients = tf.gradients(self.cost, tvars)
                grads, _ = tf.clip_by_global_norm(self.gradients, self.grad_clip)
                optimizer = tf.train.RMSPropOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.summary = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()

    def step(self, contents, **kwargs):
        '''This step is specific to all lstms that make a single 
           step per agent ... perhaps I can redefine this in model_tools 
           once I have a better idea of what I am doing ...
        '''
        accum_err = 0
        for batch in range(self.batch_size):
            # x_batch, y_batch, l_batch, first_batch, az_contents = \
            #     contents['inputs'][batch], contents['targets'][batch], contents['latent_sequence'][batch], contents['firsts'][batch], contents['az_contents'][batch]

            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                 z_contents, act_contents = contents['az_contents']['inputs'][batch], contents['az_contents']['actions'][batch]
            else:
                l_batch = contents['latent_sequence'][batch]
                # l_batch = l_batch[batch, :, 96:]

            x_batch, y_batch, first_batch = contents['inputs'][batch], contents['targets'][batch], contents['firsts'][batch]
            err, gradients, summary, velocities, _ = self.conditionedTrain(x_batch, y_batch, l_batch) if self.mode != tf.contrib.learn.ModeKeys.INFER \
                else self.conditionedInfer(
                    x_batch,
                    # latent_sequence=l_batch,
                    real_data=first_batch,                   #redundant
                    z_contents=z_contents,
                    act_contents=act_contents,
                    initial_states=kwargs['initial_states'],
                    # comment out for R only
                    error_type=kwargs['error_type'],
                    dynamics_model=contents['dynamics_model'],
                    dynamics_initial_states=contents['dynamics_initial_states'],
                    which=contents['which'],
                    # End
                    frame_id=contents['frames_ids'][0])

            accum_err += err

        return accum_err, gradients, summary, velocities, None
