'''
Social LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 17th October 2016
'''

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from utils import DEFAULT_SEED
from utils.grid import getSequenceGridMask
from utils.optimiser import Optimiser

from models.network import NeuralNetwork


class SocialLSTM(Optimiser, NeuralNetwork):
    def __init__(self, args, rnd, reuse=False):
        '''
        Initialisation function for the class SocialLSTM
        params:
        args : Contains arguments required for the model creation
        '''
        super(SocialLSTM, self).__init__(args, rnd)
        self.neighbourhood_size = args.neighbourhood_size
        self.grid_size = args.grid_size
        self.dimensions = args.dimensions
        # ensure all ops are from the same graph
        with tf.variable_scope("mapcond_lstm", reuse=reuse):
            self.g = tf.Graph()
            with self.g.as_default():
                tf.set_random_seed(DEFAULT_SEED)
                self.build_graph()

        self.init_session()

    def _build_graph(self):
        # NOTE : For now assuming, batch_size is always 1. That is the input
        # to the model is always a sequence of frames

        # Construct the basicLSTMCell recurrent unit with a dimension given by args.rnn_size
        cell = rnn_cell.BasicLSTMCell(self.num_units, state_is_tuple=False)
        # if not infer and args.keep_prob < 1:
        # cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=args.keep_prob)

        # placeholders for the input data and the target data
        # A sequence contains an ordered set of consecutive frames
        # Each frame can contain a maximum of 'args.maxNumPeds' number of peds
        # For each ped we have their (pedID, x, y) positions as input
        self.input_data = tf.placeholder(tf.float32, [self.sequence_length, self.max_num_agents, self.input_dim], name="input_data")
        # Grid data would be a binary matrix which encodes whether a pedestrian is present in
        # a grid cell of other pedestrian
        self.grid_data = tf.placeholder(tf.float32, [self.sequence_length, self.max_num_agents, self.max_num_agents, self.grid_size*self.grid_size], name="grid_data")
        # target data would be the same format as input_data except with
        # one time-step ahead
        self.target_data = tf.placeholder(tf.float32, [self.sequence_length, self.max_num_agents, self.input_dim], name="target_data")
        # Variable to hold the value of the learning rate
        self.lr = tf.Variable(self.learning_rate, trainable=False, name="learning_rate")
        # Output dimension of the model
        self.output_size = 5
        # Define LSTM states for each pedestrian
        self.LSTM_states = tf.zeros([self.max_num_agents, cell.state_size], name="LSTM_states")
        self.initial_states = tf.split(self.LSTM_states, self.max_num_agents, 0)
        # Define hidden output states for each pedestrian
        # self.output_states = tf.zeros([args.maxNumPeds, cell.output_size], name="hidden_states")
        self.output_states = tf.split(tf.zeros([self.max_num_agents, cell.output_size]), self.max_num_agents, 0)
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
        # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, self.sequence_length, 0)]
        # This would contain a list of tensors each of shape MNP x MNP x (GS**2) encoding the mask
        # grid_frame_data = tf.split(0, args.seq_length, self.grid_data, name="grid_frame_data")
        grid_frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.grid_data, self.sequence_length, 0)]
        # frame_target_data = tf.split(0, args.seq_length, self.target_data, name="frame_target_data")
        frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, self.sequence_length, 0)]

        # Tensor to represent non-existent ped
        nonexistent_agent = tf.constant(0.0, name="zero_agent")
        self.cost = tf.constant(0.0, name="cost")
        self.counter = tf.constant(0.0, name="counter")
        self.increment = tf.constant(1.0, name="increment")

        # Define embedding and output layers
        embeddings_w, embeddings_b = self.buildEmbeddings(input_dim=self.input_dim-1, output_dim=5)

        # Containers to store output distribution parameters
        # self.initial_output = tf.zeros([args.maxNumPeds, self.output_size], name="distribution_parameters")
        self.initial_output = tf.split(tf.zeros([self.max_num_agents, self.output_size]), self.max_num_agents, 0)

        # Iterate over each frame in the sequence
        for seq, frame in enumerate(frame_data):
            print("Frame number", seq)
            current_frame_data = frame  # MNP x 3 tensor
            current_grid_frame_data = grid_frame_data[seq]  # MNP x MNP x (GS**2) tensor
            social_tensor = self.getSocialTensor(current_grid_frame_data, self.output_states)  # MNP x (GS**2 * RNN_size)
            # NOTE: Using a tensor of zeros as the social tensor
            # social_tensor = tf.zeros([args.maxNumPeds, args.grid_size*args.grid_size*args.rnn_size])
            for agent in range(self.max_num_agents):
                # agent_id of the current pedestrian
                agent_id = current_frame_data[agent, 0]
                # Extract x and y positions of the current ped
                self.spatial_input = tf.slice(current_frame_data, [agent, 1], [1, 2])  # Tensor of shape (1,2)
                # Extract the social tensor of the current ped
                self.tensor_input = tf.slice(social_tensor, [agent, 0], [1, self.grid_size*self.grid_size*self.num_units])  # Tensor of shape (1, g*g*r)
                embedded_spatial_input = tf.nn.relu(tf.nn.xw_plus_b(self.spatial_input, embeddings_w['input_embed'], embeddings_b['input_embed']))
                # Embed the tensor input
                embedded_tensor_input = tf.nn.relu(tf.nn.xw_plus_b(self.tensor_input, embeddings_w['social_embed'], embeddings_b['social_embed']))

                complete_input = tf.concat([embedded_spatial_input, embedded_tensor_input], 1)
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

    def getSequenceGridMask(self, x_batch):
        return getSequenceGridMask(x_batch, self.dimensions, self.neighbourhood_size, self.grid_size)

    def getSocialTensor(self, grid_frame_data, output_states):
        '''
        Computes the social tensor for all the maxNumPeds in the frame
        params:
        grid_frame_data : A tensor of shape MNP x MNP x (GS**2)
        output_states : A list of tensors each of shape 1 x RNN_size of length MNP
        '''
        # Create a zero tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.zeros([self.max_num_agents, self.grid_size*self.grid_size, self.num_units], name="social_tensor")
        # Create a list of zero tensors each of shape 1 x (GS**2) x RNN_size of length MNP
        social_tensor = tf.split(social_tensor, self.max_num_agents, 0)
        # Concatenate list of hidden states to form a tensor of shape MNP x RNN_size
        hidden_states = tf.concat(output_states, 0)
        # Split the grid_frame_data into grid_data for each pedestrians
        # Consists of a list of tensors each of shape 1 x MNP x (GS**2) of length MNP
        grid_frame_agent_data = tf.split(grid_frame_data, self.max_num_agents, 0)
        # Squeeze tensors to form MNP x (GS**2) matrices
        grid_frame_agent_data = [tf.squeeze(input_, [0]) for input_ in grid_frame_agent_data]

        # For each pedestrian
        for agent in range(self.max_num_agents):
            # Compute social tensor for the current pedestrian
            with tf.name_scope("tensor_calculation"):
                social_tensor_agent = tf.matmul(tf.transpose(grid_frame_agent_data[agent]), hidden_states)
                social_tensor[agent] = tf.reshape(social_tensor_agent, [1, self.grid_size*self.grid_size, self.num_units])

        # Concatenate the social tensor from a list to a tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.concat(social_tensor, 0)
        # Reshape the tensor to match the dimensions MNP x (GS**2 * RNN_size)
        social_tensor = tf.reshape(social_tensor, [self.max_num_agents, self.grid_size*self.grid_size*self.num_units])
        return social_tensor

    def step(self, contents, **kwargs):
        '''This step is specific to all lstms that make a single 
           step per agent ... perhaps I can redefine this in model_tools 
           once I have a better idea of what I am doing ...
        '''
        accum_err = 0
        for batch in range(self.batch_size):
            x_batch, y_batch, first_batch = \
                contents['inputs'][batch], contents['targets'][batch], contents['firsts']

            err, gradients, summary = self.socialTrain(x_batch, y_batch) if self.mode != tf.contrib.learn.ModeKeys.INFER \
                else self.socialInfer(
                    x_batch,
                    real_data=first_batch,                   #redundant
                    initial_states=kwargs['initial_states'],
                    error_type=kwargs['error_type'],
                    frame_id=contents['frames_ids'][0])

            accum_err += err

        return accum_err, gradients, summary, None, None
