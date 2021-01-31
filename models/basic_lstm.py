import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from utils.optimiser import Optimiser

from models.network import NeuralNetwork


class BasicLSTM(Optimiser, NeuralNetwork):
    def __init__(self, args, rnd, reuse=False):
        super(BasicLSTM, self).__init__(args, rnd)
        self.max_num_agents = args.max_num_agents
        self.embedding_size = args.embedding_size
        # ensure all ops are from the same graph
        with tf.variable_scope(self.model_type, reuse=reuse):
            self.g = tf.Graph()
            with self.g.as_default():
                self._build_graph()

        self.init_session()

    def _build_graph(self):
        '''Method that builds the graph as per our blog post.'''

        cell = rnn_cell.BasicLSTMCell(self.num_units, state_is_tuple=False)

        self.input_data = tf.placeholder(tf.float32, [None, self.sequence_length, self.input_dim])
        self.target_data = tf.placeholder(tf.float32, [None, self.sequence_length, self.output_dim])

        self.lr = tf.Variable(self.learning_rate, trainable=False, name="learning_rate")
        self.LSTM_states = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        self.output_size = 5
        
        # input dimensionality is the x and y position at every step
        # the output is comprised of two means, two std and 1 corr variable
        embeddings_w, embeddings_b = self.buildEmbeddings(input_dim=self.input_dim, output_dim=self.output_size)

        # Prepare inputs ..
        inputs = tf.split(self.input_data, self.sequence_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # the actual LSTM model
        embedded_inputs = self.embedInputs(inputs, embeddings_w["input_embed"], embeddings_b["input_embed"])
        outputs = []
        for idx, e_input in enumerate(embedded_inputs):
            reuse = True if idx > 0 else False
            output, last_state = self.lstmAdvance(e_input, cell, self.LSTM_states)
            outputs.append(output)
 
        self.final_output = self.finalLayer(self.g, outputs, embeddings_w["output_embed"], embeddings_b["output_embed"])

        self.final_states = last_state
        # reshape target data so that it aligns with predictions
        flat_target_data = tf.reshape(self.target_data, [-1, 2])
        # Extract the x-coordinates and y-coordinates from the target data
        [x_data, y_data] = tf.split(flat_target_data, 2, 1)

        # Extract coef from output of the linear output layer
        [o_mux, o_muy, o_sx, o_sy, o_corr] = self.getCoef(self.final_output, self.output_size)
        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr

        # to have the same shape as per-frame LSTM -> use same inference
        self.final_output = tf.expand_dims(self.final_output, 0)

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.name_scope("Optimization"):
                lossfunc = self.get2DGaussianLossFunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

                self.cost = tf.div(lossfunc, (self.batch_size * self.sequence_length))
                trainable_params = tf.trainable_variables()

                # apply L2 regularisation
                # l2 = 0.0005*sum(tf.nn.l2_loss(tvar) for tvar in trainable_params)
                # self.cost = self.cost + l2
                tf.summary.scalar('cost', self.cost)

                self.gradients = tf.gradients(self.cost, trainable_params)
                grads, _ = tf.clip_by_global_norm(self.gradients, self.grad_clip)

                # Adam might also do a good job as in Graves (2013)
                optimizer = tf.train.RMSPropOptimizer(self.lr)
                # Train operator
                self.train_op = optimizer.apply_gradients(zip(grads, trainable_params))

        self.summary = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()

    def step(self, contents, **kwargs):
        '''This step is specific to all lstms that make a single 
           step per agent ... perhaps I can redefine this in model_tools 
           once I have a better idea of what I am doing ...
        '''
        # accum_err = 0
        # for batch in range(self.batch_size):
            # x_batch, y_batch = contents['inputs'][batch], contents['targets'][batch]
        x_batch, y_batch, first_batch = contents['inputs'], contents['targets'], contents['firsts']

        return self.train(x_batch, y_batch) if self.mode != tf.contrib.learn.ModeKeys.INFER \
            else self.basicInfer(
                x_batch[0],
                real_data=first_batch,
                initial_states=kwargs['initial_states'],
                error_type=kwargs['error_type'])
