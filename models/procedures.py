import numpy as np
import utils.distributions as distributions


class Procedures(object):
    def __init__(self, args, rnd):
        super(Procedures, self).__init__(args, rnd)
        self.rnd = rnd
        self.observed_length = args.observed_length
        self.predicted_length = args.predicted_length
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.temperature = args.temperature

    def train(self, x_batch, y_batch):
        # Feed the source, target data and the initial LSTM state to the model
        feed = {
            self.input_data: x_batch,
            self.target_data: y_batch
        }
        cur_train_loss, _ = self.sess.run(
            [self.cost, self.train_op], feed)

        return cur_train_loss

    def conditionedTrain(self, x_batch, y_batch, l_batch):
        # Feed the source, target data and the initial LSTM state to the model
        feed = {
            self.input_data: x_batch,
            self.target_data: y_batch,
            self.extra_data: l_batch
        }
        cur_train_loss, _ = self.sess.run(
            [self.cost, self.train_op], feed)

        return cur_train_loss

    def inferHa(self, x_batch, a_batch, **kwargs):
        # why is this taking so long ...
        latent_sequence = np.zeros((x_batch.shape[1], self.input_dim + self.num_units))
        state = kwargs['initial_states']
        for pos in range(x_batch.shape[1]):
            # using real one every time
            cur_z = np.reshape(x_batch[:, pos, :], (1, 1, self.input_dim))
            # don't need if action is not included...
            cur_act = np.reshape(a_batch[:, pos, :], (1, 1, 2 * self.max_num_agents))
            feed = {
                self.inputs: cur_z,
                self.action_inputs: cur_act
            }

            [state, o_pi, o_mu, o_sx, summary_test] = self.sess.run(
                [self.final_state, self.o_pi, self.o_mu, self.o_sx, self.merged],
                feed
            )
            chosen_mean, chosen_logstd = distributions.choose_moments(o_pi, o_mu, o_sx, self.input_dim, self.temperature, self.rnd)
            rand_gaussian = self.rnd.randn(self.input_dim)*np.sqrt(self.temperature)
            next_z = chosen_mean + chosen_logstd*rand_gaussian
            next_z = next_z.reshape((1, next_z.shape[0]))
            # prev_z is 96 and next_state.h is 768
            # should this be previous h ? should be this one yeh..
            # latent_sequence[pos] = cur_z.reshape(cur_z.shape[2])
            # latent_sequence[pos] = state.h[0]
            latent_sequence[pos] = np.concatenate((cur_z[0], state.h), axis=1).reshape((cur_z.shape[2] + state.h.shape[1]))
            # latent_sequence[pos] = np.concatenate((next_z, state.h), axis=1).reshape((next_z.shape[1] + state.h.shape[1]))
            # latent_sequence[pos] = next_z.reshape(next_z.shape[1])

        return latent_sequence

    def conditionedInfer(self, x_batch, **kwargs):
        obs_traj = x_batch[:self.observed_length]
        states = kwargs['initial_states']
        latent_sequence = kwargs['latent_sequence']
        for idx, position in enumerate(obs_traj[:-1]):
            input_data_tensor = np.reshape(position, (1, self.max_num_agents, self.input_dim))
            target_data_tensor = np.reshape(obs_traj[idx+1], (1, self.max_num_agents, self.input_dim))
            extra_data_tensor = np.zeros((1, self.extra_dim), dtype=np.float32)
            extra_data_tensor[0, :] = latent_sequence[idx]

            feed = {
                self.input_data: input_data_tensor,
                self.extra_data: extra_data_tensor,
                self.target_data: target_data_tensor,
                self.LSTM_states: states
                }

            [states, cost] = self.sess.run([self.final_states, self.cost], feed)

        returned_traj = obs_traj
        last_position = obs_traj[-1]
        last_extra = latent_sequence[obs_traj.shape[0]-1] # think if that's correct ..

        prev_data = np.reshape(last_position, (1, self.max_num_agents, self.input_dim))
        prev_extra = np.reshape(last_extra, (1, self.extra_dim)) #96, 768
        prev_target_data = np.reshape(x_batch[obs_traj.shape[0]], (1, self.max_num_agents, self.input_dim))
        for t in range(self.predicted_length):
            feed = {
                    self.input_data: prev_data,
                    self.extra_data: prev_extra,
                    self.LSTM_states: states,
                    self.target_data: prev_target_data}

            [output, states] = self.sess.run(
                [self.final_output, self.final_states], feed)

            newpos = np.zeros((1, self.max_num_agents, 3))
            for a_index, a_output in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(a_output[0], self.output_size, 0)
                mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])
                next_x, next_y = distributions.sample_2d_normal(mux, muy, sx, sy, corr, self.rnd)
                newpos[0, a_index, :] = [prev_data[0, a_index, 0], next_x, next_y]

                if prev_data[0, a_index, 0] != 0:
                    print("Pedestrian ID", prev_data[0, a_index, 0])
                    print("Predicted parameters", mux, muy, sx, sy, corr)
                    print("New Position", next_x, next_y)
                    print("Target Position", prev_target_data[0, a_index, 1], prev_target_data[0, a_index, 2])
                    print()
            
            returned_traj = np.vstack((returned_traj, newpos))
            prev_data = newpos
            prev_extra[0, :] = latent_sequence[obs_traj.shape[0]+t]
            if t != self.predicted_length - 1:
                prev_target_data = np.reshape(x_batch[obs_traj.shape[0] + t + 1], (1, self.max_num_agents, self.output_dim))

        complete_traj = returned_traj
        return distributions.get_mean_error(
            predicted_traj=np.round(complete_traj, decimals=4), 
            true_traj=np.round(x_batch, decimals=4),
            observed_length=self.observed_length,
            max_num_agents=self.max_num_agents)

    def infer(self, x_batch, **kwargs):
        ''' Very very complicated to read ... '''
        x_batch = np.around(x_batch, decimals=4)
        obs_traj = np.copy(x_batch[:self.observed_length])

        states = kwargs['initial_states']
        for idx, position in enumerate(obs_traj[:-1]):
            input_data_tensor = np.reshape(position, (1, self.max_num_agents, self.input_dim))
            target_data_tensor = np.reshape(obs_traj[idx+1], (1, self.max_num_agents, self.input_dim))
            feed = {self.input_data: input_data_tensor, self.target_data: target_data_tensor, self.LSTM_states: states}
            [states] = self.sess.run([self.final_states], feed)

        if self.input_dim == 2:
            per_frame = False
            x_dim, y_dim = 0, 1
        elif self.input_dim == 3:
            x_dim, y_dim = 1, 2

        returned_traj = obs_traj
        last_position = obs_traj[-1]

        prev_data = np.reshape(last_position, (1, self.max_num_agents, self.input_dim))
        prev_target_data = np.reshape(x_batch[obs_traj.shape[0]], (1, self.max_num_agents, self.input_dim))

        for t in range(self.predicted_length):
            feed = {
                self.input_data : prev_data,
                self.LSTM_states: states,
                self.target_data: prev_target_data}
            [output, states] = self.sess.run(
                [self.final_output, self.final_states], feed)
            newpos = np.zeros((1, self.max_num_agents, self.output_dim))
            for a_index, a_output in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(a_output[0], 5, 0)
                mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])
                next_x, next_y = np.around(distributions.sample_2d_normal(mux, muy, sx, sy, corr, self.rnd), decimals=4)
                if self.output_dim == 2:
                    newpos[0, a_index, :] = [next_x, next_y]
                elif self.output_dim == 3:
                    newpos[0, a_index, :] = [prev_data[0, a_index, 0], next_x, next_y]

                # (prev_data[0, a_index][-1].shape > 2 and was there?! for cases where input_dim == 2
                # if prev_data[0, a_index].shape[-1] == 2 or prev_data[0, a_index, 0] != 0:
                #     print("Pedestrian ID", prev_data[0, a_index, 0])
                #     print("Predicted parameters", mux, muy, sx, sy, corr)
                #     print("New Position", next_x, next_y)
                #     print("Target Position", prev_target_data[0, a_index, x_dim], prev_target_data[0, a_index, y_dim])
                #     print()

            if self.output_dim == 2:
                newpos = np.reshape(newpos, (1, 2))

            returned_traj = np.vstack((returned_traj, newpos))
            prev_data = newpos.reshape((1, self.max_num_agents, self.output_dim))
            if t != self.predicted_length - 1:
                prev_target_data = np.reshape(x_batch[self.observed_length + t + 1], (1, self.max_num_agents, self.input_dim))

        complete_traj = returned_traj
        return distributions.get_mean_error(
            predicted_traj=complete_traj,
            true_traj=x_batch,
            observed_length=self.observed_length,
            max_num_agents=self.max_num_agents)
