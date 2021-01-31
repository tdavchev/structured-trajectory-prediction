import numpy as np

import utils.distributions as distributions


class Optimiser(object):
    def __init__(self, args, rnd):
        super(Optimiser, self).__init__(args)
        self.rnd = rnd
        self.observed_length = args.observed_length
        self.predicted_length = args.predicted_length
        # self.input_dim = args.input_dim
        # self.output_dim = args.output_dim
        self.temperature = args.temperature

    def train(self, x_batch, y_batch):
        # Feed the source, target data and the initial LSTM state to the model
        feed = {
            self.input_data: x_batch,
            self.target_data: y_batch
        }
        # SHOULD RETURN GRADIENT FOR TESTING PURPOSES instead of self.mask
        cur_train_loss, _, gradients, summary = self.sess.run(
            [self.cost, self.train_op, self.gradients, self.summary], feed)

        return cur_train_loss, gradients, summary, None, None

    def conditionedTrain(self, x_batch, y_batch, l_batch):
        # Feed the source, target data and the initial LSTM state to the model
        feed = {
            self.input_data: x_batch,
            self.target_data: y_batch,
            self.extra_data: l_batch
        }
        cur_train_loss, _, gradients, summary = self.sess.run(
            [self.cost, self.train_op, self.gradients, self.summary], feed)
        # cur_train_loss, _, gradients, summary, brei, brqh = self.sess.run(
        #     [self.cost, self.train_op, self.gradients, self.summary, self.output_states, self.initial_states], feed)

        return cur_train_loss, gradients, summary, None, None

    def socialTrain(self, x_batch, y_batch):
        g_batch = self.getSequenceGridMask(x_batch)
        feed = {
            self.input_data: x_batch,
            self.grid_data: g_batch,
            self.target_data: y_batch
        }

        cur_train_loss, _, gradients, summary = self.sess.run(
            [self.cost, self.train_op, self.gradients, self.summary], feed)

        return cur_train_loss, gradients, summary # , None, None

    def inferHa(self, x_batch, a_batch, **kwargs):
        latent_sequence = np.zeros((x_batch.shape[1], kwargs['last_axis']))
        # latent_sequence = np.zeros((x_batch.shape[1], self.input_dim))
        state = kwargs['initial_states']
        for pos in range(x_batch.shape[1]):
            # using real one every time
            cur_z = np.reshape(x_batch[:, pos, :], (1, 1, self.input_dim))
            # don't need if action is not included...
            cur_act = np.reshape(a_batch[:, pos, :], (1, 1, self.max_num_agents, 2))
            # cur_act = np.reshape(a_batch[:, pos, :], (1, 1, 2 * self.max_num_agents))
            feed = {
                self.inputs: cur_z,
                self.action_inputs: cur_act,
                self.initial_state: state
            }

            # start = time.time()
            [state, o_pi, o_mu, o_sx] = self.sess.run(
                [self.final_state, self.o_pi, self.o_mu, self.o_sx],
                feed
            )
            # end = time.time()
            # print("time taken: ", end - start)
            chosen_mean, chosen_logstd = distributions.choose_moments(o_pi, o_mu, o_sx, self.input_dim, self.temperature, self.rnd)
            rand_gaussian = self.rnd.randn(self.input_dim)*np.sqrt(self.temperature)
            next_z = chosen_mean + chosen_logstd*rand_gaussian
            next_z = next_z.reshape((1, next_z.shape[0]))
            # prev_z is 96 and next_state.h is 768
            # should this be previous h ? should be this one yeh..
            if kwargs['last_axis'] == self.vae_z_size + self.num_units:
                if kwargs['which'] == 'next':
                    latent_sequence[pos] = np.concatenate((next_z, state.h), axis=1).reshape((next_z.shape[1] + state.h.shape[1]))
                else:
                    latent_sequence[pos] = np.concatenate((cur_z[0], state.h), axis=1).reshape((cur_z.shape[2] + state.h.shape[1]))
            elif kwargs['last_axis'] == self.vae_z_size:
                # compare performance against next_z only ... hm, ha, ah?! hehe
                # TODO: add condition for next_z and I should only have next_z here ...
                latent_sequence[pos] = cur_z.reshape(cur_z.shape[2])
            elif kwargs['last_axis'] == self.num_units:
                latent_sequence[pos] = state.h[0]

            # latent_sequence[pos] = np.concatenate((next_z, state.h), axis=1).reshape((next_z.shape[1] + state.h.shape[1]))
            # latent_sequence[pos] = next_z.reshape(next_z.shape[1])

        return latent_sequence

    def conditionedInfer(self, x_batch, **kwargs):
        traj_distrib = []
        decode_this  = []
        frame_id = kwargs['frame_id'][0]
        # if frame_id == 880:
        #     print("Stignah do 880!")
        # for i in range(25):
        for i in range(1):
            obs_traj = np.copy(x_batch[:self.observed_length])
            states = kwargs['initial_states']
            # stupid_states = kwargs['initial_states'].copy()
            # latent_sequence = kwargs['latent_sequence']

            z_contents = kwargs['z_contents']
            act_contents = kwargs['act_contents']
            # comment out for R only
            dynamics_model = kwargs['dynamics_model']
            dynamics_states = kwargs['dynamics_initial_states']
            which = kwargs['which']
            # End

            obs_z = np.copy(z_contents[:self.observed_length])
            obs_act = np.copy(act_contents[:self.observed_length])

            error_type = kwargs['error_type']
            returned_states = np.zeros((self.max_num_agents, states.shape[1]))


            for idx, position in enumerate(obs_traj[:-1]):
                cur_z = np.reshape(obs_z[idx], (1, 1, self.vae_z_size))
                cur_act = np.reshape(obs_act[idx], (1, 1, self.max_num_agents, 2))

                # comment out for R only
                feed = {
                    dynamics_model.inputs: cur_z,
                    dynamics_model.action_inputs: cur_act,
                    dynamics_model.initial_state: dynamics_states
                }

                [dynamics_states, o_pi, o_mu, o_sx] = dynamics_model.sess.run(
                    [dynamics_model.final_state, dynamics_model.o_pi, dynamics_model.o_mu, dynamics_model.o_sx],
                    feed
                )

                chosen_mean, chosen_logstd = distributions.choose_moments(o_pi, o_mu, o_sx, dynamics_model.input_dim, dynamics_model.temperature, dynamics_model.rnd)
                rand_gaussian = dynamics_model.rnd.randn(dynamics_model.input_dim)*np.sqrt(dynamics_model.temperature)
                next_z_dum = np.copy(chosen_mean) + np.copy(chosen_logstd)*np.copy(rand_gaussian)
                next_z_dum = next_z_dum.reshape((1, next_z_dum.shape[0]))
                # end

                input_data_tensor = np.reshape(np.copy(position), (1, self.max_num_agents, self.input_dim))
                target_data_tensor = np.reshape(obs_traj[idx+1], (1, self.max_num_agents, self.input_dim))
                # comment out for R only
                extra_data_tensor = np.concatenate((np.copy(cur_z[0]), np.copy(dynamics_states.h)), axis=1).reshape((cur_z.shape[2] + dynamics_states.h.shape[1]))
                # end
                # uncomment for R only
                # extra_data_tensor = cur_z[0].reshape((cur_z.shape[2]))
                # end
                # extra_data_tensor = dynamics_states.h.reshape((dynamics_states.h.shape[1]))
                extra_data_tensor = np.reshape(extra_data_tensor, (1, self.extra_dim)).astype(np.float32)

                feed = {
                    self.input_data: input_data_tensor,
                    self.extra_data: extra_data_tensor,
                    self.target_data: target_data_tensor,
                    self.LSTM_states: states
                    }

                [states] = self.sess.run([self.final_states], feed)


            returned_z = np.copy(obs_z)
            returned_act = np.copy(obs_act)
            last_z = np.copy(obs_z[-1])
            last_act = np.copy(obs_act[-1])
            decode_this.append(last_z)
            # decode_this.append(np.copy(next_z_dum))

            prev_z = np.reshape(np.copy(last_z), (1, 1, self.vae_z_size))
            prev_act = np.reshape(np.copy(last_act), (1, 1, self.max_num_agents, 2))

            returned_traj = obs_traj
            last_position = obs_traj[-1]

            prev_data = np.reshape(last_position, (1, self.max_num_agents, self.input_dim))
            prev_target_data = np.reshape(x_batch[obs_traj.shape[0]], (1, self.max_num_agents, self.input_dim))
            for t in range(self.predicted_length):
                # comment out for R only
                feed = {
                    dynamics_model.inputs: prev_z,
                    dynamics_model.action_inputs: prev_act,
                    dynamics_model.initial_state: dynamics_states
                }

                # start = time.time()
                [dynamics_states, o_pi, o_mu, o_sx] = dynamics_model.sess.run(
                    [dynamics_model.final_state, dynamics_model.o_pi, dynamics_model.o_mu, dynamics_model.o_sx],
                    feed
                )

                prev_extra = np.concatenate((np.copy(prev_z[0]), np.copy(dynamics_states.h)), axis=1).reshape((prev_z.shape[2] + dynamics_states.h.shape[1]))
                # end
                # uncomment for R
                # prev_extra = prev_z[0].reshape((prev_z.shape[2]))
                #
                # prev_extra = dynamics_states.h.reshape((dynamics_states.h.shape[1]))
                prev_extra = np.reshape(prev_extra, (1, self.extra_dim)).astype(np.float32)

                # comment out for R only
                chosen_mean, chosen_logstd = distributions.choose_moments(o_pi, o_mu, o_sx, dynamics_model.input_dim, dynamics_model.temperature, dynamics_model.rnd)
                rand_gaussian = dynamics_model.rnd.randn(dynamics_model.input_dim)*np.sqrt(dynamics_model.temperature)
                next_z = chosen_mean + chosen_logstd*rand_gaussian
                next_z = next_z.reshape((1, next_z.shape[0]))
                prev_z[0] = np.copy(next_z)
                # End

                # uncomment for R only
                # prev_z[0] = z_contents[self.observed_length + t]
                # End

                feed = {
                        self.input_data: prev_data,
                        self.extra_data: prev_extra,
                        self.LSTM_states: states,
                        self.target_data: prev_target_data}

                [output, states] = self.sess.run(
                    [self.final_output, self.final_states], feed)

                newpos = np.zeros((1, self.max_num_agents, self.output_dim))
                next_act = np.zeros((1, self.max_num_agents, self.output_dim-1))
                for a_index, a_output in enumerate(output):
                    if prev_data[0][a_index][0] != 0 and prev_target_data[0][a_index][0] != 0:
                        returned_states[a_index] = states[a_index]
                    [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(np.copy(a_output[0]), self.output_size, 0)
                    mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])
                    next_x, next_y = distributions.sample_2d_normal(mux, muy, sx, sy, corr, self.rnd)
                    newpos[0, a_index, :] = [prev_data[0, a_index, 0], np.copy(next_x), np.copy(next_y)]

                    if x_batch[t, a_index, 0] == 0.0:
                        continue
                    elif newpos[0, a_index, 0] == 0.0:
                        continue
                    else:
                        if x_batch[t, a_index, 1] > 1.0 or x_batch[t, a_index, 1] < 0.0:
                            continue
                        elif x_batch[t, a_index, 2] > 1.0 or x_batch[t, a_index, 2] < 0.0:
                            continue

                        next_act[0, a_index, :] = [np.copy(next_x), np.copy(next_y)]

                    # if prev_data[0, a_index, 0] != 0:
                    #     print("Pedestrian ID", prev_data[0, a_index, 0])
                    #     print("Predicted parameters", mux, muy, sx, sy, corr)
                    #     print("New Position", next_x, next_y)
                    #     print("Target Position", prev_target_data[0, a_index, 1], prev_target_data[0, a_index, 2])
                    #     print()

                returned_traj = np.vstack((returned_traj, np.copy(newpos)))
                # prev_data = newpos
                prev_data = newpos.reshape((1, self.max_num_agents, self.output_dim))
                prev_act[0] = next_act
                if t != self.predicted_length - 1:
                    prev_target_data = np.reshape(x_batch[obs_traj.shape[0] + t + 1], (1, self.max_num_agents, self.output_dim))

            idcs = []
            motion = []
            onlys = [False for _ in range(self.max_num_agents)]
            # I only use the perdictions of agents from the last part of the observed_length
            for yo, agent in enumerate(x_batch[self.observed_length-1]):
                if agent[1] == 0.0 and agent[2] == 0.0:
                    continue
                
                if x_batch[self.observed_length][yo][1] == 0.0 and x_batch[self.observed_length][yo][2] == 0.0:
                    continue

                onlys[yo] = True
                idcs.append(yo)

            starts = [-1 for i in range(self.max_num_agents)]
            for t in range(x_batch.shape[0]):
                for a_idx, a_output in enumerate(x_batch[t, :]):
                    if a_output[0] == 0.0 or a_idx not in idcs:
                        continue

                    if a_output[1] != 0 or a_output[2] != 0:
                        if starts[a_idx] == -1:
                            starts[a_idx] = t

            agents_in = [idx for idx, agent_loc in enumerate(starts) if agent_loc != -1]
            trajs = []
            for agent_id in agents_in:
                if agent_id == 0:
                    continue

                trajs.append(x_batch[starts[int(agent_id)]:, int(agent_id)])

            full_trajs = []
            for traj in trajs:
                temp = []
                for pos in traj:
                    if pos[1] != 0.0 and pos[2] != 0.0:
                        temp.append(pos)

                full_trajs.append(temp)

            velocities = [[0, 0] for i in range(self.max_num_agents)]
            for a_id, tr in enumerate(full_trajs):
                for idx, pos in enumerate(tr[1:]):
                    if idx == 0:
                        continue

                    velocities[idcs[a_id]][0] = pos[1] - tr[idx-1][1]
                    velocities[idcs[a_id]][1] = pos[2] - tr[idx-1][2]

            complete_traj = np.copy(returned_traj)
            traj_distrib.append(np.copy(complete_traj))

        if error_type == 'ade':
            answer = distributions.get_mean_error(
                predicted_traj=np.round(complete_traj, decimals=4), 
                true_traj=np.round(x_batch, decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)
        elif error_type == 'fde':
            answer = distributions.get_final_error(
                predicted_traj=np.round(complete_traj, decimals=4), 
                true_traj=np.round(x_batch, decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)

        return answer, [state for idx, state in enumerate(returned_states) if idx in idcs], [x_batch[:, i, 1:] for i in range(self.max_num_agents) if i in idcs], [velocity for idx, velocity in enumerate(velocities) if idx in idcs], None

    def socialInfer(self, x_batch, **kwargs):
        # print("tuk sum spoko.")
        obs_traj = np.copy(x_batch[:self.observed_length])
        states = kwargs['initial_states']
        error_type = kwargs['error_type']
        frame_id = kwargs['frame_id'][0]
        traj_distrib = []
        for i in range(25):
            grid = self.getSequenceGridMask(x_batch)
            for idx, position in enumerate(obs_traj[:-1]):
                input_data_tensor = np.reshape(position, (1, self.max_num_agents, self.input_dim))
                grid_data = np.reshape(grid[idx, :], (1, self.max_num_agents, self.max_num_agents, self.grid_size*self.grid_size))
                target_data_tensor = np.reshape(obs_traj[idx+1], (1, self.max_num_agents, self.input_dim))
                feed = {self.input_data: input_data_tensor, self.target_data: target_data_tensor, self.grid_data: grid_data, self.LSTM_states: states}
                [states] = self.sess.run([self.final_states], feed)

            returned_traj = obs_traj
            last_position = obs_traj[-1]

            prev_data = np.reshape(last_position, (1, self.max_num_agents, self.input_dim))
            prev_grid_data = np.reshape(grid[-1], (1, self.max_num_agents, self.max_num_agents, self.grid_size*self.grid_size))

            prev_target_data = np.reshape(x_batch[obs_traj.shape[0]], (1, self.max_num_agents, self.input_dim))
            for t in range(self.predicted_length):
                feed = {
                        self.input_data: prev_data,
                        self.LSTM_states: states,
                        self.grid_data: prev_grid_data,
                        self.target_data: prev_target_data}

                [output, states] = self.sess.run(
                    [self.final_output, self.final_states], feed)

                newpos = np.zeros((1, self.max_num_agents, 3))
                for a_index, a_output in enumerate(output):
                    [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(a_output[0], self.output_size, 0)
                    mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])
                    next_x, next_y = distributions.sample_2d_normal(mux, muy, sx, sy, corr, self.rnd)
                    newpos[0, a_index, :] = [prev_data[0, a_index, 0], next_x, next_y]

                    if x_batch[t, a_index, 0] == 0.0:
                        continue
                    elif newpos[0, a_index, 0] == 0.0:
                        continue
                    else:
                        if x_batch[t, a_index, 1] > 1.0 or x_batch[t, a_index, 1] < 0.0:
                            continue
                        elif x_batch[t, a_index, 2] > 1.0 or x_batch[t, a_index, 2] < 0.0:
                            continue
                        


                returned_traj = np.vstack((returned_traj, newpos))
                prev_data = newpos
                prev_grid_data = self.getSequenceGridMask(prev_data)
                if t != self.predicted_length - 1:
                    prev_target_data = np.reshape(x_batch[obs_traj.shape[0] + t + 1], (1, self.max_num_agents, self.output_dim))

            complete_traj = returned_traj
            traj_distrib.append(np.copy(complete_traj))

        traj_distrib = np.array(traj_distrib)
        
        if error_type == 'ade':
            answer = distributions.get_mean_error(
                predicted_traj=np.round(complete_traj, decimals=4),
                true_traj=np.round(x_batch, decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)
        elif error_type == 'fde':
            answer = distributions.get_final_error(
                predicted_traj=np.round(complete_traj, decimals=4),
                true_traj=np.round(x_batch, decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)

        return answer, None, None

    
    def basicInfer(self, x_batch, **kwargs):
        ''' Very very complicated to read ... '''
        # x_batch = np.around(x_batch, decimals=4)
        obs_traj = np.copy(x_batch[:self.observed_length])
        states = kwargs['initial_states']
        # fake_states = kwargs['initial_states'].copy()
        error_type = kwargs['error_type']
        returned_states = np.zeros((self.max_num_agents, states.shape[1]))

        # for idx, position in enumerate(x_batch[:-1]):
        #     input_data_tensor = np.reshape(position, (1, self.max_num_agents, self.input_dim))
        #     target_data_tensor = np.reshape(x_batch[idx+1], (1, self.max_num_agents, self.input_dim))
        #     feed = {self.input_data: input_data_tensor, self.target_data: target_data_tensor, self.LSTM_states: fake_states}
        #     [fake_states] = self.sess.run([self.final_states], feed)

        for idx, position in enumerate(obs_traj[:-1]):
            input_data_tensor = np.reshape(position, (1, 1, self.input_dim))
            target_data_tensor = np.reshape(obs_traj[idx+1], (1, 1, self.input_dim))
            feed = {self.input_data: input_data_tensor, self.target_data: target_data_tensor, self.LSTM_states: states}
            [states] = self.sess.run([self.final_states], feed)

        if self.input_dim == 2:
            per_frame = False
            x_dim, y_dim = 0, 1
        elif self.input_dim == 3:
            x_dim, y_dim = 1, 2

        returned_traj = obs_traj
        last_position = obs_traj[-1]

        prev_data = np.reshape(last_position, (1, 1, self.input_dim))
        prev_target_data = np.reshape(x_batch[obs_traj.shape[0]], (1, 1, self.input_dim))

        for t in range(self.predicted_length):
            feed = {
                self.input_data : prev_data,
                self.LSTM_states: states,
                self.target_data: prev_target_data}
            [output, states] = self.sess.run(
                [self.final_output, self.final_states], feed)
            newpos = np.zeros((1, 1, self.output_dim))
            returned_states = states
            [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(output[0][0], 5, 0)
            mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])
            #print("--> ", mux, muy, sx, sy, corr)
            next_x, next_y = distributions.sample_2d_normal(mux, muy, sx, sy, corr, self.rnd)
            if self.output_dim == 2:
                newpos[0, 0, :] = [next_x, next_y]

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
            prev_data = newpos.reshape((1, 1, self.output_dim))
            if t != self.predicted_length - 1:
                prev_target_data = np.reshape(x_batch[self.observed_length + t + 1], (1, 1, self.input_dim))

        complete_traj = returned_traj
        if error_type == 'ade':
            answer = distributions.get_mean_error(
                predicted_traj=np.round(complete_traj, decimals=4),
                true_traj=np.round(x_batch, decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)
        elif error_type == 'fde':
            answer = distributions.get_final_error(
                predicted_traj=np.round(complete_traj, decimals=4),
                true_traj=np.round(x_batch, decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)

        return answer, None, None, None, None


    def infer(self, x_batch, **kwargs):
        traj_distrib = []
        decode_this  = []
        frame_id = kwargs['frame_id'][0]
        for i in range(25):
            ''' Very very complicated to read ... '''
            # x_batch = np.around(x_batch, decimals=4)
            obs_traj = np.copy(x_batch[:self.observed_length])
            states = kwargs['initial_states']
            # fake_states = kwargs['initial_states'].copy()
            error_type = kwargs['error_type']
            returned_states = np.zeros((states.shape))

            # for idx, position in enumerate(x_batch[:-1]):
            #     input_data_tensor = np.reshape(position, (1, self.max_num_agents, self.input_dim))
            #     target_data_tensor = np.reshape(x_batch[idx+1], (1, self.max_num_agents, self.input_dim))
            #     feed = {self.input_data: input_data_tensor, self.target_data: target_data_tensor, self.LSTM_states: fake_states}
            #     [fake_states] = self.sess.run([self.final_states], feed)

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
                    if prev_data[0][a_index][0] != 0 and prev_target_data[0][a_index][0] != 0:
                        # print(np.array(returned_states[a_index]).shape)
                        # print(np.array(a_index).shape)
                        # print("---")
                        returned_states[a_index] = states[a_index]
                    [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(a_output[0], 5, 0)
                    mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])
                    #print("--> ", mux, muy, sx, sy, corr)
                    next_x, next_y = distributions.sample_2d_normal(mux, muy, sx, sy, corr, self.rnd)
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

            idcs = []
            motion = []
            onlys = [False for _ in range(self.max_num_agents)]
            # I only use the perdictions of agents from the last part of the observed_length
            for yo, agent in enumerate(x_batch[self.observed_length-1]):
                if agent[1] == 0.0 and agent[2] == 0.0:
                    continue

                if x_batch[self.observed_length][yo][1] == 0.0 and x_batch[self.observed_length][yo][2] == 0.0:
                    continue

                onlys[yo] = True
                idcs.append(yo)

            starts = [-1 for i in range(self.max_num_agents)]
            for t in range(x_batch.shape[0]):
                for a_idx, a_output in enumerate(x_batch[t, :]):
                    if a_output[0] == 0.0 or a_idx not in idcs:
                        continue

                    if a_output[1] != 0 or a_output[2] != 0:
                        if starts[a_idx] == -1:
                            starts[a_idx] = t

            agents_in = [idx for idx, agent_loc in enumerate(starts) if agent_loc != -1]
            trajs = []
            for agent_id in agents_in:
                if agent_id == 0:
                    continue

                trajs.append(x_batch[starts[int(agent_id)]:, int(agent_id)])

            full_trajs = []
            for traj in trajs:
                temp = []
                for pos in traj:
                    if pos[1] != 0.0 and pos[2] != 0.0:
                        temp.append(pos)

                full_trajs.append(temp)

            velocities = [[0, 0] for i in range(self.max_num_agents)]
            for a_id, tr in enumerate(full_trajs):
                for idx, pos in enumerate(tr[1:]):
                    if idx == 0:
                        continue

                    velocities[idcs[a_id]][0] = pos[1] - tr[idx-1][1]
                    velocities[idcs[a_id]][1] = pos[2] - tr[idx-1][2]

            complete_traj = np.copy(returned_traj)
            traj_distrib.append(np.copy(complete_traj))

        traj_distrib = np.array(traj_distrib)
        if error_type == 'ade':
            answer = distributions.get_mean_error(
                predicted_traj=np.round(complete_traj, decimals=4),
                true_traj=np.round(x_batch, decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)
        elif error_type == 'fde':
            answer = distributions.get_final_error(
                predicted_traj=np.round(complete_traj, decimals=4),
                true_traj=np.round(x_batch, decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)

        return answer, [state for idx, state in enumerate(returned_states) if idx in idcs], [x_batch[:, i, 1:] for i in range(self.max_num_agents) if i in idcs], [velocity for idx, velocity in enumerate(velocities) if idx in idcs], None
