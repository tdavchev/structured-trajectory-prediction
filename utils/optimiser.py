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

    def dilConvTrain(self, x, v, yaw, y):

        inp_mask = np.zeros(x.shape[:-1])
        target_mask = np.zeros(y.shape[:-1])
        for b in range(x.shape[0]):
            for s in range(x.shape[1]):
                for a in range(x.shape[2]):
                    if x[b, s, a, 1] != 0.0 or x[b, s, a, 2] != 0.0:
                        inp_mask[b, s, a] = 1
                    if y[b, s, a, 1] != 0.0 or y[b, s, a, 2] != 0.0:
                        target_mask[b, s, a] = 1

        # take away ids..
        _x = x[:, :, :, 1:]
        _y = y[:, :, :, 1:]
        _v = np.sum(v[:, :, :, 1:], axis=3).reshape((v.shape[0], v.shape[1], v.shape[2], 1))
        _yaw = yaw[:, :, :, 1:]

        target_v = _v[:, 1:, :, :]
        target_v = np.concatenate((target_v, target_v[:, -1, :, :].reshape((_v.shape[0], 1, _v.shape[2], _v.shape[3]))), axis=1)

        target_yaw = _yaw[:, 1:, :, :]
        target_yaw = np.concatenate((target_yaw, target_yaw[:, -1, :, :].reshape((_yaw.shape[0], 1, _yaw.shape[2], _yaw.shape[3]))), axis=1)

        feed = {
            self.input_data: np.concatenate((_x, _v, _yaw), axis=3),
            self.inp_mask: inp_mask,
            self.target_data: np.concatenate((_y, target_v, target_yaw), axis=3),
            self.target_mask: target_mask
        }
        # SHOULD RETURN GRADIENT FOR TESTING PURPOSES instead of self.mask
        cur_train_loss, _ = self.sess.run(
            [self.cost, self.train_op], feed_dict=feed)

        return cur_train_loss, None, None, None, None

    def train_vc(self, x_batch, y_batch, i_batch):
        # Feed the source, target data and the initial LSTM state to the model
        feed = {
            self.input_data: x_batch,
            self.target_data: y_batch,
            self.image_data: i_batch
        }
        # SHOULD RETURN GRADIENT FOR TESTING PURPOSES instead of self.mask
        cur_train_loss, r_loss, mmd_loss, _, gradients, summary = self.sess.run(
            [self.cost, self.r_loss, self.mmd_loss, self.train_op, self.gradients, self.summary], feed)

        return cur_train_loss, gradients, summary, r_loss, mmd_loss

    def train_foxy_vc(self, x_batch, y_batch, i_batch, l_batch):
        # Feed the source, target data and the initial LSTM state to the model
        feed = {
            self.input_data: x_batch,
            self.target_data: y_batch,
            self.l: l_batch
        }
        # SHOULD RETURN GRADIENT FOR TESTING PURPOSES instead of self.mask
        cur_train_loss, r_loss, kl_loss, _, gradients, summary = self.sess.run(
            [self.cost, self.r_loss, self.kl_loss, self.train_op, self.gradients, self.summary], feed)

        return cur_train_loss, gradients, summary, r_loss, kl_loss

    def trainClassif(self, x_batch, y_batch, labels):
        # Feed the source, target data and the initial LSTM state to the model
        feed = {
            self.input_data: x_batch,
            self.target_data: y_batch,
            self.target_labels: labels
        }
        cur_train_loss, _, gradients, summary, accuracy, brei, brqh = self.sess.run(
            [self.cost, self.train_op, self.gradients, self.summary, self.accuracy, self.output_states, self.initial_states], feed)

        return cur_train_loss, gradients, summary, None, np.mean(accuracy)

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

    def dilatedInfer(self, x_batch, **kwargs):
        inp_mask = np.zeros(x_batch.shape[:-1])
        target_mask = np.zeros(x_batch.shape[:-1])
        for b in range(x_batch.shape[0]):
            for s in range(x_batch.shape[1]):
                for a in range(x_batch.shape[2]):
                    if x_batch[b, s, a, 1] != 0.0 or x_batch[b, s, a, 2] != 0.0:
                        inp_mask[b, s, a] = 1
                    if kwargs['target'][b, s, a, 1] != 0.0 or kwargs['target'][b, s, a, 2] != 0.0:
                        target_mask[b, s, a] = 1

        error_type = kwargs['error_type']

        # take away ids..
        _x = x_batch[:, :, :, 1:]
        _y = kwargs['target'][:, :, :, 1:]
        _v = np.sum(kwargs['velocity'][:, :, :, 1:], axis=3)
        _v = _v.reshape((_v.shape[0], _v.shape[1], _v.shape[2], 1))
        _yaw = kwargs['yaw'][:, :, :, 1:]

        obs_traj = np.copy(_x[0, :self.observed_length+1].reshape((1, 5, self.max_num_agents, 2)))
        obs_vel = np.copy(_v[0, :self.observed_length+1].reshape((1, 5, self.max_num_agents)))
        obs_yaw = np.copy(_yaw[0, :self.observed_length+1].reshape((1, 5, self.max_num_agents, 2)))
        obs_y = np.copy(_y[0, :self.observed_length+1].reshape((1, 5, self.max_num_agents, 2)))

        target_v = obs_vel[:, 1:, :]
        target_v = np.concatenate((target_v, target_v[:, -1, :].reshape((obs_vel.shape[0], 1, obs_vel.shape[2]))), axis=1)

        target_yaw = obs_yaw[:, 1:, :, :]
        target_yaw = np.concatenate((target_yaw, target_yaw[:, -1, :, :].reshape((obs_yaw.shape[0], 1, obs_yaw.shape[2], obs_yaw.shape[3]))), axis=1)

        feed = {
                self.input_data: np.concatenate((obs_traj[:, :-1, :], obs_vel[:, :-1, :].reshape((1, 4, self.max_num_agents, 1)), obs_yaw[:, :-1, :]), axis=3),
                self.inp_mask: inp_mask[:, :self.observed_length, :].reshape((1, 4, self.max_num_agents)),
                self.target_data: np.concatenate((obs_traj[:, 1:, :], obs_vel[:, 1:, :].reshape((1, 4, self.max_num_agents, 1)), obs_yaw[:, 1:, :]), axis=3),
                self.target_mask: inp_mask[:,1:self.observed_length+1, :].reshape((1, 4, self.max_num_agents))
            }

            # SHOULD RETURN GRADIENT FOR TESTING PURPOSES instead of self.mask
        mux, muy, muv, sx, sy, sv, corr, output_rz, output_rw = self.sess.run(
            [self.mux, self.muy, self.muv, self.sx, self.sy, self.sv, self.corr, self.output_rz, self.output_rw], feed_dict=feed)

        newpos = np.zeros((1, self.max_num_agents, self.output_dim))
        newyaw = np.zeros((1, self.max_num_agents, 2))
        for a_index in range(self.max_num_agents):
            if a_index != 0:
                if x_batch[0, self.observed_length-1, a_index, 1] != 0 and x_batch[0, self.observed_length-1, a_index, 2] != 0:
                    next_x, next_y = distributions.sample_2d_normal(mux[0, -1, a_index], muy[0, -1, a_index], sx[0, -1, a_index], sy[0, -1, a_index], corr[0, -1, a_index], self.rnd)
                    newpos[0, a_index, :] = [x_batch[0, self.observed_length-1, a_index, 0], next_x, next_y]
                    newyaw[0, a_index, :] = [output_rz[0, -1, a_index], output_rw[0, -1, a_index]]

        returned_traj = np.array([[[x_batch[0, _pt, a_id, 0], obs_traj[0, _pt, a_id, 0], obs_traj[0, _pt, a_id, 1]] for a_id in range(self.max_num_agents)] for _pt in range(obs_traj.shape[1]-1)])
        returned_traj = np.vstack((returned_traj, newpos))
        last_position = obs_traj[:, 1:-1, :]
        last_v_position = obs_vel[:, 1:-1, :]
        last_yaw_position = obs_yaw[:, 1:-1, :]

        prev_data = np.concatenate((last_position, newpos[:, :, 1:].reshape((1, 1, self.max_num_agents, 2))), axis=1)
        prev_data = np.reshape(prev_data, (1, self.observed_length, self.max_num_agents, 2))
        # prev_vel = np.reshape(last_v_position, (1, 1, self.max_num_agents))
        prev_yaw = np.concatenate((last_yaw_position, newyaw.reshape((1, 1, self.max_num_agents, 2))), axis=1)
        prev_yaw = np.reshape(prev_yaw, (1, self.observed_length, self.max_num_agents, 2))
        prev_target_data = np.reshape(_x[0, 1:obs_traj.shape[1]], (1, self.observed_length, self.max_num_agents, self.input_dim-1))
        # prev_target_v_data = np.reshape(_v[obs_vel.shape[0]], (1, self.max_num_agents, self.input_dim))
        prev_target_yaw_data = np.reshape(_yaw[0, 1:obs_yaw.shape[1]], (1, self.observed_length, self.max_num_agents, self.input_dim-1))
        for t in range(self.predicted_length-1):
            feed = {
                self.input_data: np.concatenate((prev_data.reshape((1, self.observed_length, self.max_num_agents, 2)), prev_yaw.reshape((1, self.observed_length, self.max_num_agents, 2))), axis=3),
                self.inp_mask: inp_mask[:, 1 + t:self.observed_length+1+t, :].reshape((1, self.observed_length, self.max_num_agents)),
                self.target_data: np.concatenate((prev_target_data.reshape((1, self.observed_length, self.max_num_agents, 2)), prev_target_yaw_data.reshape((1, self.observed_length, self.max_num_agents, 2))), axis=3),
                self.target_mask: inp_mask[:, 2 + t:self.observed_length+2+t, :].reshape((1, self.observed_length, self.max_num_agents))
            }
            mux, muy, sx, sy, corr, output_rz, output_rw = self.sess.run(
                [self.mux, self.muy, self.sx, self.sy, self.corr, self.output_rz, self.output_rw], feed_dict=feed)

            newpos = np.zeros((1, self.max_num_agents, self.output_dim))
            newyaw = np.zeros((1, self.max_num_agents, 2))
            for a_index in range(self.max_num_agents):
                if a_index != 0:
                    if x_batch[0, self.observed_length-1, a_index, 1] != 0 and x_batch[0, self.observed_length-1, a_index, 2] != 0:
                        next_x, next_y = distributions.sample_2d_normal(mux[0, -1, a_index], muy[0, -1, a_index], sx[0, -1, a_index], sy[0, -1, a_index], corr[0, -1, a_index], self.rnd)
                        newpos[0, a_index, :] = [x_batch[0, self.observed_length-1, a_index, 0], next_x, next_y]
                        newyaw[0, a_index, :] = [output_rz[0, -1, a_index], output_rw[0, -1, a_index]]

            returned_traj = np.vstack((returned_traj, newpos))
            # prev_data = newpos
            prev_data = np.concatenate((prev_data[:, 1:, :, :], newpos[:, :, 1:].reshape((1, 1, self.max_num_agents, 2))), axis=1)
            prev_data = np.reshape(prev_data, (1, self.observed_length, self.max_num_agents, 2))
            # prev_vel = np.reshape(asdadsas, (1, 1, self.max_num_agents))
            prev_yaw = np.concatenate((prev_yaw[:, 1:, :, :], newyaw.reshape((1, 1, self.max_num_agents, 2))), axis=1)
            prev_yaw = np.reshape(prev_yaw, (1, self.observed_length, self.max_num_agents, 2))

            # if t != self.predicted_length - 1:
            #     prev_target_data = np.reshape(x_batch[:, obs_traj.shape[0] + t + 1, :, 1:], (1, self.max_num_agents, self.output_dim-1))
            #     prev_target_yaw_data = np.reshape(_yaw[:, obs_yaw.shape[0] + t + 1], (1, self.max_num_agents, self.output_dim-1))

        complete_traj = returned_traj
        if error_type == 'ade':
            answer = distributions.get_mean_error(
                predicted_traj=np.round(complete_traj, decimals=4),
                true_traj=np.round(x_batch[0], decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)
        elif error_type == 'fde':
            answer = distributions.get_final_error(
                predicted_traj=np.round(complete_traj, decimals=4),
                true_traj=np.round(x_batch[0], decimals=4),
                observed_length=self.observed_length,
                max_num_agents=self.max_num_agents)

        return answer, None, None, None, None

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

    # im_list = sorted(np.genfromtxt('../gaussian_observation_maps/data/robot/rss_new/setup_five/reduced_im_list.txt', dtype='str').tolist())
    # from scipy.misc import imresize as resize
    # import matplotlib.pyplot as plt
    # import cv2
    # im = cv2.imread("../gaussian_observation_maps/"+im_list[0][3:])

    # frame = im[20:-70, 56:-70]
    # frame = frame.astype(np.uint8)
    # frame = resize(frame, (64, 64))
    # frame = ((1.0 - frame) * 255).round().astype(np.uint8)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # plt.imshow(frame)
    # plt.plot(complete_traj[7:, 1, 2]*64, complete_traj[7:, 1, 1]*64, 'x', c='red', label='RDB')
    # plt.plot(complete_traj[:8, 1, 2]*64, complete_traj[:8, 1, 1]*64, 'x', c='green', label='observed')
    # plt.legend()
    # plt.show()

            complete_traj = np.copy(returned_traj)
            traj_distrib.append(np.copy(complete_traj))

        # ############## ROBOT DATASET IMAGE PRINTS START!!!! ##############

        traj_distrib = np.array(traj_distrib)
        # im_list = sorted(np.genfromtxt('../gaussian_observation_maps/data/robot/rss_new/setup_five/reduced_im_list.txt', dtype='str').tolist())
        # # im_list = sorted(np.genfromtxt('../gaussian_observation_maps/data/robot/rss_new/setup_five/complete_im_list.txt', dtype='str').tolist())
        # from scipy.misc import imresize as resize
        # import matplotlib.pyplot as plt
        # import cv2

        # # img = cv2.imread("../gaussian_observation_maps/"+im_list[int(kwargs['frame_id'][0])+4*self.observed_length][3:])
        # img = cv2.imread("../gaussian_observation_maps/"+im_list[0][3:])

        # # from models.info_vae import InfoVAE
        # # vae = InfoVAE(z_size=96,
        # #     batch_size=1,
        # #     learning_rate=0.001,
        # #     is_training=False,
        # #     reuse=False,
        # #     gpu_mode=False) # use GPU on batchsize of 1000 -> much faster

        # # vae.load_json('save/info_vae/OVERFIT_info_vae_reduced_setup_five_processing_50.json')
        # # img = decode_this[0]

        # # img = vae.decode(img.reshape(1, 96)) * 255.
        # # img = np.round(img).astype(np.uint8)
        # # img = img.reshape(1, 64, 64, 3)

        # # plt.imshow(img[0])
        # # plt.axis('off')
        # # plt.savefig('results/images/'+str(int(kwargs['frame_id'][0]))+"_justim.jpg", pad_inches=0)

        # frame = img[20:-70, 56:-70]
        # frame = frame.astype(np.uint8)
        # frame = resize(frame, (64, 64))
        # frame = ((1.0 - frame) * 255).round().astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # import seaborn as sns
        # clrs = sns.color_palette("husl", self.predicted_length)
        # sns.set_style("dark")
        # with sns.axes_style("dark"):
        #     plt.imshow(frame)
        #     mean_traj = np.mean(traj_distrib, axis=0)
        #     plt.plot(traj_distrib[:, 8:, 1, 2]*64, traj_distrib[:, 8:, 1, 1]*64, 'x', c='red', alpha=0.15)
        #     plt.plot(mean_traj[8:, 1, 2]*64, mean_traj[8:, 1, 1]*64, 'x', c='red', alpha=0.75, label='RDB')
        #     plt.plot(traj_distrib[0, :8, 1, 2]*64, traj_distrib[0, :8, 1, 1]*64, 'x', c='green', label='observed')
        #     plt.legend()
        #     plt.axis('off')
        #     plt.savefig('results/singlenorm/'+str(int(kwargs['frame_id'][0]))+".svg", pad_inches=0)
        #     plt.clf()
        
        # Final Used Plots ETH + UNIV
        # agents_to_use = []
        # up_to_idx = []
        # for agent in range(self.max_num_agents):
        #     if traj_distrib[0, self.observed_length, agent, 0] != 0.0:
        #         agents_to_use.append(agent)
        #         stop = False
        #         added = False
        #         for it, p in enumerate(x_batch[self.observed_length:, agent, :]):
        #             if p[0] == 0 and not stop:
        #                 stop = True
        #                 added = True
        #                 up_to_idx.append(self.observed_length+it)
        #         if not added:
        #             up_to_idx.append(self.observed_length + self.predicted_length)

        # import matplotlib.pyplot as plt
        # import cv2
        # from scipy.misc import imresize as resize
        # POS_MSEC = cv2.CAP_PROP_POS_MSEC
        # POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
        # videopath_dir = {
        #     'univ': 'data/eth/univ/seq_eth.avi', 
        #     'hotel': 'data/eth/hotel/seq_hotel.avi', 
        #     'zara01': 'data/ucy/zara/zara01/crowds_zara01.avi',
        #     'zara02': 'data/ucy/zara/zara02/crowds_zara02.avi'}

        # name = 'zara02'
        # height = 576
        # width = 720
        # # height = 480
        # # width = 640
        # blocking = False

        # videopath = videopath_dir[name]
        # captured = cv2.VideoCapture(videopath)
        # captured.set(POS_FRAMES, int(frame_id))
        # frame_num = int(captured.get(POS_FRAMES))
        # now = int(captured.get(POS_MSEC) / 1000)
        # _, frame = captured.read()
        # # frame = frame.astype(np.uint8)

        # # frame = np.int16(frame)
        # # frame = frame * (50/127+1) - 50 + 70
        # # frame = np.clip(frame, 0, 255)
        # frame = resize(frame, (height, width))
        # # frame = ((1.0 - frame) * 255).round().astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # import seaborn as sns
        # clrs = sns.color_palette("husl", self.predicted_length)
        # sns.set_style("dark")
        # with sns.axes_style("dark"):
        #     plt.imshow(frame)
        #     for idx, agent in enumerate(agents_to_use):
        #         if blocking:
        #             # with kind of working up_to_idx
        #             plt.plot(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 2]*width, traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='#00ff00', alpha=0.1)
        #             plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan')
        #             plt.plot(x_batch[self.observed_length:up_to_idx[idx], agent, 2]*width, x_batch[self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='yellow')
        #         elif not blocking:
        #             # without the up_to_idx
        #             plt.plot(traj_distrib[:, self.observed_length:, agent, 2]*width, traj_distrib[:, self.observed_length:, agent, 1]*height, 'x', c='#00ff00', alpha=0.1)
        #             plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan')
        #             plt.plot(x_batch[self.observed_length:, agent, 2]*width, x_batch[self.observed_length:, agent, 1]*height, 'x', c='yellow')

        #     if blocking:
        #         plt.plot(np.mean(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 2]*width, axis=0), np.mean(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 1]*height, axis=0), 'x', c='#00ff00', label='RDB')
        #         plt.plot(x_batch[self.observed_length:up_to_idx[idx], agent, 2]*width, x_batch[self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='yellow', label='GT')
        #     elif not blocking:
        #         plt.plot(np.mean(traj_distrib[:, self.observed_length:, agent, 2]*width, axis=0), np.mean(traj_distrib[:, self.observed_length:, agent, 1]*height, axis=0), 'x', c='#00ff00', label='RDB')
        #         plt.plot(x_batch[self.observed_length:, agent, 2]*width, x_batch[self.observed_length:, agent, 1]*height, 'x', c='yellow', label='GT')

        #     plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan', label='Observed')
        #     plt.legend()
        #     plt.axis('off')
        #     if blocking:
        #         plt.savefig('results/' + name + '_rd/'+str(int(kwargs['frame_id'][0]))+".svg", pad_inches=0)
        #     elif not blocking:
        #         plt.savefig('results/' + name + '_rd_no_blocks/'+str(int(kwargs['frame_id'][0]))+".svg", pad_inches=0)
        #     plt.clf()

        # END

        # ############## ROBOT DATASET IMAGE PRINTS END!!!! ##############
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

    # def conditionedInfer(self, x_batch, **kwargs):
    #     obs_traj = np.copy(x_batch[:self.observed_length])
    #     states = kwargs['initial_states']
    #     # stupid_states = kwargs['initial_states'].copy()
    #     latent_sequence = kwargs['latent_sequence']
    #     error_type = kwargs['error_type']
    #     returned_states = np.zeros((self.max_num_agents, states.shape[1]))

    #     # ### NEED TO REMOVE!!!
    #     # for idx, position in enumerate(x_batch[:-1]):
    #     #     input_data_tensor = np.reshape(position, (1, self.max_num_agents, self.input_dim))
    #     #     target_data_tensor = np.reshape(x_batch[idx+1], (1, self.max_num_agents, self.input_dim))
    #     #     extra_data_tensor = np.zeros((1, self.extra_dim), dtype=np.float32)
    #     #     extra_data_tensor[0, :] = latent_sequence[idx]

    #     #     feed = {
    #     #         self.input_data: input_data_tensor,
    #     #         self.extra_data: extra_data_tensor,
    #     #         self.target_data: target_data_tensor,
    #     #         self.LSTM_states: stupid_states
    #     #         }

    #     #     [stupid_states] = self.sess.run([self.final_states], feed)

    #     for idx, position in enumerate(obs_traj[:-1]):
    #         input_data_tensor = np.reshape(position, (1, self.max_num_agents, self.input_dim))
    #         target_data_tensor = np.reshape(obs_traj[idx+1], (1, self.max_num_agents, self.input_dim))
    #         extra_data_tensor = np.zeros((1, self.extra_dim), dtype=np.float32)
    #         extra_data_tensor[0, :] = latent_sequence[idx]

    #         feed = {
    #             self.input_data: input_data_tensor,
    #             self.extra_data: extra_data_tensor,
    #             self.target_data: target_data_tensor,
    #             self.LSTM_states: states
    #             }

    #         [states] = self.sess.run([self.final_states], feed)

    #     returned_traj = obs_traj
    #     last_position = obs_traj[-1]
    #     last_extra = latent_sequence[obs_traj.shape[0]-1] # think if that's correct ..

    #     returned_states = states.copy()

    #     # prev_data = np.reshape(last_position, (1, self.max_num_agents, self.input_dim))
    #     # prev_extra = np.reshape(last_extra, (1, self.extra_dim)) #96, 768
    #     # prev_target_data = np.reshape(x_batch[obs_traj.shape[0]], (1, self.max_num_agents, self.input_dim))

    #     idcs = []
    #     motion = []
    #     onlys = [False for _ in range(self.max_num_agents)]
    #     # I only use the perdictions of agents from the last part of the observed_length
    #     for yo, agent in enumerate(x_batch[0]):
    #         if agent[1] == 0.0 and agent[2] == 0.0:
    #             continue
            
    #         if x_batch[0][yo][1] == 0.0 and x_batch[0][yo][2] == 0.0:
    #             continue

    #         onlys[yo] = True
    #         idcs.append(yo)

    #     starts = [-1 for i in range(self.max_num_agents)]
    #     for t in range(x_batch.shape[0]):
    #         for a_idx, a_output in enumerate(x_batch[t, :]):
    #             if a_output[0] == 0.0 or a_idx not in idcs:
    #                 continue

    #             if a_output[1] != 0 or a_output[2] != 0:
    #                 if starts[a_idx] == -1:
    #                     starts[a_idx] = t

    #     agents_in = [idx for idx, agent_loc in enumerate(starts) if agent_loc != -1]
    #     trajs = []
    #     for agent_id in agents_in:
    #         if agent_id == 0:
    #             continue

    #         trajs.append(x_batch[starts[int(agent_id)]:, int(agent_id)])

    #     full_trajs = []
    #     for traj in trajs:
    #         temp = []
    #         for pos in traj:
    #             if pos[1] != 0.0 and pos[2] != 0.0:
    #                 temp.append(pos)

    #         full_trajs.append(temp)

    #     velocities = [[0, 0] for i in range(self.max_num_agents)]
    #     for a_id, tr in enumerate(full_trajs):
    #         for idx, pos in enumerate(tr[1:]):
    #             if idx == 0:
    #                 continue

    #             velocities[idcs[a_id]][0] = pos[1] - tr[idx-1][1]
    #             velocities[idcs[a_id]][1] = pos[2] - tr[idx-1][2]
 
    #     complete_traj = returned_traj
    #     if error_type == 'ade':
    #         answer = distributions.get_mean_error(
    #             predicted_traj=np.round(complete_traj, decimals=4),
    #             true_traj=np.round(x_batch, decimals=4),
    #             observed_length=self.observed_length,
    #             max_num_agents=self.max_num_agents)
    #     elif error_type == 'fde':
    #         answer = distributions.get_final_error(
    #             predicted_traj=np.round(complete_traj, decimals=4),
    #             true_traj=np.round(x_batch, decimals=4),
    #             observed_length=self.observed_length,
    #             max_num_agents=self.max_num_agents)

    #     complete_traj = returned_traj
    #     if error_type == 'ade':
    #         answer = distributions.get_mean_error(
    #             predicted_traj=np.round(complete_traj, decimals=4), 
    #             true_traj=np.round(x_batch, decimals=4),
    #             observed_length=self.observed_length,
    #             max_num_agents=self.max_num_agents)
    #     elif error_type == 'fde':
    #         answer = distributions.get_final_error(
    #             predicted_traj=np.round(complete_traj, decimals=4), 
    #             true_traj=np.round(x_batch, decimals=4),
    #             observed_length=self.observed_length,
    #             max_num_agents=self.max_num_agents)

    #     return answer, [state for idx, state in enumerate(returned_states) if idx in idcs], [x_batch[:, i, 1:] for i in range(self.max_num_agents) if i in idcs], [velocity for idx, velocity in enumerate(velocities) if idx in idcs], None


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

                    # if prev_data[0, a_index, 0] != 0:
                    #     print("Pedestrian ID", prev_data[0, a_index, 0])
                    #     print("Predicted parameters", mux, muy, sx, sy, corr)
                    #     print("New Position", next_x, next_y)
                    #     print("Target Position", prev_target_data[0, a_index, 1], prev_target_data[0, a_index, 2])
                    #     print()
                
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
        
        # agents_to_use = []
        # up_to_idx = []
        # for agent in range(self.max_num_agents):
        #     if traj_distrib[0, self.observed_length, agent, 0] != 0.0:
        #         agents_to_use.append(agent)
        #         stop = False
        #         added = False
        #         for it, p in enumerate(x_batch[self.observed_length:, agent, :]):
        #             if p[0] == 0 and not stop:
        #                 stop = True
        #                 added = True
        #                 up_to_idx.append(self.observed_length+it)
        #         if not added:
        #             up_to_idx.append(self.observed_length + self.predicted_length)

        # # print("Consider agents: {0} up to the following indices: {1}".format(agents_to_use, up_to_idx))

        # import matplotlib.pyplot as plt
        # import cv2
        # from scipy.misc import imresize as resize
        # POS_MSEC = cv2.CAP_PROP_POS_MSEC
        # POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
        # videopath_dir = {
        #     'univ': 'data/eth/univ/seq_eth.avi', 
        #     'hotel': 'data/eth/hotel/seq_hotel.avi', 
        #     'zara01': 'data/ucy/zara/zara01/crowds_zara01.avi',
        #     'zara02': 'data/ucy/zara/zara02/crowds_zara02.avi'}

        # name = 'hotel'
        # # height = 480
        # # width = 640
        # height = 576
        # width = 720
        # blocking = False

        # videopath = videopath_dir[name]
        # captured = cv2.VideoCapture(videopath)
        # captured.set(POS_FRAMES, int(frame_id))
        # frame_num = int(captured.get(POS_FRAMES))
        # now = int(captured.get(POS_MSEC) / 1000)
        # _, frame = captured.read()

        # if frame_id == 101:
        #     print("tuka sam!")
        # # frame = frame.astype(np.uint8)

        # # frame = np.int16(frame)
        # # frame = frame * (50/127+1) - 50 + 70
        # # frame = np.clip(frame, 0, 255)
        # frame = resize(frame, (height, width))
        # # frame = ((1.0 - frame) * 255).round().astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # import seaborn as sns
        # clrs = sns.color_palette("husl", self.predicted_length)
        # sns.set_style("dark")
        # with sns.axes_style("dark"):
        #     plt.imshow(frame)
        #     # for idx, agent in enumerate(agents_to_use):
        #     #     if blocking:
        #     #         # with kind of working up_to_idx
        #     #         plt.plot(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 2]*width, traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='#ff00ff', alpha=0.1)
        #     #         plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan')
        #     #         plt.plot(x_batch[self.observed_length:up_to_idx[idx], agent, 2]*width, x_batch[self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='yellow')
        #     #     elif not blocking:
        #     #         # without
        #     #         plt.plot(traj_distrib[:, self.observed_length:, agent, 2]*width, traj_distrib[:, self.observed_length:, agent, 1]*height, 'x', c='#ff00ff', alpha=0.1)
        #     #         plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan')
        #     #         plt.plot(x_batch[self.observed_length:, agent, 2]*width, x_batch[self.observed_length:, agent, 1]*height, 'x', c='yellow')

        #     # if blocking:
        #     #     plt.plot(np.mean(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 2]*width, axis=0), np.mean(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 1]*height, axis=0), 'x', c='#ff00ff', label='Social')
        #     #     plt.plot(x_batch[self.observed_length:up_to_idx[idx], agent, 2]*width, x_batch[self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='yellow', label='GT')
        #     # elif not blocking:
        #     #     plt.plot(np.mean(traj_distrib[:, self.observed_length:, agent, 2]*width, axis=0), np.mean(traj_distrib[:, self.observed_length:, agent, 1]*height, axis=0), 'x', c='#ff00ff', label='Social')
        #     #     plt.plot(x_batch[self.observed_length:, agent, 2]*width, x_batch[self.observed_length:, agent, 1]*height, 'x', c='yellow', label='GT')

        #     # plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan', label='Observed')
        #     # plt.legend()
        #     plt.axis('off')
        #     if blocking:
        #         plt.savefig('results/' + name + '_social/'+str(int(kwargs['frame_id'][0]))+".svg", pad_inches=0)
        #     elif not blocking:
        #         plt.savefig('results/' + name + '_pure_no_blocks/'+str(int(kwargs['frame_id'][0]))+".svg", pad_inches=0)
        #     plt.clf()
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
            # print(states.shape)
            # print("---------")
            # returned_states = np.zeros((states.shape[1]))
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


        #  plt.imshow(img[0])
        # plt.savefig('results/images/'+str(int(kwargs['frame_id'][0]))+"_justim.jpg")

        # frame = im[20:-70, 56:-70]
        # frame = frame.astype(np.uint8)
        # frame = resize(frame, (64, 64))
        # frame = ((1.0 - frame) * 255).round().astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # im_list = sorted(np.genfromtxt('../gaussian_observation_maps/data/robot/rss_new/setup_five/reduced_im_list.txt', dtype='str').tolist())
        # from scipy.misc import imresize as resize
        # import matplotlib.pyplot as plt
        # import cv2
        # im = cv2.imread("../gaussian_observation_maps/"+im_list[int(kwargs['frame_id'][0])+4*self.observed_length][3:])
        # # im = cv2.imread('results/images/'+str(int(kwargs['frame_id'][0]))+"_justim.jpg")

        # frame = im[20:-70, 56:-70]
        # frame = frame.astype(np.uint8)
        # frame = resize(frame, (64, 64))
        # frame = ((1.0 - frame) * 255).round().astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        traj_distrib = np.array(traj_distrib)
        # plt.imshow(frame)
        # mean_traj = np.mean(traj_distrib, axis=0)
        # plt.plot(mean_traj[8:, 1, 2]*64, mean_traj[8:, 1, 1]*64, 'x', c='blue', alpha=0.75, label='LSTM')
        # plt.plot(traj_distrib[:, 8:, 1, 2]*64, traj_distrib[:, 8:, 1, 1]*64, 'x', c='blue', alpha=0.15)
        # plt.plot(traj_distrib[0, :8, 1, 2]*64, traj_distrib[0, :8, 1, 1]*64, 'x', c='green', label='observed')
        # plt.legend()

        # plt.axis('off')
        # plt.savefig('results/singlenorm/'+str(int(kwargs['frame_id'][0]))+".jpg")
        # plt.clf()

        # plt.imshow(frame)
        # plt.plot(x_batch[7:, 1, 2]*64, x_batch[7:, 1, 1]*64, 'x', c='orange', label='GT')
        # plt.plot(complete_traj[:8, 1, 2]*64, complete_traj[:8, 1, 1]*64,'x', c='green', label='observed')
        # plt.legend()
        # plt.axis('off')
        # plt.savefig('results/GT/'+str(int(kwargs['frame_id'][0]))+"_gt.jpg")
        # plt.clf()

        # agents_to_use = []
        # up_to_idx = []
        # for agent in range(self.max_num_agents):
        #     if traj_distrib[0, self.observed_length, agent, 0] != 0.0:
        #         agents_to_use.append(agent)
        #         stop = False
        #         added = False
        #         for it, p in enumerate(x_batch[self.observed_length:, agent, :]):
        #             if p[0] == 0 and not stop:
        #                 stop = True
        #                 added = True
        #                 up_to_idx.append(self.observed_length+it)
        #         if not added:
        #             up_to_idx.append(self.observed_length + self.predicted_length)

        # import matplotlib.pyplot as plt
        # import cv2
        # from scipy.misc import imresize as resize
        # POS_MSEC = cv2.CAP_PROP_POS_MSEC
        # POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
        # videopath_dir = {
        #     'univ': 'data/eth/univ/seq_eth.avi', 
        #     'hotel': 'data/eth/hotel/seq_hotel.avi', 
        #     'zara01': 'data/ucy/zara/zara01/crowds_zara01.avi',
        #     'zara02': 'data/ucy/zara/zara02/crowds_zara02.avi'}
        
        # name = 'zara02'
        # # height = 480#576#480
        # # width = 640#720#640
        # height = 576#480
        # width =  720#640
        # blocking = False

        # videopath = videopath_dir[name]
        # captured = cv2.VideoCapture(videopath)
        # captured.set(POS_FRAMES, int(frame_id))
        # frame_num = int(captured.get(POS_FRAMES))
        # now = int(captured.get(POS_MSEC) / 1000)
        # _, frame = captured.read()
        # # frame = frame.astype(np.uint8)

        # # frame = np.int16(frame)
        # # frame = frame * (50/127+1) - 50 + 70
        # # frame = np.clip(frame, 0, 255)
        # frame = resize(frame, (height, width))
        # # frame = ((1.0 - frame) * 255).round().astype(np.uint8)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # import seaborn as sns
        # clrs = sns.color_palette("husl", self.predicted_length)
        # sns.set_style("dark")
        # with sns.axes_style("dark"):
        #     plt.imshow(frame)
        #     for idx, agent in enumerate(agents_to_use):
        #         if blocking:
        #             # with kind of working up_to_idx
        #             plt.plot(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 2]*width, traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='red', alpha=0.1)
        #             plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan')
        #             plt.plot(x_batch[self.observed_length:up_to_idx[idx], agent, 2]*width, x_batch[self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='yellow')
        #         elif not blocking:
        #             # without
        #             plt.plot(traj_distrib[:, self.observed_length:, agent, 2]*width, traj_distrib[:, self.observed_length:, agent, 1]*height, 'x', c='red', alpha=0.1)
        #             plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan')
        #             plt.plot(x_batch[self.observed_length:, agent, 2]*width, x_batch[self.observed_length:, agent, 1]*height, 'x', c='yellow')

        #     if blocking:
        #         plt.plot(np.mean(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 2]*width, axis=0), np.mean(traj_distrib[:, self.observed_length:up_to_idx[idx], agent, 1]*height, axis=0), 'x', c='red', label='LSTM')
        #         plt.plot(x_batch[self.observed_length:up_to_idx[idx], agent, 2]*width, x_batch[self.observed_length:up_to_idx[idx], agent, 1]*height, 'x', c='yellow', label='GT')
        #     elif not blocking:
        #         plt.plot(np.mean(traj_distrib[:, self.observed_length:, agent, 2]*width, axis=0), np.mean(traj_distrib[:, self.observed_length:, agent, 1]*height, axis=0), 'x', c='red', label='LSTM')
        #         plt.plot(x_batch[self.observed_length:, agent, 2]*width, x_batch[self.observed_length:, agent, 1]*height, 'x', c='yellow', label='GT')

        #     plt.plot(traj_distrib[0, :self.observed_length, agent, 2]*width, traj_distrib[0, :self.observed_length, agent, 1]*height, 'x', c='cyan', label='Observed')
        #     plt.legend()
        #     plt.axis('off')
        #     if blocking:
        #         plt.savefig('results/' + name + '_just/'+str(int(kwargs['frame_id'][0]))+".svg", pad_inches=0)
        #     elif not blocking:
        #         plt.savefig('results/' + name + '_just_no_blocks/'+str(int(kwargs['frame_id'][0]))+".svg", pad_inches=0)
        #     plt.clf()

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

    def inferClassif(self, x_batch, labels, **kwargs):
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
            input_data_tensor = np.reshape(position, (1, self.max_num_agents, self.input_dim))
            target_data_tensor = np.reshape(obs_traj[idx+1], (1, self.max_num_agents, self.input_dim))
            feed = {self.input_data: input_data_tensor, self.target_data: target_data_tensor, self.target_labels:labels, self.LSTM_states: states}
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
        batch_acc = []

        for t in range(self.predicted_length):
            feed = {
                self.input_data : prev_data,
                self.LSTM_states: states,
                self.target_data: prev_target_data,
                self.target_labels:labels}
            [output, predictions, corrects, accuracy, mask, traj_len, states] = self.sess.run(
                [self.final_output, self.predictions, self.correct_pred, self.accuracy, self.mask, self.traj_len, self.final_states], feed)

            # handle cases when no classification is done..
            if accuracy == accuracy:
                batch_acc.append(accuracy)
            # else:
            #     print("tuk sam")

            newpos = np.zeros((1, self.max_num_agents, self.output_dim))
            for a_index, a_output in enumerate(output):
                if prev_data[0][a_index][0] != 0 and prev_target_data[0][a_index][0] != 0:
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

        return answer, [state for idx, state in enumerate(returned_states) if idx in idcs], [x_batch[:, i, 1:] for i in range(self.max_num_agents) if i in idcs], [velocity for idx, velocity in enumerate(velocities) if idx in idcs], np.mean(batch_acc) if len(batch_acc) > 0 else -1
