import numpy as np

from utils.params import DataLoadParams


class DataLoader(object):
    __slots__ = ['data', 'rnd']

    def __init__(self, data, rnd):
        self.data = data
        self.rnd = rnd

    def __len__(self):
        '''len() returns the number of batches'''
        if self.args.mode != 'disproportional_infer' and self.args.mode != 'infer':
            counter = np.sum([int(len(data) / (self.args.sequence_length + 1)) for data in self.data.all_data])
            # On an average, we need twice the number of batches to cover the data
            # due to randomization introduced: https://github.com/vvanirudh/social-lstm-tf
            return 2 * int((counter / self.args.batch_size))
            # return int((counter / self.args.batch_size))
        elif self.args.mode == 'infer':
            counter = np.sum([int(len(data) - (self.args.sequence_length + 1)) for data in self.data.all_data])
            return int((counter / self.args.batch_size))
        else:
            # this results in testing against a non-proportional data size compared to training data
            # this is due to the number of samples considered. Ideally, I need to use the same length as in
            # training data and evaluate by sliding the window by one ... it will still be proportionally more
            # but not as much ... Results here do not differ too much but its still not reasonable.
            counter = np.sum([int(len(data)) for data in self.data.all_data])
            return int((counter / self.args.batch_size))

    @property
    def args(self):
        return self.__args

    @args.setter
    def args(self, args):
        self.__args = DataLoadParams(
            dataset_names=args.dataset_names,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            mode=args.mode,
            data_type=args.data_type)

    def reset(self):
        self.frame_pointer = 0
        self.dataset_pointer = 0

class PerFrameLoader(DataLoader):
    def __init__(self, data, rnd):
        super(PerFrameLoader, self).__init__(data, rnd)

    @property
    def frame_pointer(self):
        return self.data.frame_pointer

    @frame_pointer.setter
    def frame_pointer(self, value):
        if self.args.mode == 'train' and value != 0:
            # self.data.frame_pointer = self.rnd.randint(self.data.frame_pointer+1, value)
            self.data.frame_pointer = value
        else:
            self.data.frame_pointer = value

    @property
    def dataset_pointer(self):
        return self.data.dataset_pointer

    @dataset_pointer.setter
    def dataset_pointer(self, value):
        # iadd assigns the value the dataset + value to value
        self.data.dataset_pointer = value
        # Set the frame pointer to zero for the current dataset
        self.frame_pointer = 0
        # If all datasets are done, then go to the first one again
        if self.data.dataset_pointer >= len(self.data.all_data):
            self.data.dataset_pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # add a test setting ...
        # self.unitTest()
        counter = 0
        x_batch, y_batch, d_batch, f_batch, l_batch, v_batch, yaw_batch = [], [], [], [], [], [], []
        firsts_batch = np.zeros((self.args.batch_size, self.data.args.sequence_length + 1, self.data.args.max_num_agents, self.data.args.input_dim))
        while counter < self.args.batch_size:
            frame_data = self.data.all_data[self.data.dataset_pointer]
            frame_ids = self.data.frames_list[self.data.dataset_pointer]
            if self.frame_pointer + self.args.sequence_length < frame_data.shape[0]:
                input_data, target_data, velocity_data, yaw_data, fid_data = next(self.data)
                labels_data = self.label(velocity_data)
                if self.args.data_type == 'location_invariant':
                    full_traj = np.concatenate((input_data, target_data[-1, :, :].reshape((1, self.data.args.max_num_agents, self.data.args.input_dim))))
                    zapomni, real_full_traj = self.loc_inv(full_traj)
                    input_data = zapomni[:-1, :, :]
                    target_data = zapomni[1:, :, :]
                    firsts_batch[counter] = real_full_traj

                elif self.args.data_type == 'previous_relative':
                    # use the first point for both pred and traj! but then
                    # first targ point is different from the second one ... ?!
                    full_traj = np.concatenate((input_data, target_data[-1, :, :].reshape((1, self.data.args.max_num_agents, self.data.args.input_dim))))
                    zapomni, real_full_traj = self.prev_rel(full_traj)
                    input_data = zapomni[:-1, :, :]
                    target_data = zapomni[1:, :, :]
                    firsts_batch[counter] = real_full_traj

                x_batch.append(input_data)
                y_batch.append(target_data)
                l_batch.append(labels_data)
                f_batch.append(fid_data)
                v_batch.append(velocity_data)
                yaw_batch.append(yaw_data)
                counter += 1

                self.frame_pointer += 1 if self.args.mode == 'infer' else self.rnd.randint(1, self.args.sequence_length)
                # self.frame_pointer += self.rnd.randint(1, self.args.sequence_length)
                d_batch.append(self.args.dataset_names[self.data.dataset_pointer])
            else:
                # update dataset pointer
                self.dataset_pointer += 1

        return {
            'inputs': np.array(x_batch),
            'targets': np.array(y_batch),
            'labels': np.array(l_batch),
            'velocity': np.array(v_batch),
            'yaw': np.array(yaw_batch),
            'dataset_names': np.array(d_batch),
            'frames_ids': np.array(f_batch),
            'firsts': firsts_batch}

    def label(self, velocities):
        total_velocity = np.sum(velocities[:, :, 1:], axis=0)
        labels = [[0.0, 0.0, 1.0] for _ in range(total_velocity.shape[0])]
        # labels = [[0.0, 1.0] for _ in range(total_velocity.shape[0])]
        for agent, vel in enumerate(total_velocity):
            # needs further fine tuning ..
            if (vel[0] < -0.009 or vel[0] > 0.009) or (vel[1] < -0.009 or vel[1] > 0.009):
                if vel[0] + vel[1] <= 0:
                    labels[agent] = [0.0, 1.0, 0.0]
                else:
                    labels[agent] = [1.0, 0.0, 0.0]
                # labels[agent] = [1.0, 0.0]

        return labels

    def loc_inv(self, full_traj):
        real_full_traj = np.zeros(full_traj.shape)
        zapomni = np.zeros(full_traj.shape)
        parts_traj = np.zeros(full_traj.shape)
        agenti = np.arange(self.data.args.max_num_agents)
        for aid in range(full_traj.shape[1]):
            if aid == 0:
                continue

            # bools = full_traj[:, aid, 0] == aid
            # seq = full_traj[full_traj[:, aid, 0] == aid, aid, :]
            seq = full_traj[:, aid, :]
            s_idcs = []
            e_idcs = []
            start = False
            for idx, pt in enumerate(seq):
                # the idea is that interpolation happens between two points
                # first one is the nset to 0 if there is a zero-pt prediction
                # it is not very useful, we also need to have some small history ~ 1-2 pts
                # SocialGAN uses 4 = 10 at least!
                # not sure why I do this here ...
                if idx < len(seq)-4 and not start and pt[1] != 0.0 and pt[2] != 0.0:
                    s_idcs.append(idx)
                    start = True
                elif pt[1] == 0.0 and pt[2] == 0.0 and start:
                    e_idcs.append(idx)
                    start = False
            if start == True and s_idcs[-1] != idx and len(s_idcs) > len(e_idcs):
                e_idcs.append(idx+1)
                start = False
            for pts in zip(s_idcs, e_idcs):
                if pts[1] - pts[0] >= 4:
                    cur_seq = seq[pts[0]:pts[1]]
                    real_full_traj[pts[0]:pts[1], aid, :] = np.copy(cur_seq[0])
                    parts_traj[pts[0], aid, 1:] = cur_seq[0, 1:]
                    cur_seq[:, 1:] = np.subtract(cur_seq[:, 1:], parts_traj[pts[0], aid, 1:])
                    cur_seq = np.around(cur_seq, decimals=4)
                    zapomni[pts[0]:pts[1], aid, :] = cur_seq

        zapomni[:, :, 0] = agenti
        real_full_traj[:, :, 0] = agenti
        return zapomni, real_full_traj

    def prev_rel(self, full_traj):
        real_full_traj = np.zeros(full_traj.shape)
        zapomni = np.zeros(full_traj.shape)
        parts_traj = np.zeros(full_traj.shape)
        agenti = np.arange(self.data.args.max_num_agents)
        for aid in range(full_traj.shape[1]):
            if aid == 0:
                continue

            # bools = full_traj[:, aid, 0] == aid
            # seq = full_traj[full_traj[:, aid, 0] == aid, aid, :]
            seq = full_traj[:, aid, :]
            s_idcs = []
            e_idcs = []
            start = False
            for idx, pt in enumerate(seq):
                # note that 4 is an empirically set number
                # the idea is that interpolation happesn between two points
                # first one is the nset to 0 if there is a zero-pt prediction
                # it is not very useful, we also need to have some small history ~ 1-2 pts
                # SocialGAN uses 4 = 10 at least!
                if idx < len(seq)-1 and not start and pt[1] != 0.0 and pt[2] != 0.0:
                    s_idcs.append(idx)
                    start = True
                elif pt[1] == 0.0 and pt[2] == 0.0 and start:
                    e_idcs.append(idx)
                    start = False
            if start == True and s_idcs[-1] != idx and len(s_idcs) > len(e_idcs):
                e_idcs.append(idx+1)
                start = False
            for pts in zip(s_idcs, e_idcs):
                # needs to be bigger than the prediction length
                # otherwise it wouldn't make sense ..
                if pts[1] - pts[0] >= self.args.sequence_length:
                    cur_seq = seq[pts[0]:pts[1]]
                    real_full_traj[pts[0]:pts[1], aid, :] = np.copy(cur_seq)
                    parts_traj[pts[0]:pts[1]-1, aid, 1:] = cur_seq[:-1, 1:]
                    # get velocity
                    cur_seq[1:, 1:] = cur_seq[1:, 1:] - parts_traj[pts[0]:pts[1]-1, aid, 1:]
                    cur_seq[0, 1:] = 0.0
                    cur_seq = np.around(cur_seq, decimals=4)
                    if np.array_equal(cur_seq[:, 1:], np.zeros(cur_seq[:, 1:].shape)):
                        real_full_traj[pts[0]:pts[1], aid, :] = np.copy(cur_seq)
                    zapomni[pts[0]:pts[1], aid, :] = cur_seq

        zapomni[:, :, 0] = agenti
        real_full_traj[:, :, 0] = agenti
        return np.round(zapomni, decimals=4), np.round(real_full_traj, decimals=4)

    def unitTest(self):
        # REVISIT ONCE using
        agenti = np.arange(40)
        test_traj = np.zeros((9, 40, 3))
        result_traj = np.zeros((9, 40, 3))

        # test_traj[0, 1, :] = [1, 0.25, 0.75]
        # test_traj[1, 1, :] = [1, 0.26, 0.74]
        # test_traj[2, 1, :] = [1, 0.27, 0.73]
        # test_traj[3, 1, :] = [1, 0.28, 0.72]
        # test_traj[4, 1, :] = [1, 0.29, 0.71]
        # test_traj[5, 1, :] = [1, 0.30, 0.70]

        # tempo = test_traj[test_traj[:, 1, 0] == 1, 1, :]
        # because the trajectory is smaller than 8!
        # result_traj[1:6, 1, 1:] = tempo[1:, 1:] - tempo[:-1, 1:]

        # test_traj[0, 2, :] = [2, 0.35, 0.65]
        # test_traj[1, 2, :] = [2, 0.36, 0.64]
        # test_traj[2, 2, :] = [2, 0.37, 0.63]
        # test_traj[3, 2, :] = [2, 0.38, 0.62]
        # test_traj[4, 2, :] = [2, 0.39, 0.61]
        # test_traj[5, 2, :] = [2, 0.30, 0.60]

        # tempo = test_traj[test_traj[:, 2, 0] == 2, 2, :]
        # because the trajectory is smaller than 8!
        # result_traj[1:6, 2, 1:] = tempo[1:, 1:] - tempo[:-1, 1:]

        # This scenario is assumed to be removed
        # during data preparation ... BUT I AM NOT SURE
        # (should be I think I checked a long time ago)!
        # test_traj[0, 3, :] = [3, 0.15, 0.65]
        # test_traj[1, 3, :] = [3, 0.16, 0.64]

        # test_traj[3, 3, :] = [3, 0.68, 0.32]
        # test_traj[4, 3, :] = [3, 0.69, 0.31]
        # test_traj[5, 3, :] = [3, 0.60, 0.30]
        # test_traj[6, 3, :] = [3, 0.61, 0.29]
        # test_traj[7, 3, :] = [3, 0.62, 0.29]

        # tempo = test_traj[test_traj[:, 3, 0] == 3, 3, :]
        # tempo = tempo[2:]
        # result_traj[4:8, 3, 1:] = tempo[1:, 1:] - tempo[:-1, 1:]
        # END

        test_traj[0, 3, :] = [3, 0.15, 0.65]
        test_traj[1, 3, :] = [3, 0.16, 0.64]
        test_traj[2, 3, :] = [3, 0.46, 0.44]
        test_traj[3, 3, :] = [3, 0.68, 0.32]
        test_traj[4, 3, :] = [3, 0.69, 0.31]
        test_traj[5, 3, :] = [3, 0.60, 0.30]
        test_traj[6, 3, :] = [3, 0.61, 0.29]
        test_traj[7, 3, :] = [3, 0.62, 0.29]

        tempo = test_traj[test_traj[:, 3, 0] == 3, 3, :]
        result_traj[1:8, 3, 1:] = tempo[1:, 1:] - tempo[:-1, 1:]

        test_traj[:, :, 0] = agenti
        result_traj[:, :, 0] = agenti
        result_traj = np.around(result_traj, decimals=4)

        processed_traj = self.prev_rel(np.copy(test_traj))[0]
        assert np.array_equal(processed_traj, result_traj)

        agenti = np.arange(40)
        test_traj = np.zeros((9, 40, 3))
        result_traj = np.zeros((9, 40, 3))

        test_traj[0, 1, :] = [1, 0.25, 0.75]
        test_traj[1, 1, :] = [1, 0.26, 0.74]
        test_traj[2, 1, :] = [1, 0.27, 0.73]
        test_traj[3, 1, :] = [1, 0.28, 0.72]
        test_traj[4, 1, :] = [1, 0.29, 0.71]
        test_traj[5, 1, :] = [1, 0.30, 0.70]


        first = test_traj[0, 1, :]
        tempo = test_traj[test_traj[:, 1, 0] == 1, 1, :]
        # I don't think this should even be considered .. as the dataset shouldnt contain..
        result_traj[:6, 1, 1:] = np.subtract(tempo[:, 1:], first[1:])

        test_traj[0, 2, :] = [2, 0.35, 0.65]
        test_traj[1, 2, :] = [2, 0.36, 0.64]
        test_traj[2, 2, :] = [2, 0.37, 0.63]
        test_traj[3, 2, :] = [2, 0.38, 0.62]
        test_traj[4, 2, :] = [2, 0.39, 0.61]
        test_traj[5, 2, :] = [2, 0.30, 0.60]

        first = test_traj[0, 2, :]
        tempo = test_traj[test_traj[:, 2, 0] == 2, 2, :]
        result_traj[:6, 2, 1:] = np.subtract(tempo[:, 1:], first[1:])

        test_traj[0, 3, :] = [3, 0.15, 0.65]
        test_traj[1, 3, :] = [3, 0.16, 0.64]

        test_traj[3, 3, :] = [3, 0.68, 0.32]
        test_traj[4, 3, :] = [3, 0.69, 0.31]
        test_traj[5, 3, :] = [3, 0.60, 0.30]
        test_traj[6, 3, :] = [3, 0.61, 0.29]
        test_traj[7, 3, :] = [3, 0.62, 0.29]

        first = test_traj[3, 3, :]
        tempo = test_traj[test_traj[:, 3, 0] == 3, 3, :]
        tempo = tempo[2:]
        result_traj[3:8, 3, 1:] = np.subtract(tempo[:, 1:], first[1:])

        test_traj[:, :, 0] = agenti
        result_traj[:, :, 0] = agenti
        result_traj = np.around(result_traj, decimals=4)
        loc_inv_res = self.loc_inv(np.copy(test_traj))[0]
        assert np.array_equal(loc_inv_res, result_traj)

        print("Loader's tests passed!")

class PerAgentLoader(DataLoader):
    def __init__(self, data, rnd):
        super(PerAgentLoader, self).__init__(data, rnd)

    @property
    def frame_pointer(self):
        return self.data.frame_pointer

    @frame_pointer.setter
    def frame_pointer(self, value):
        self.data.frame_pointer = value
        if self.data.frame_pointer >= len(self.data.all_data):
            self.data.frame_pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        counter = 0
        x_batch, y_batch, firsts_batch = [], [], []
        for i in range(self.args.batch_size):
            # otherwise features like location invariant modifies self.data.all_data
            traj_data = np.copy(self.data.all_data[self.data.frame_pointer][:, 1:])
            temp_traj = np.zeros(traj_data.shape)
            n_batch = int(traj_data.shape[0] / (self.args.sequence_length + 1))

            slack = traj_data.shape[0] - self.args.sequence_length - 1 if self.args.mode == 'infer' else 0
            idx = 0 if slack < 1 else self.rnd.randint(0, slack)
            # idx = 0

            if self.args.data_type == 'location_invariant':
                firsts_batch.append(np.copy(traj_data[idx, :]))
                temp_traj[idx:, 0] = np.subtract(traj_data[idx:, 0], traj_data[idx, 0])
                temp_traj[idx:, 1] = np.subtract(traj_data[idx:, 1], traj_data[idx, 1])
                traj = temp_traj[idx:idx+self.args.sequence_length+1]
            elif self.args.data_type == 'previous_relative':
                traj = np.zeros((self.args.sequence_length+1, 2))
                # it should take (idx : idx + seq_len + 1) --> train = 8, target = 8
                # so if I want to make it relative to previous location
                # I should take the idx+1 .. to last one and subtract idx ... to last one - 1
                # sooo... idx+1 : idx+1 + seq_len + 1 and idx:idx + seq_len + 1
                # results is centered around zero
                trajectory_of_interest = traj_data[idx:idx+self.args.sequence_length+1]
                traj[1:] = trajectory_of_interest[1:] - trajectory_of_interest[:-1]
                trajectory_of_interest = np.around(trajectory_of_interest, decimals=4)
                firsts_batch.append(trajectory_of_interest)

            x_batch.append(np.copy(traj_data[idx:idx+self.args.sequence_length, :]))
            y_batch.append(np.copy(traj_data[1+idx:idx+self.args.sequence_length+1, :]))

            if self.rnd.rand() < (1.0/float(n_batch + 10e-6)) or self.args.mode == 'infer':
                # update dataset pointer
                self.frame_pointer += 1

        return {
            'inputs': x_batch,
            'targets': y_batch,
            'firsts': firsts_batch}

class ImageLoader(DataLoader):
    def __init__(self, data, rnd):
        super(ImageLoader, self).__init__(data, rnd)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        x_batch = np.zeros((self.args.batch_size, self.args.sequence_length, self.data.args.image_dim[0], self.data.args.image_dim[1], self.data.args.image_dim[2]))
        for b in range(self.args.batch_size):
            if len(self.data.all_data.keys()) > 0:
                for seq, frame in enumerate(self.data.frames_list[b]):
                    input_data = self.data.all_data[self.data.dataset_ids[b]][frame]
                    # actions = actions.reshape((actions.shape[0], actions.shape[1], actions.shape[2] * actions.shape[3]))
                    x_batch[b, seq] = input_data

        return {
            'images': np.array(x_batch).astype(np.float)/255.0
        }

class DynamicsLoader(DataLoader):
    def __init__(self, data, rnd):
        super(DynamicsLoader, self).__init__(data, rnd)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # Note actions are being loaded from a preprocessed dataset.
        # hence the change of basis options won't be applied here ..
        # but the ones taken out because of their small size will still be considered
        x_batch = np.zeros((self.args.batch_size, self.args.sequence_length, self.data.args.input_dim))
        y_batch = np.zeros((self.args.batch_size, self.args.sequence_length, self.data.args.output_dim))
        # a_batch = np.zeros((self.args.batch_size, self.args.sequence_length, 2 * self.data.args.max_num_agents))
        a_batch = np.zeros((self.args.batch_size, self.args.sequence_length, self.data.args.max_num_agents, 2))
        for b in range(self.args.batch_size):
            assert len(self.data.all_data.keys()) > 0
            for seq, frame in enumerate(self.data.frames_list[b]):
                if frame not in list(self.data.all_data[self.data.dataset_ids[b]].keys()):
                    print("frame: {0}, seq: {1} not in dataset id: {2}".format(frame, seq, self.data.dataset_ids[b]))
                    continue
                input_data = self.data.all_data[self.data.dataset_ids[b]][frame]
                z_input, z_target = self.getLatent(input_data)
                begin = 2 * self.data.args.input_dim
                end = 2 * (self.data.args.input_dim + self.data.args.max_num_agents)
                action_input = input_data[0][begin:end]
                action_input = action_input.reshape((1, 1, action_input.shape[0]))
                actions = np.concatenate(
                    (
                        action_input[:, :, :self.data.args.max_num_agents].reshape(
                            action_input.shape[0], action_input.shape[1], action_input.shape[2]//2, 1),
                        action_input[:, :, self.data.args.max_num_agents:].reshape(
                            action_input.shape[0], action_input.shape[1], action_input.shape[2]//2, 1)
                    ),
                    axis=3
                )
                # actions = actions.reshape((actions.shape[0], actions.shape[1], actions.shape[2] * actions.shape[3]))
                x_batch[b, seq] = z_input
                y_batch[b, seq] = z_target
                a_batch[b, seq] = actions

        return {
            'inputs': x_batch,
            'targets': y_batch,
            'actions': a_batch
        }

    def getLatent(self, data):
        data = data.reshape((1, data.shape[0], data.shape[1]))
        mu = data[:, :, :self.data.args.input_dim]
        logvar = data[:, :, self.data.args.input_dim:(2*self.data.args.input_dim)]

        s = logvar.shape
        z = mu + np.exp(logvar/2.0) * self.rnd.randn(*s)

        return z[:, :-1, :], z[:, 1:, :]
