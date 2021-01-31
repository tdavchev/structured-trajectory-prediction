import os

import numpy as np

from utils.params import TrajArgParams


class Data(object):

    # in case we scaled to a large scale experimentation reduces memory usage..
    __slots__ = ['frames_list', 'end_instance', 'all_data', 'dataset_pointer', 'frame_pointer']

    def __init__(self):
        # These should be in getters and setters, used by subclass
        # I might end up having to change here but not there
        # stop abusing the init Todor...
        self.frames_list = [] # I think that's redundant too
        self.end_instance = [] # I don't know if I use this...revisit
        self.all_data = []
        self.dataset_pointer = 0
        self.frame_pointer = 0

    @property
    def args(self):
        return self.__args

    @args.setter
    def args(self, args):
        self.__args = TrajArgParams(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            dynamics_seq_len=args.dynamics_seq_len,
            input_dim=args.input_dim,
            image_dim = args.image_dim,
            output_dim=args.output_dim,
            max_num_agents=args.max_num_agents,
            group_by=args.group_by,
            test_size=args.test_size,
            save_path=args.save_path,
            latent_size=args.latent_size,
            mode=args.mode
            )

class DataPerFrame(Data):
    """Consider agents per frame."""

    def __init__(self):
        super(DataPerFrame, self).__init__()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # TODO: Move this to the DataLoader
        # All the data in this sequence
        frame_data = self.all_data[self.dataset_pointer]
        frame_ids = self.frames_list[self.dataset_pointer]
        seq_frame_data = \
            frame_data[self.frame_pointer:self.frame_pointer+self.args.sequence_length+1, :]
        seq_source_frame_data = \
            frame_data[self.frame_pointer:self.frame_pointer+self.args.sequence_length, :]
        seq_target_frame_data = \
            frame_data[self.frame_pointer+1:self.frame_pointer+self.args.sequence_length+1, :]
        # Number of unique agents in this sequence of frames
        agents_id_list = np.unique(seq_frame_data[:, :, 0])
        num_unique_agents = agents_id_list.shape[0]

        input_data = np.zeros((self.args.sequence_length, self.args.max_num_agents, self.args.input_dim))
        target_data = np.zeros((self.args.sequence_length, self.args.max_num_agents, self.args.output_dim))
        velocity_data = np.zeros((self.args.sequence_length, self.args.max_num_agents, self.args.input_dim))
        yaw_data = np.zeros((self.args.sequence_length, self.args.max_num_agents, self.args.input_dim))
        fid_data = frame_ids[self.frame_pointer:self.frame_pointer+self.args.sequence_length] # I only need the input frames!

        visits = np.zeros(len(agents_id_list))
        for seq in range(self.args.sequence_length):
            iseq_frame_data = seq_source_frame_data[seq, :]
            tseq_frame_data = seq_target_frame_data[seq, :]
            for age in range(num_unique_agents):
                agent_id = agents_id_list[age]
                if agent_id == 0:
                    continue
                else:
                    iage = iseq_frame_data[iseq_frame_data[:, 0] == agent_id, :]
                    vage = iseq_frame_data[iseq_frame_data[:, 0] == agent_id, :].copy()
                    yage = iseq_frame_data[iseq_frame_data[:, 0] == agent_id, :].copy()
                    tage = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == agent_id, :])
                    if iage.size != 0 and age < self.args.max_num_agents:
                        visits[age] += 1
                        input_data[seq, age, :] = iage
                    if tage.size != 0 and age < self.args.max_num_agents:
                        target_data[seq, age, :] = tage
                    if vage.size != 0 and age < self.args.max_num_agents:
                        if iage.size != 0 and tage.size != 0:
                            vage[0, 1] = input_data[seq, age, 1] - target_data[seq, age, 1]
                            vage[0, 2] = input_data[seq, age, 2] - target_data[seq, age, 2]
                        # if visits[age] == 1:
                        #     vage[0, 1] = 0.0
                        #     vage[0, 2] = 0.0
                        # else:
                        #     vage[0, 1] = input_data[seq-1, age, 1] - vage[0, 1]
                        #     vage[0, 2] = input_data[seq-1, age, 2] - vage[0, 2]

                        velocity_data[seq, age, :] = vage
                    if yage.size != 0 and age < self.args.max_num_agents:
                        if iage.size != 0 and tage.size != 0:
                            ## yaw
                            _x = iage[0, 1:]
                            _y = tage[1:]
                            dxdy = np.subtract(_x, _y)
                            yaw = np.arctan2(dxdy[0], dxdy[1])
                            ###
                            quats = self.toQuaterion(yaw, 0, 0)
                            yage[0, 1] = quats[0] # q_w
                            yage[0, 2] = quats[-1] # q_z

                            yaw_data[seq, age, :] = yage


        return np.copy(input_data), np.copy(target_data), np.copy(velocity_data), np.copy(yaw_data), fid_data

    def toQuaterion(self, yaw, pitch, roll):
        def normalise(array):
            """ 
            Normalize a 4 element array/list/numpy.array for use as a quaternion
            
            :param quat_array: 4 element list/array
            :returns: normalized array
            :rtype: numpy array
            """
            quat = np.array(array)
            return quat / np.sqrt(np.dot(quat, quat))

        cy = np.cos(yaw * 0.5)
        sy = np.cos(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        q_w = cy * cp * cr + sy * sp * sr
        q_x = cy * cp * sr - sy * sp * cr
        q_y = sy * cp * sr + cy * sp * cr
        q_z = sy * cp * cr - cy * sp * sr

        return normalise([q_w, q_x, q_y, q_z])

    def add(self, content):
        '''
        Collect all data from the already preprocessed csv files.
        These files are used for training and to speed up the process
        of training as opposed to loading and preprocessing each and every
        one individually.
        The spit out format is frame oriented where each frame has a list of
        agents and their locations.
        NOTE: modified from https://github.com/vvanirudh/social-lstm-tf
        args:
            content         : (idx, the directories names used)
        loads:
            all_data        : an np.array() of size [num_dirs x num_unique_frames x max_num_agents x 3 (aid, height, width)]
            frames_list     : an array with the actual frames ordered accordingly
            end_instance    : the indices of the last frame per dataset
        '''
        idx, data_set = content[0], content[1]
        unique_frames = np.unique(data_set[0, :])

        self.frames_list.append(unique_frames)
        self.end_instance.append(len(unique_frames))

        self.all_data.append((np.zeros((len(unique_frames), self.args.max_num_agents, self.args.input_dim))))
        for frame_id, frame in enumerate(unique_frames):
            agents_in_frame = data_set[:, data_set[0, :] == frame].T
            self.all_data[idx][frame_id, 0:len(agents_in_frame), :] = agents_in_frame[:, [1,2,3]]

class DataPerAgent(Data):
    """Consider agents per frame."""

    def __init__(self):
        super(DataPerAgent, self).__init__()

    def add(self, content):
        # I need to improve this..do I need a node?!
        idx, data_set = content[0], content[1]
        counter = 0
        num_agents = np.size(np.unique(data_set[1,:]))
        for agent in range(1, num_agents+1):
            traj = data_set[:, data_set[1, :] == agent]
            traj = traj[[0, 2, 3], :].T

            if traj.shape[0] > 150:
                print("wtf..?!")

            if traj.shape[0] > (self.args.sequence_length + 2):
                self.all_data.append(traj)
                counter += 1

        if idx == 0:
            self.end_instance.append(counter)
        else:
            self.end_instance.append(self.end_instance[-1] + counter)

class ImageData(Data):

    def __init__(self):
        super(ImageData, self).__init__()

    def add(self, data_names):
        '''
        Load already saved sequences from the datasets.
        args:
            data_names: the names of the datasets
            seq_len   : the sequence length.
        '''
        im_data = {}
        for name in data_names:
            name = name.split("/")
            data_path = '/'.join(name[:-1])

            im_data[name[-1]] = np.load(os.path.join(data_path, "{0}_loaded.npy".format(name[-1]))).item()

        #TODO: implement the prepare data method...

        self.all_data = im_data

    @property
    def series(self):
        return self.__series

    @series.setter
    def series(self, series):
        if self.args.group_by == 'frame_id':
            self.__series = self.combineFrameImSeries(series)
        else:
            raise NotImplementedError

    @staticmethod
    def combineFrameImSeries(series):
        '''
        TODO: when loading agents it mixes the (x,y) positioning
        We need the mean, log variance and the 
        sequence of (x,y) positions of agents within a single frame.
        This data is used as input for the dynamics model. We need to address them by frame id.
        args:
            seq_len   : the length of the sequences considered for this training.
        returns:
            a dictionary with all datasets' mu, logvar, (x,y) positions of agents.
        '''
        # TODO: improve load time.. 
        combined_data = {}
        for key in series.keys():
            print("processing.. ", key)
            combined_data[key] = {}
            for idx in range(series[key]["dataset"].shape[0]):
                # this will save both input and target
                # series[key]["frame_id"].shape: (875, 2, 1), input and target frames
                frame_id = series[key]["framelist"][idx]
                combined_data[key][frame_id] = series[key]["dataset"][idx]

        print("Returning combined data: ")
        return combined_data

class ActionsAndLatentsData(Data):
    """Consider agents per frame."""

    def __init__(self):
        super(ActionsAndLatentsData, self).__init__()

    def add(self, data_names):
        '''
        Load already saved sequences from the datasets.
        args:
            data_names: the names of the datasets
            seq_len   : the sequence length.
        '''
        train_data = {}
        test_data = {}
        data_path = ""
        if self.args.test_size < 0.9:
            if self.args.group_by == 'name' or self.args.group_by == 'crops':
                train_data = np.load(os.path.join(self.args.save_path, "all_train_{0}.npz".format(self.args.dynamics_seq_len))) # where is that coming from ..
            else:
                for name in data_names:
                    name = name.split("/")
                    data_path = '/'.join(name[:-1])
                    if 'ucy' in name[-1]:
                        dict_name = 'ucy'
                    else:
                        dict_name = name[-1]

                    train_data[dict_name] = np.load(os.path.join(data_path, "{0}_train_{1}.npy".format(name[-1], self.args.dynamics_seq_len)), allow_pickle=True).item()

        # print(data_names)
        # if len(train_data.keys()) > 0 and self.any(data_names, 'hotel'):
        #     assert np.loadtxt('tests/hotel_test_train_data.txt').all() == train_data['hotel'][1.0].all()
        # if len(train_data.keys()) > 0 and self.any(data_names, 'zara01'):
        #     assert np.loadtxt('tests/zara01_test_train_data.txt').all() == train_data['zara01'][0.0].all()
        # if len(train_data.keys()) > 0 and self.any(data_names, 'zara02'):
        #     assert np.loadtxt('tests/zara02_test_train_data.txt').all() == train_data['zara02'][10.0].all()

        if self.args.test_size > 0.05:
            for name in data_names:
                data_path = ''
                name = name.split("/")
                for folder in name[:-1]:
                        data_path = os.path.join(data_path, folder)

                if self.args.group_by == 'name':
                    test_data[name[-1]] = np.load(os.path.join(data_path, "{0}_infer_{1}.npz".format(name[-1], self.args.dynamics_seq_len)))
                else:
                    test_data[name[-1]] = np.load(os.path.join(data_path, "{0}_infer_{1}.npy".format(name[-1], self.args.dynamics_seq_len)), allow_pickle=True).item()

        self.all_data = train_data if self.args.mode == 'train' else test_data

    @staticmethod
    def any(arr, word):
        for item in arr:
            if word in item:
                return True

        return False

    @property
    def dataset_ids(self):
        return self.__dataset_ids

    @dataset_ids.setter
    def dataset_ids(self, contents):
        self.__dataset_ids = contents

    @property
    def series(self):
        return self.__series

    @series.setter
    def series(self, series):
        if self.args.group_by == 'name':
            self.__series = self.combineSeries(series)
        elif self.args.group_by == 'frame_id':
            self.__series = self.combineFrameSeries(series)
        elif self.args.group_by == 'crops':
            self.__series = self.combineCroppedSeries(series)

    @staticmethod
    def combineFrameSeries(series):
        '''
        TODO: when loading agents it mixes the (x,y) positioning
        We need the mean, log variance and the 
        sequence of (x,y) positions of agents within a single frame.
        This data is used as input for the dynamics model. We need to address them by frame id.
        args:
            seq_len   : the length of the sequences considered for this training.
        returns:
            a dictionary with all datasets' mu, logvar, (x,y) positions of agents.
        '''
        # TODO: improve load time.. 
        combined_data = {}
        for key in series.keys():
            combined_data[key] = {}
            for idx in range(series[key]["mu"].shape[0]):
                # this will save both input and target
                # series[key]["frame_id"].shape: (875, 2, 1), input and target frames
                frame_id = series[key]["frame_id"][idx][0]
                combined_data[key][frame_id] = np.concatenate(
                    (
                        series[key]["mu"][idx],
                        series[key]["logvar"][idx],
                        series[key]["act_x"][idx],
                        series[key]["act_y"][idx],
                        series[key]["frame_id"][idx].reshape(
                            (series[key]["frame_id"][idx].shape[0],1))
                    ), axis=1)

        return combined_data

    @staticmethod
    def combineCroppedSeries(series):
        '''
        We need the mean, log variance and the 
        sequence of (x,y) positionsof agents within a single frame.
        This data is used as input for the dynamics model.
        args:
            seq_len   : the length of the sequences considered for this training.
        returns:
            a dictionary with all datasets' mu, logvar, (x,y) positions of agents.
        '''
        combined_data = {}
        for key in series.keys():
            for dset in range(4):
                combined_data[key] = np.concatenate(
                    (
                        series[key]["mu"][dset],
                        series[key]["logvar"][dset],
                        series[key]["acts_x"][dset],
                        series[key]["acts_y"][dset],
                        series[key]["agent_id"][dset]
                        # series[key]["obs"][dset]
                    ), axis=2)

        return combined_data

    @staticmethod
    def combineSeries(series):
        '''
        We need the mean, log variance and the 
        sequence of (x,y) positionsof agents within a single frame.
        This data is used as input for the dynamics model.
        args:
            seq_len   : the length of the sequences considered for this training.
        returns:
            a dictionary with all datasets' mu, logvar, (x,y) positions of agents.
        '''
        combined_data = {}
        for key in series.keys():
            combined_data[key] = np.concatenate(
                (
                    series[key]["mu"],
                    series[key]["logvar"],
                    series[key]["act_x"],
                    series[key]["act_y"],
                    series[key]["frame_id"].reshape(
                        (series[key]["frame_id"].shape[0], series[key]["frame_id"].shape[1], 1))
                ), axis=2)

        return combined_data
