import os

import numpy as np

import utils.datastructures as datastructures
import utils.loader as loader
from utils.params import ArgParams


class DataTool(object):
    """ Create a new data tool."""

    def __init__(self, rnd):
        super(DataTool,self).__init__(rnd)
        self.rnd = rnd

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @property
    def args(self):
        """The data dictionary we will use."""
        return self.__args

    @args.setter
    def args(self, args):
        self.__args = ArgParams(
            error_type=args.error_type,
            dataset_names=args.dataset_names,
            log_path=args.log_path,
            which=args.which,
            data_path=args.data_path,
            save_name=args.save_name,
            writer_name=args.writer_name,
            mode=args.mode,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            observed_length=args.observed_length,
            predicted_length=args.predicted_length,
            sequence_length=args.observed_length + args.predicted_length, # ...
            set_dtype=args.set_dtype,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            l_size=args.l_size,
            dynamics_seq_len=args.dynamics_seq_len,
            dynamics_output_dim=args.l_size,
            max_num_agents=args.max_num_agents,
            data_type=args.data_type,
            prepare_network_tests=args.prepare_network_tests,
            test=args.test,
            preload=args.preload,
            rotate=args.rotate,
            degrees=args.degrees,
            series_path=args.series_path,
            test_size=args.test_size,
            group_by=args.group_by,
            save_path=args.save_path,
            dynamics_num_units=args.dynamics_num_units,
            latent_size=args.dynamics_num_units + args.l_size,
            image_dim=args.image_dim)

    @property
    def data(self):
        """The data dictionary we will use."""
        return self.__data

    @data.setter
    def data(self, dictionary):
        # updating pointers calls this setter too
        # perhaps I ned a different approach ..
        if dictionary is not None:
            self.__data = self.reset()
            self.__data.args = self.args
            self.buildData(dictionary)
            if self.args.rotate:
                self.args = self.args._replace(
                    dataset_names=self.update_names(self.args.dataset_names, self.args.degrees))

            self.__batch_loader = self.loader()
            self.__batch_loader.args = self.args
            # I need to come up with a better location for this fella
            if self.args.test and self.args.set_dtype == 'per_frame':
                self.batch_loader.unitTest()

    @property
    def batch_loader(self):
        return self.__batch_loader

    def buildData(self, dictionary):
        """Builds the Data."""
        _index = 0
        for idx, file_path in enumerate(dictionary):
            rotations = self.args.degrees if self.args.rotate else [0]
            for aidx, degrees in enumerate(rotations):
                data = np.genfromtxt(file_path, delimiter=',')
                if 'zara02' in file_path:
                    # comes from 10520 frame cannot be processed, leads to 10510 fails
                    data = data[:, :-3] # tackles a weird bug..

                if self.args.rotate:
                    print("Rotating trajectory data {0} by {1} degrees...".format(file_path, degrees))
                    data = self.rotate(data, degrees)

                self.data.add((_index, data))
                _index += 1

    def update(self, frames_list, dataset_ids, which_data='az'):
        def pick(which_data):
            if which_data == 'az':
                return self.az_data
            elif which_data == 'im':
                return self.im_data

        _data = pick(which_data)
        _data.frame_pointer = self.data.frame_pointer
        _data.dataset_pointer = self.data.dataset_pointer
        _data.dataset_ids = dataset_ids
        _data.frames_list = frames_list

    def reset(self):
        """Resets the datastructure."""
        if 'per_agent' in self.args.set_dtype:
            return datastructures.DataPerAgent()
        elif 'per_frame' in self.args.set_dtype:
            return datastructures.DataPerFrame()

    def loader(self, dynamics=False, images=False):
        if not dynamics and 'per_agent' in self.args.set_dtype and not images:
            return loader.PerAgentLoader(self.data, self.rnd)
        elif not dynamics and 'per_frame' in self.args.set_dtype and not images:
            return loader.PerFrameLoader(self.data, self.rnd)
        elif not dynamics and images:
            return loader.ImageLoader(self.im_data, self.rnd)
        elif dynamics:
            return loader.DynamicsLoader(self.az_data, self.rnd)

    @property
    def im_data(self):
        return self.__im_data

    @property
    def im_loader(self):
        return self.__im_loader

    @im_data.setter
    def im_data(self, dictionary):
        '''
        Either loads already saved data or preloads a fresh set of datasets sequences.
        [batch_size x sequence_length x MU+LOGVAR+2*ACTS], 2* since its x and y
        args:
            data_names: the names of the used data sets;
            seq_len   : the length of the sequences considered for this training;
            preload   : preload from scratch or use an already saved data.
        returns:
            train_data: combined training series from all datasets;
            test_data : combined test series from all datasets;
        '''
        self.__im_data = datastructures.ImageData()
        self.__im_data.args = self.args
        self.getImData(dictionary)
        self.__im_loader = self.loader(images=True)
        self.__im_loader.args = self.args
        if self.args.rotate:
            self.__im_loader.args= self.args._replace(
                dataset_names=self.update_names(self.args.dataset_names, self.args.degrees))

    @property
    def az_data(self):
        return self.__az_data

    @property
    def az_loader(self):
        return self.__az_loader

    @az_data.setter
    def az_data(self, dictionary):
        '''
        Either loads already saved data or preloads a fresh set of datasets sequences.
        [batch_size x sequence_length x MU+LOGVAR+2*ACTS], 2* since its x and y
        args:
            data_names: the names of the used data sets;
            seq_len   : the length of the sequences considered for this training;
            preload   : preload from scratch or use an already saved data.
        returns:
            train_data: combined training series from all datasets;
            test_data : combined test series from all datasets;
        '''
        self.__az_data = datastructures.ActionsAndLatentsData()
        self.__az_data.args = self.args._replace(
            input_dim=self.args.l_size,
            output_dim=self.args.dynamics_output_dim)
        self.getData(dictionary)
        self.__az_loader = self.loader(dynamics=True)
        self.__az_loader.args = self.args
        if self.args.rotate:
            self.__az_loader.args= self.args._replace(
                dataset_names=self.update_names(self.args.dataset_names, self.args.degrees))

    def rotate(self, dictionary, degrees):
        if degrees == 0:
            pass
        elif degrees == -90:
            # swap axes
            dictionary = dictionary[[0, 1, 3, 2], :]
            dictionary[3, :] = 1.0 - dictionary[3, :]
        elif degrees == -180:
            dictionary[2, :] = 1.0 - dictionary[2, :]
            dictionary[3, :] = 1.0 - dictionary[3, :]
        elif degrees == -270:
            # swap axes
            dictionary = dictionary[[0, 1, 3, 2], :]
            dictionary[2, :] = 1.0 - dictionary[2, :]
        else:
            raise NotImplementedError

        return dictionary

    def getImData(self, dictionary):
        # this should be redundant
        if self.args.rotate:
            print("*** Using rotated data ***")
            dictionary = self.update_names(dictionary, self.args.degrees)

        if self.args.preload:
            # dictionary is already defined in args ...
            # this relies upon having used time_series.py
            # to generate file_name.npz which can then be preloaded.
            # TODO: remove this local referencing ..
            print("Preloading data...")

            series = {}
            for name in dictionary:
                name = name.split("/")
                series[name[-1]] = np.load("{0}.npz".format('/'.join(name)))

            self.im_data.series = series

            self.saveImData(self.im_data.series, np.copy(dictionary))

        self.im_data.add(np.copy(dictionary))

    def getData(self, dictionary):
        # this should be redundant
        if self.args.rotate:
            print("*** Using rotated data ***")
            dictionary = self.update_names(dictionary, self.args.degrees)

        if self.args.preload:
            # dictionary is already defined in args ...
            # this relies upon having used time_series.py
            # to generate file_name.npz which can then be preloaded.
            # TODO: remove this local referencing ..
            print("Preloading data...")

            train_data, test_data = self.split(np.copy(dictionary))
            if self.args.test_size < 0.9:
                self.saveData(train_data, np.copy(dictionary))
            if self.args.test_size > 0.05:
                self.saveData(test_data, np.copy(dictionary))

        self.az_data.add(np.copy(dictionary))

    def saveImData(self, data, data_path):
        for idx, key in enumerate(data.keys()):
            split_path = data_path[idx].split("/")
            # this if might be not necessary over time but still..
            if type(data[key]) == dict:
                np.save(os.path.join('/'.join(split_path[:-1]), "{0}_loaded".format(key)), data[key])
            else:
                raise NotImplementedError

            print("{2} data saved to {0}/{1}_loaded.npy".format('/'.join(split_path[:-1]), key, self.args.mode, self.args.dynamics_seq_len))

    def saveData(self, data, data_path):
        for idx, key in enumerate(data.keys()):
            split_path = data_path[idx].split("/")
            # this if might be not necessary over time but still..
            if type(data[key]) == dict:
                np.save(os.path.join('/'.join(split_path[:-1]), "{0}_{1}_{2}".format(key, self.args.mode, self.args.dynamics_seq_len)), data[key])
            else:
                np.savez_compressed(os.path.join('/'.join(split_path[:-1]), "{0}_{1}_{2}".format(key, self.args.mode, self.args.dynamics_seq_len)), data=data[key][0], indices=data[key][1])

            print("{2} data saved to {0}/{1}_{2}_{3}.npz".format('/'.join(split_path[:-1]), key, self.args.mode, self.args.dynamics_seq_len))

    @staticmethod
    def update_names(data_names, deg):
        updated_names = []
        for idx, name in enumerate(data_names):
            for aidx, degrees in enumerate(deg):
                new_name = name.split("/")
                new_name[-1] = "".join([new_name[-1], "_rotated_{0}".format(degrees)])
                updated_names.append(os.path.join('/'.join(new_name)))

        return updated_names

    def split(self, data_names):
        '''
        Split data to train and test.
        returns:
            train_data: a dictionary with all train inputs and targets data
            test_data : a dictionary with all train/target per dataset.
        '''
        from sklearn.model_selection import train_test_split

        self.az_data.series = self.preload(data_names=data_names)
        if self.args.group_by == 'frame_id':
            return (self.az_data.series, None) if self.args.test_size < 0.05 else (None, self.az_data.series)

        # targets will be determined in random batch sampling
        # as it makes more sense to process all data together and randomly select one of the 
        # entire batch.
        test_data = {}
        train_data = []
        dataset_indices = []
        for name in data_names:
            current_data, test_data[name], _, _ = \
                train_test_split(data[name], data[name], test_size=self.args.test_size, random_state=self.rnd)
            if len(train_data) == 0:
                train_data = current_data
            else:
                train_data = np.concatenate((train_data, current_data), axis=0)

            dataset_indices.append(len(train_data))

        # throughout time I know where the indices are.
        train_data_dict = {"all"    : [train_data, dataset_indices]}

        test_data_dict  = {}
        for name in data_names:
            test_data_dict[name] = [test_data[name], dataset_indices]

        return train_data_dict, test_data_dict

    def preload(self, data_names):
        '''
        Used during LSTM training, refers to the sequences of frames recorded from the datasets.
        args:
            data_names: the names of the used data sets.
        returns:
            combined series from all datasets. agents[frame_id, height, width]
        '''
        series = {}
        for name in data_names:
            name = name.split("/")
            if 'ucy' in name[-1]:
                dict_name = name[1]
            else:
                dict_name = name[-1]

            series[dict_name] = np.load("{0}/{1}_series_{2}.npz".format(self.args.series_path, self.args.dynamics_seq_len, dict_name))

        return series
