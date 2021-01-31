# -*- coding: utf-8 -*-
"""
Write CentralTool's functionality to enable a user to link
models with datasets and use this to train/infer.

author: Todor Davchev
date: 08.11.2018

"""
import os
import time
from time import gmtime, strftime

import tensorflow as tf

from utils.model_tools import ModelTool
from utils.tools import DataTool

DAY_TIME = strftime("%d_%b_%Y_%H_%M_%S", gmtime())
class CentralTool(DataTool, ModelTool):
    """Create a Neural central Unit."""

    def __init__(self, args, rnd):
        super(CentralTool, self).__init__(rnd)
        self.args = args
        self.hps = args
        self.rnd = rnd ### Need to take this away!!!
        self.data = self.define_training_dirs(args.data_path, args.dictionary)
        self.model = args.model_type
        self.dynamics_model = None
        if 'real_world' in args.save_project:
            self.az_data = self.define_training_dirs(args.data_path, args.dictionary, dynamics=True)
            self.dynamics_model = args.dynamics_model_type

    @staticmethod
    def define_training_dirs(data_loc, data_names, file_name='pixel_pos.csv', dynamics=False):
        training_directories = []
        for dataset_name in data_names:
            folder = ''
            if dataset_name in ['univ', 'hotel']:
                folder = 'eth'
            elif dataset_name== 'ucy':
                folder = 'ucy'
                dataset_name = 'univ'
            else:
                folder = 'ucy/zara'

            if dynamics:
                file_name = dataset_name

            full_dir_path = os.path.join(data_loc, os.path.join(folder, dataset_name))
            full_dir_path = os.path.join(full_dir_path, file_name)
            training_directories.append(full_dir_path)

        return training_directories

    def save(self, json_name):
        self.model.save_json(json_name)

    def load(self, json_name):
        self.model.load_json(json_name)

    def train(self):
        # TODO: add tqdm
        avg_time = 0
        avg_loss = 0
        writer_name = os.path.join(self.args.log_path, self.args.writer_name + DAY_TIME)
        self.writer = tf.summary.FileWriter(writer_name)
        self.writer.add_graph(self.model.sess.graph)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
        for e in range(self.args.num_epochs):
            self.batch_loader.reset()
            ### Think if I should do that for frames ...
            if self.args.set_dtype == 'per_agent':
                self.rnd.shuffle(self.data.all_data)

            assert self.data.dataset_pointer == 0 and self.data.frame_pointer == 0
            assert self.batch_loader.dataset_pointer == 0 and self.batch_loader.frame_pointer == 0
            # Assign the learning rate (decayed acc. to the epoch number)
            self.model.sess.run(tf.assign(self.model.lr, self.model.learning_rate * (self.hps.decay_rate ** e)))
            if self.dynamics_model is not None:
                dynamics_initial_states = self.dynamics_model.sess.run(self.dynamics_model.initial_state)
            for b in range(len(self.batch_loader)):
                start = time.time() # Tic

                batch_contents = next(self.batch_loader)
                if self.dynamics_model is not None:
                    # very ugly ... very ... updates az_data to have the pointers and frames of data
                    self.update(batch_contents['frames_ids'], batch_contents['dataset_names'])
                    az_contents = next(self.az_loader)
                    batch_contents["latent_sequence"] = self.dynamics_model.step(az_contents, initial_states=dynamics_initial_states)

                train_loss, summary = self.model.step(batch_contents)

                end = time.time() # Toc
                cur_time = end - start

                step = e * len(self.batch_loader) + b
                train_loss = train_loss / float(self.args.batch_size)
                self.writer.add_summary(summary, step)

                avg_time += cur_time
                avg_loss += train_loss
                if (step%99) == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                          step, self.args.num_epochs * len(self.batch_loader), e, avg_loss/99.0, avg_time/99.0))
                    avg_time = 0
                    avg_loss = 0

        print("closing writer..")
        self.writer.close()

    def infer(self):
        import numpy as np
        self.batch_loader.reset()
        assert self.data.dataset_pointer == 0 and self.data.frame_pointer == 0
        assert self.batch_loader.dataset_pointer == 0 and self.batch_loader.frame_pointer == 0
        total_error = 0
        initial_states = self.model.sess.run(self.model.LSTM_states)
        if self.dynamics_model is not None:
            dynamics_initial_states = self.dynamics_model.sess.run(self.dynamics_model.initial_state)
        for b in range(len(self.batch_loader)):
            # print("********************** SAMPLING A NEW TRAJECTORY", b, "******************************")
            batch_contents = next(self.batch_loader)
            if self.dynamics_model is not None:
                start_sub = time.time() # Tic
                # very ugly ... very ... updates az_data to have the pointers and frames of data
                self.update(batch_contents['frames_ids'], batch_contents['dataset_names'])
                az_contents = next(self.az_loader)
                batch_contents["latent_sequence"] = self.dynamics_model.step(az_contents, initial_states=dynamics_initial_states)
                end_sub = time.time() # Toc
                print("Sampling a latent sequence takes {0} ".format(end_sub-start_sub))

            error, summary = self.model.step(batch_contents, initial_states=initial_states)
            total_error += error

            if (b+1) % 50 == 0:
                print("Processed trajectory number : ", b+1, "out of ", len(self.batch_loader), " trajectories")

        tot_mean_err = total_error/len(self.batch_loader)
        print("Total mean error of the model is ", tot_mean_err)
        return tot_mean_err

    def unitTest(self):
        # Might end up moving this in a separate class
        print("Initiating Unit Testing..")
        import numpy as np

        temp_args = self.args

        self.args = self.args._replace(set_dtype='per_agent')
        self.args = self.args._replace(sequence_length=8)
        self.args = self.args._replace(batch_size=10)
        self.data = [
            'data/eth/hotel/pixel_pos.csv',
            'data/eth/univ/pixel_pos.csv',
            'data/ucy/zara/zara01/pixel_pos.csv',
            'data/ucy/zara/zara02/pixel_pos.csv']
        assert len(self.data.frames_list) == 0
        assert np.array(self.data.all_data).shape == (890, )
        assert np.array(self.data.all_data[0]).shape == (14,3) # Do I need this indexing ..?
        assert np.array(self.data.all_data[1]).shape == (26,3)
        assert np.array(self.data.all_data[-1]).shape == (12,3)
        assert np.sum([np.array(self.data.all_data[i]).shape for i in range(len(self.data.all_data))], axis=0).tolist() == [25249, 2670]
        # [390, 750, 898, 1102] - sequences with all trajectories and not only those long enough
        assert self.data.end_instance == [251, 543, 689, 890]
        assert len(self.batch_loader) == 480 # num batches

        self.args = self.args._replace(set_dtype='per_frame')
        self.data = ['data/eth/hotel/pixel_pos.csv']
        assert len(self.data.frames_list) == 1
        assert len(self.data.frames_list[0]) == 1168
        assert self.data.frames_list[0][0] == 1.0
        assert self.data.frames_list[0][-1] == 18061.0
        assert np.array(self.data.all_data).shape == (1, 1168, self.args.max_num_agents, 3)
        assert self.data.end_instance == [1168]
        assert len(self.batch_loader) == 24 # num batches

        self.data = [
            'data/eth/hotel/pixel_pos.csv',
            'data/eth/univ/pixel_pos.csv',
            'data/ucy/zara/zara01/pixel_pos.csv',
            'data/ucy/zara/zara02/pixel_pos.csv']
        assert len(self.data.frames_list) == 4
        assert len(self.data.frames_list[0]) == 1168
        assert len(self.data.frames_list[1]) == 876
        assert len(self.data.frames_list[2]) == 872
        assert len(self.data.frames_list[3]) == 1051
        assert self.data.frames_list[0][0] == 1.0
        assert self.data.frames_list[0][-1] == 18061.0
        assert self.data.frames_list[1][0] == 780.0
        assert self.data.frames_list[1][-1] == 12380.0
        assert self.data.frames_list[2][0] == 0.0
        assert self.data.frames_list[2][-1] == 9010.0
        assert self.data.frames_list[3][0] == 10.0
        assert self.data.frames_list[3][-1] == 10510.0
        assert np.array(self.data.all_data[0]).shape == (1168, self.args.max_num_agents, 3)
        assert np.array(self.data.all_data[1]).shape == (876, self.args.max_num_agents, 3)
        assert np.array(self.data.all_data[2]).shape == (872, self.args.max_num_agents, 3)
        assert np.array(self.data.all_data[3]).shape == (1051, self.args.max_num_agents, 3)
        assert self.data.end_instance == [1168, 876, 872, 1051]
        assert len(self.batch_loader) == 86 # num batches
        self.args = temp_args
