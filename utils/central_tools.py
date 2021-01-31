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

import numpy as np
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
        self.model = None
        self.dynamics_model = None
        ### requirement for model based predictions
        if 'vc_world' in args.data_type:
            self.im_data = self.define_training_dirs(args.data_path, args.dataset_names, dynamics=True)

        if 'real_world' in args.data_type or '_r_' in args.save_name:
            self.az_data = self.define_training_dirs(args.data_path, args.dataset_names, dynamics=True)

        if 'real_world' in args.save_project:# and '_r_' not in args.save_name:
            self.dynamics_model = args.dynamics_model_type

        ### end of requirement
        self.data = self.define_training_dirs(args.data_path, args.dataset_names)
        self.model = args.model_type

        self.announce(args)

    @staticmethod
    def define_training_dirs(data_loc, data_names, file_name='pixel_pos.csv', dynamics=False):
        training_directories = []
        for dataset_name in data_names:
            folder = ''
            if dataset_name in ['univ', 'hotel']:
                folder = 'eth'
            elif dataset_name == 'ucy':
                folder = 'ucy'
                dataset_name = 'univ'
            elif dataset_name in ['zara01', 'zara02']:
                folder = 'ucy/zara'
            elif dataset_name in ['setup_one', 'setup_two', 'setup_three', 'setup_four', 'setup_five']:
                folder = 'robot/trajs'
            elif dataset_name in ['reduced_setup_one', 'reduced_setup_two', 'reduced_setup_three', 'reduced_setup_four', 'reduced_setup_five']:
                folder = 'robot/trajs'
            elif dataset_name in ['noskips_reduced_setup_one', 'noskips_reduced_setup_two', 'noskips_reduced_setup_three', 'noskips_reduced_setup_four', 'noskips_reduced_setup_five']:
                folder = 'robot/trajs'
            elif dataset_name in ['singlenorm_reduced_setup_one', 'singlenorm_reduced_setup_two', 'singlenorm_reduced_setup_three', 'singlenorm_reduced_setup_four', 'singlenorm_reduced_setup_five']:
                folder = 'robot/trajs'
            elif dataset_name in ['noclahe_singlenorm_reduced_setup_one', 'noclahe_singlenorm_reduced_setup_two', 'noclahe_singlenorm_reduced_setup_three', 'noclahe_singlenorm_reduced_setup_four', 'noclahe_singlenorm_reduced_setup_five']:
                folder = 'robot/trajs'
            elif dataset_name in ['complete_setup_one', 'complete_setup_two', 'complete_setup_three', 'complete_setup_four', 'complete_setup_five']:
                folder = 'robot/trajs'

            if dynamics:
                if folder is 'ucy':
                    file_name = folder
                else:
                    file_name = dataset_name

            full_dir_path = os.path.join(data_loc, os.path.join(folder, dataset_name))
            full_dir_path = os.path.join(full_dir_path, file_name)
            training_directories.append(full_dir_path)

        return training_directories

    @staticmethod
    def announce(args):
        print("########################################")
        print("Project: {0}; data type: {1}; model: {2}; results: {3}; error: {4}".format(
            args.save_project, args.data_type, args.model_type, args.results_name, args.error_type))
        print("########################################")

    def save(self, json_name):
        self.model.save_json(json_name)

    def load(self, json_name):
        self.model.load_json(json_name)

    def train(self):
        # TODO: add tqdm
        avg_time = 0
        avg_loss = 0
        avg_acc = 0
        avg_r = 0
        avg_mmd = 0
        writer_name = os.path.join(self.args.log_path, self.args.writer_name + DAY_TIME)
        self.writer = tf.summary.FileWriter(writer_name)
        self.writer.add_graph(self.model.sess.graph)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
        stop = False
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

                if self.args.data_type == 'vc_world':
                    # very ugly ... very ... updates az_data to have the pointers and frames of data
                    self.update(batch_contents['frames_ids'], batch_contents['dataset_names'], which_data='im')
                    im_contents = next(self.im_loader)
                    batch_contents["images"] = im_contents['images']

                if self.args.data_type == 'real_world' or '_r_' in self.args.save_name:
                    # very ugly ... very ... updates az_data to have the pointers and frames of data
                    self.update(batch_contents['frames_ids'], batch_contents['dataset_names'], which_data='az')
                    az_contents = next(self.az_loader)

                    if self.dynamics_model is not None:
                        batch_contents["latent_sequence"] = self.dynamics_model.step(az_contents, initial_states=dynamics_initial_states, which=self.args.which)
                    else:
                        batch_contents["latent_sequence"] = az_contents['inputs']

                train_loss, gradients, summary, velocity, train_acc = self.model.step(batch_contents)

                end = time.time() # Toc
                cur_time = end - start

                step = e * len(self.batch_loader) + b
                train_loss = train_loss / float(self.args.batch_size)
                if train_acc != None:
                    train_acc = train_acc / float(self.args.batch_size)

                # self.writer.add_summary(summary, step)

                if e == 2:
                    if self.args.prepare_network_tests:
                        self.prepare_network_tests(step, batch_contents, az_contents, gradients)

                    if self.args.test:
                        self.network_tests(step, batch_contents, gradients, az_contents)

                avg_time += cur_time
                avg_loss += train_loss
                if train_acc != None:
                    avg_acc += train_acc

                if self.args.data_type == 'vc_world':
                    avg_r += velocity
                    avg_mmd += train_acc

                if (step%99) == 0:
                    # velocity, train_acc == r_loss, mmd_loss in vclstm
                    if self.args.data_type == 'vc_world':
                        print("{}/{} (epoch {}), total_train_loss = {:.3f}, rec_loss = {:.3f}, kl_loss = {:.3f}, time/batch = {:.3f}".format(
                            step, self.args.num_epochs * len(self.batch_loader), e, avg_loss/99.0, avg_r/99.0, avg_mmd/99.0, avg_time/99.0))
                    else:
                        print("{}/{} (epoch {}), train_loss = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}".format(
                            step, self.args.num_epochs * len(self.batch_loader), e, avg_loss/99.0, avg_acc/99.0, avg_time/99.0))

                    avg_time = 0
                    avg_loss = 0
                    avg_acc = 0

                # if step == 18000:
                #     stop = True
                #     print("Early stopping ...")
                #     break

            if stop:
                print("...")
                break

        print("closing writer..")
        self.writer.close()

    def infer(self):
        blqk = []
        frame_no=[]
        self.batch_loader.reset()
        assert self.data.dataset_pointer == 0 and self.data.frame_pointer == 0
        assert self.batch_loader.dataset_pointer == 0 and self.batch_loader.frame_pointer == 0
        total_error = 0
        total_acc = 0
        missed_ones = 0
        initial_states = None
        if 'dilated_conv' not in self.args.save_name:
            initial_states = self.model.sess.run(self.model.LSTM_states)

        all_seq_states = []
        all_velocities = []
        all_labels = []
        if self.dynamics_model is not None:
            dynamics_initial_states = self.dynamics_model.sess.run(self.dynamics_model.initial_state)

        for b in range(len(self.batch_loader)):
            # print("********************** SAMPLING A NEW TRAJECTORY", b, "******************************")
            batch_contents = next(self.batch_loader)
            if self.args.data_type == 'real_world':
                    # very ugly ... very ... updates az_data to have the pointers and frames of data
                    self.update(batch_contents['frames_ids'], batch_contents['dataset_names'])
                    az_contents = next(self.az_loader)

                    ## NEW bit ###
                    batch_contents['az_contents'] = az_contents
                    if self.dynamics_model is not None:
                        # batch_contents["latent_sequence"] = self.dynamics_model.step(az_contents, initial_states=dynamics_initial_states, which=self.args.which)
                        batch_contents["dynamics_model"] = self.dynamics_model
                        batch_contents["dynamics_initial_states"] = dynamics_initial_states
                        batch_contents["which"] = self.args.which
                    else:
                        batch_contents["latent_sequence"] = az_contents['inputs']
                    ####

                    # if self.dynamics_model is not None:
                    #     batch_contents["latent_sequence"] = self.dynamics_model.step(az_contents, initial_states=dynamics_initial_states, which=self.args.which)
                    # else:
                    #     batch_contents["latent_sequence"] = az_contents['inputs']
            # print(batch_contents['frames_ids'][0])
            # print(batch_contents['frames_ids'][0][0])
            # if batch_contents['frames_ids'][0][0] > 371.:
            #     print("aha!")
# import glob
# im_list = sorted(glob.glob('../gaussian_observation_maps/data/robot/rss_new/setup_five/3/*.jpg'))
# im_list = sorted(np.genfromtxt(directory+'../gaussian_observation_maps/data/robot/rss_new/setup_five/reduced_im_list.txt', dtype='str').tolist())
# from scipy.misc import imresize as resize
# import matplotlib.pyplot as plt
# import cv2
# im = cv2.imread(im_list[4])
# frame = im[20:-70, 56:-70]
# frame = frame.astype(np.uint8)
# frame = resize(frame, (64, 64))
# frame = ((1.0 - frame) * 255).round().astype(np.uint8)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #### NEW modified #####
            error, state_h, trues_, velocities, accuracy = self.model.step(
                batch_contents,
                initial_states=initial_states,
                error_type=self.args.error_type)
            #####

            # all_seq_states.append(state_h)
            # all_velocities.append(velocities)
            # labels = []
            # for vel in velocities:
            #     if vel[0] > -0.009 and vel[0] < 0.009 and vel[1] > -0.009 and vel[1] < 0.009:
            #         labels.append('stay')
            #     else:
            #         if vel[0] + vel[1] <= 0:
            #             labels.append("->")
            #         else:
            #             labels.append("<-")

            #     # if (vel[0] > -0.009 and vel[0] < -0.005 or vel[0] < 0.009 and vel[0] > 0.005):
            #     #     if vel[1] > -0.009 and vel[1] < -0.005 or vel[1] < 0.009 and vel[1] > 0.005:
            #     #         print("opa, a toq stoi li... ")
            #     #         # this seems like the loitering stage..

            # frame_no.append(batch_contents['frames_ids'])
            # all_labels.append(labels)
            # for tr in trues_:
            #     blqk.append(tr)

            if accuracy != accuracy:
                raise Exception('accuracy should not be nan.')
            elif accuracy == -1:
                missed_ones += 1
                # print("a missed batch...none of the preheats continues at prediction time...")
            else:
                # if accuracy < 1.0:
                #     print("ibasi")
                if accuracy is not None:
                    total_acc += accuracy

            total_error += error

            if total_acc != total_acc:
                raise Exception('total_acc should not be nan.')

            # print("---> {0}/{1}".format(b, total_error))
            if (b+1) % 50 == 0:
                print("Processed trajectory number : ", b+1, "out of ", len(self.batch_loader), " trajectories")

        tot_mean_err = total_error/len(self.batch_loader)
        tot_mean_acc = total_acc/(len(self.batch_loader) - missed_ones)
        print("Total {0} error of the model is {1} and total accuracy is: {2}".format(self.args.error_type, tot_mean_err, tot_mean_acc))
        # np.savez_compressed("pred_rd_{0}_{1}.npz".format(self.args.dataset_names[0], self.args.predicted_length), h=np.array(all_seq_states), v=np.array(all_velocities), l=np.array(all_labels))
        # print("stuff saved on: pred_rd_{0}_{1}.npz".format(self.args.dataset_names[0], self.args.predicted_length))
        return tot_mean_err

    @staticmethod
    def prepare_network_tests(step, batch_contents, az_contents, gradients):
        np.save("tests/batch_contents_firsts_{0}.npy".format(step), batch_contents["firsts"])
        np.save("tests/batch_contents_frames_ids_{0}.npy".format(step), batch_contents["frames_ids"])
        np.save("tests/batch_contents_inputs_{0}.npy".format(step), batch_contents["inputs"])
        np.save("tests/batch_contents_targets_{0}.npy".format(step), batch_contents["targets"])
        np.save("tests/batch_contents_latent_sequence_{0}.npy".format(step), batch_contents["latent_sequence"])
        np.save("tests/az_contents_{0}.npy".format(step), az_contents)
        np.save("tests/gradients_{0}.npy".format(step), gradients)

    @staticmethod
    def network_tests(step, batch_contents, gradients, az_contents):
        # static module tests
        assert np.load("tests/batch_contents_firsts_{0}.npy".format(step)).all() == batch_contents["firsts"].all()
        assert np.load("tests/batch_contents_frames_ids_{0}.npy".format(step)).all() == batch_contents["frames_ids"].all()
        assert np.load("tests/batch_contents_inputs_{0}.npy".format(step)).all() == batch_contents["inputs"].all()
        assert np.load("tests/batch_contents_targets_{0}.npy".format(step)).all() == batch_contents["targets"].all()
        # gradients test
        test_gradients = np.load("tests/gradients_{0}.npy".format(step))
        for test_grad, grad in zip(test_gradients, gradients):
            try:
                assert test_grad.all() == grad.all()
            except:
                print("gradient shape is: ", grad.shape)

        # dynamic module tests
        loaded_az = np.load("tests/az_contents_{0}.npy".format(step)).item()
        assert loaded_az['inputs'].all() == az_contents['inputs'].all()
        assert loaded_az['targets'].all() == az_contents['targets'].all()
        assert loaded_az['actions'].all() == az_contents['actions'].all()
        assert np.load("tests/batch_contents_latent_sequence_{0}.npy".format(step)).all() == batch_contents["latent_sequence"].all()

    def unitTest(self):
        # Might end up moving this in a separate class
        print("Initiating Unit Testing..")
        import numpy as np

        temp_args = self.args

        # self.args = self.args._replace(set_dtype='per_agent')
        self.args = self.args._replace(observed_length=4)
        self.args = self.args._replace(predicted_length=4)
        self.args = self.args._replace(batch_size=10)
        # self.data = [
        #     'data/eth/hotel/pixel_pos.csv',
        #     'data/eth/univ/pixel_pos.csv',
        #     'data/ucy/zara/zara01/pixel_pos.csv',
        #     'data/ucy/zara/zara02/pixel_pos.csv']
        # assert len(self.data.frames_list) == 0
        # assert np.array(self.data.all_data).shape == (890, )
        # assert np.array(self.data.all_data[0]).shape == (14,3) # Do I need this indexing ..?
        # assert np.array(self.data.all_data[1]).shape == (26,3)
        # assert np.array(self.data.all_data[-1]).shape == (12,3)
        # assert np.sum([np.array(self.data.all_data[i]).shape for i in range(len(self.data.all_data))], axis=0).tolist() == [25249, 2670]
        # # [390, 750, 898, 1102] - sequences with all trajectories and not only those long enough
        # assert self.data.end_instance == [251, 543, 689, 890]
        # assert len(self.batch_loader) == 480 # num batches

        # self.args = self.args._replace(
        #     set_dtype='per_frame',
        #     sequence_length=8,
        #     latent_size=self.args.latent_size,
        #     batch_size=10)
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
        self.data = self.define_training_dirs(self.args.data_path, self.args.dataset_names)
