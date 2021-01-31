#!/usr/bin/env python
'''
Command line tool based on Python 3.
Functionality:
(1) pass a path to your data files
(2) pass the names of the different datasets

author: Todor Davchev
date: 08.11.2018
'''
import argparse
import json
import os

from numpy.random import RandomState
from utils import (DEFAULT_SEED, DEFAULT_SEED1, DEFAULT_SEED2, DEFAULT_SEED3,
                   DEFAULT_SEED4, DEFAULT_SEED5, DEFAULT_SEED6, DEFAULT_SEED7,
                   DEFAULT_SEED8, DEFAULT_SEED9)
from utils.central_tools import CentralTool

SEEDS = [DEFAULT_SEED, DEFAULT_SEED1, DEFAULT_SEED2, DEFAULT_SEED3, DEFAULT_SEED4, DEFAULT_SEED5, DEFAULT_SEED6, DEFAULT_SEED7, DEFAULT_SEED8, DEFAULT_SEED9]

# def main(train_dset, seed, name, obs, pred, deg, dim, preload, temperature, error_type, results_name):
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_no', type=int, default=0, #seed,
                        help='To unit test or not to unit test.')
    parser.add_argument('--prepare_network_tests', type=bool, default=False,
                        help='Prepare_network_tests.')
    parser.add_argument('--test', type=bool, default=False,
                        help='To unit test or not to unit test.')
    parser.add_argument('--data_path', default="data",
                        help='Where is your data at.')
    parser.add_argument('--save_path', default="save",
                        help='Where are your saves at.')
    parser.add_argument('--log_path', default="logdir",
                        help='Where are your saves at.')
    #### Modify to test inference/ training processes #####
    parser.add_argument('--learning_rate', type=float, default=5e-4,#5e-4,#0.005,
                        help='The learning rate used.')
    parser.add_argument('--num_epochs', type=int, default=100, # 100 #seems enough for frame
                        help='The number of epochs.')
    ################
    parser.add_argument('--dataset_names', nargs='*',
                        # default=["complete_setup_one", "complete_setup_two", "complete_setup_three", "complete_setup_four"],
                        default=["univ", "ucy", "zara01", "zara02"],
                        # default=train_dset,
                        help='the dictionary passed containing all dataset names')
    parser.add_argument('--mode', default="train",
                        help='The mode ran train/infer.')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='The sequence length. This controls the models sequence, data seq is sum of pred+obs')
    parser.add_argument('--observed_length', type=int, default=4,
                        help='The sequence length.')
    parser.add_argument('--predicted_length', type=int, default=4,
                        help='The sequence length.')
    parser.add_argument('--batch_size', type=int, default=10, # 10 # enough for frame
                        help='The batch size.')
    parser.add_argument('--test_size', type=float, default=0.0,
                        help='Where are your saves at.')
    ################
    # parser.add_argument('--dataset_names', nargs='*',
    #                      default=["hotel"], # ['complete_setup_five'], #[name],
    #                      help='noclahe_singlenorm_reduced_setup_five the dictionary passed containing all dataset names')
    # parser.add_argument('--mode', default="infer",
    #                      help='The mode ran train/infer.')
    # parser.add_argument('--sequence_length', type=int, default=1,
    #                      help='The sequence length.')
    # parser.add_argument('--observed_length', type=int, default=4, #obs,
    #                      help='The sequence length.')
    # parser.add_argument('--predicted_length', type=int, default=8, #pred,
    #                      help='The sequence length.')
    # parser.add_argument('--batch_size', type=int, default=1,
    #                      help='The batch size.')
    # parser.add_argument('--test_size', type=float, default=1.0,#0.99,
    #                      help='Where are your saves at.')
    ################################ End ################################
    # Modify to switch between per frame/ per agent / per sth else in future
    parser.add_argument('--which', default="this",
                        help='Where are your saves at. [this or next]')
    parser.add_argument('--save_project', default="real_world",
                        help='Where are your saves at. [\
                            basic_lstm,\
                            just_lstm,\
                            real_world,\
                            social_lstm]')
    parser.add_argument('--data_type', default="real_world",
                        help='[\
                            pixel_normalised,\
                            real_world]')  # use pixel_normalised for social lstm too
    parser.add_argument('--results_name', default="results_rd_hotel",
                        help='Name model is saved under e.g.: [\
                            results_just_hotel,\
                            results_rd_hotel, \
                            results_social_hotel]')
    parser.add_argument('--save_name', default='map_cond_per_frame_hotel',
                        help='Name model is saved under e.g.: [\
                            per_agent_lstm_hotel,\
                            social_per_frame_hotel].')  # this one trains with all rotations 0 to -270
    parser.add_argument('--writer_name', type=str, default="per_frame_hotel",
                        help='How to save logs e.g.: [\
                        per_frame_just_hotel,\
                        per_frame_social_hotel]')
    parser.add_argument('--set_dtype', default="per_frame",
                        help='How to load the data: [\
                        per_agent,\
                        per_frame]')
    parser.add_argument('--model_type', default="map_cond_per_frame_LSTM",
                        help='LSTM type: [\
                            per_agent_LSTM, \
                            per_frame_LSTM]')
    parser.add_argument('--max_num_agents', type=int, default=50,
                        help='The maximum number of agents. 50 when using all dsets + ucy, 40 otherwise, 1 when robot')
    parser.add_argument('--input_dim', type=int, default=3,
                        help='The input dimensionality.')
    parser.add_argument('--output_dim', type=int, default=3,
                        help='The output dimensionality.')
    ################################ End ################################
    # Social LSTM parameters
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid.')
    parser.add_argument('--neighbourhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid.')
    parser.add_argument('--social_grid_include', type=bool, default=False,
    # please double check the positioning!
                        help='True/False used for weight init')
    # parser.add_argument('--dimensions', type=int, default=dim,
    parser.add_argument('--dimensions', nargs='*', default=[576, 720],
    # please double check the positioning!
                        help='Used at inference time for grid selection:[\
                        eth/univ = [480, 640], \
                        others = [576, 720]]') # x (height), y (width) positions
    # The Dynamic LSTM parameters
    parser.add_argument('--series_path', default="series",
                        help='Where are your saves at.')
    parser.add_argument('--group_by', default='frame_id',
                        help='[name, frame_id, crops]')
    parser.add_argument('--dynamics_seq_len', type=int, default=2,
                        help='The dynamics data load sequence length.')
    parser.add_argument('--num_mixtures', type=int, default=5,
                        help='The sequence length.')
    #                     help='Where are your saves at.')
    parser.add_argument('--dynamics_model_type', default="mdn_rnn",
                        help='LSTM type: [mdn_rnn]')
    parser.add_argument('--dynamics_save_name',
                        default="MDNRNN_RNN_SIZE_768_8_num_mix_5_FLAT_acts_True_3_by_1_hotel.json",
                        help='LSTM type: [\
                            MDNRNN_RNN_SIZE_768_8_num_mix_5_FLAT_acts_True_3_by_1_hotel.json, \
                            MDNRNN_LR_0.001_RNN_SIZE_128_20_num_mix_3_FLAT_acts_True_3_by_1_32_complete_setup_five.json \
                        ]')
    parser.add_argument('--extra_dim', type=int, default=0,
                        help='The extra latent dimensionality.')
    parser.add_argument('--l_size', type=int, default=96,  # 96, #32
                        help='The input dimensionality.')
    parser.add_argument('--dynamics_output_dim', type=int, default=96,  # 96, #32 for robot
                        help='The output dimensionality.')
    parser.add_argument('--dynamics_learning_rate', type=float, default=0.0003,
                        help='Dyanmic NN learning rate.')
    parser.add_argument('--dynamics_grad_clip', type=float, default=1.0,
                        help='The gradient clip size.')
    parser.add_argument('--dynamics_sequence_length', type=int, default=1,
                        help='The gradient clip size.')
    parser.add_argument('--dynamics_batch_size', type=int, default=1,
                        help='The gradient clip size.')
    parser.add_argument('--dynamics_mode', type=str, default='infer',
                        help='The gradient clip size.')
    parser.add_argument('--dynamics_num_units', type=int, default=768,  #768, # 128,
                        help='The gradient clip size.')
    parser.add_argument('--rotate', type=bool, default=False,
                        help='Rotate data')
    parser.add_argument('--degrees', type=list, default=[0], # deg,
                        help='Degrees to rotate data, negative --> clockwise')
    parser.add_argument('--preload', type=int, default=0,  # preload,
                        help='The gradient clip size.')  # controlled using others, see extra_dim too!
    parser.add_argument('--error_type', type=str, default='ade',  # error_type,
                        help='Average or Final displacement error.')
    parser.add_argument('--latent_size', type=int, default=864,  # 864, #768 + 96
                        help='The gradient clip size.')
    parser.add_argument('--oned_vae', type=bool, default=False,
                        help='Used in foxy_vc_lstm.')
    parser.add_argument('--action_include', type=bool, default=True,
                        help='Condition video prediction on actions or not.')
    parser.add_argument('--extra_linear_layer', type=bool, default=False,
                        help='used for the extra linear layer.')
    parser.add_argument('--temperature', type=float, default=0.21,
                        help='Condition video prediction on actions or not.')
    ################################ End ################################
    ######################## The Classification #########################
    parser.add_argument('--num_classes', type=int, default=3,
                        help='How many classes are there.')
    parser.add_argument('--label_include', type=bool, default=False,
                        help='How many classes are there.')
    ################################ End ################################
    ######################## The VC-LSTM #########################
    parser.add_argument('--image_dim', type=int, default=(64, 64, 3),
                        help='Image dim.')
    ################################ End ################################
    parser.add_argument('--num_units', type=int, default=128,
                        help='The number of hidden units.')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='The embedding size.')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='The decay rate used.')
    parser.add_argument('--grad_clip', type=int, default=10,
                        help='The gradient clip size.')
    parser.add_argument('--use_gpu', type=bool, default=False,
                        help='The learning rate used.')
    parser.add_argument('--gpu_num', type=str, default="-1",
                        help='The learning rate used.')
    args = parser.parse_args()

    DEFAULT_SEED = SEEDS[args.seed_no]
    if args.use_gpu and args.gpu_num == "-1":
        args.gpu_num = "0"
    elif not args.use_gpu and args.gpu_num != "-1":
        args.gpu_num = "-1"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    # use central to train/infer
    central = CentralTool(args=args, rnd=RandomState(DEFAULT_SEED))

    if central.dynamics_model != None:
        central.dynamics_model.load_json(
            os.path.join(
                os.path.join(
                    os.path.join(args.save_path, args.save_project), 'dynamics'), args.dynamics_save_name))
        print("Dynamics Successfully Loaded.")

    if args.test:
        central.unitTest()
        print("Central unit tests passed!")

    if args.mode == 'train':
        central.train()
        print("Training Complete.")
        central.save(os.path.join(os.path.join(args.save_path, args.save_project), args.save_name + '.json'))
        print("Model Saved at {0}".format(os.path.join(os.path.join(args.save_path, args.save_project), args.save_name + '.json')))

    if args.mode == 'infer':
        central.load(os.path.join(os.path.join(args.save_path, args.save_project), args.save_name + '.json'))
        print("Loaded model from {0}".format(os.path.join(os.path.join(args.save_path, args.save_project), args.save_name + '.json')))
        tot_mean_err = central.infer()
        results = []
        jsonfile = os.path.join(args.save_path, args.results_name + '.json')
        if os.path.isfile(jsonfile):
            with open(jsonfile, 'r') as f:
                past = json.load(f)
                if len(past) > 0:
                    results += past

        results.append(['{0}x{1}x{2}x{3}x{4}'.format(args.save_project, args.data_type, args.observed_length, args.predicted_length, args.seed_no),
                        ["total_err", tot_mean_err]])

        with open(jsonfile, 'wt') as outfile:
            json.dump(results, outfile, sort_keys=True, indent=0, separators=(',', ': '))

if __name__ == "__main__":
    main()
