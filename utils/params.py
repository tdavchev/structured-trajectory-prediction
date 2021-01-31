
from collections import namedtuple

ArgParams = namedtuple(
    'ArgParams', 
    [
        'mode',
        'which',
        'log_path',
        'data_path',
        'writer_name',
        'num_epochs',
        'image_dim',
        'batch_size',
        'observed_length',
        'predicted_length',
        'dynamics_num_units',
        'l_size',
        'sequence_length',
        'set_dtype',
        'input_dim',
        'output_dim',
        'max_num_agents',
        'data_type',
        'test',
        'prepare_network_tests',
        'dataset_names',
        'save_name',
        'rotate',
        'degrees',
        'preload',
        'error_type',
        'series_path',
        'test_size',
        'group_by',
        'save_path',
        'dynamics_seq_len',
        'latent_size',
        'dynamics_output_dim'
    ])

HyperParams = namedtuple(
    'HyperParams',
    [
        'model_type',
        'dimensions',
        'neighbourhood_size',
        'max_num_agents',
        'sequence_length',
        'observed_length',
        'predicted_length',
        'batch_size',
        'input_dim',
        'grid_size',
        'social_grid_include',
        'extra_linear_layer',
        'output_dim',
        'l_size',
        'num_units',
        'embedding_size',
        'learning_rate',
        'decay_rate',
        'action_include',
        'oned_vae',
        'grad_clip',
        'mode',
        'temperature',
        'extra_dim',
        'num_classes',
        'label_include'
    ])

DynamicsHyperParams = namedtuple(
    'DynamicsHyperParams',
    [
        'batch_size',
        'sequence_length',
        'observed_length',
        'predicted_length',
        'mode',
        'max_num_agents',
        'decay_rate',
        'input_dim',
        'output_dim',
        'num_units',
        'num_mixtures',
        'embedding_size',
        'learning_rate',
        'grad_clip',
        'action_include',
        'model_type',
        'temperature',
        'extra_dim'
    ])

TrajArgParams = namedtuple(
    'TrajArgParams', 
    [
        'batch_size',
        'image_dim',
        'sequence_length',
        'input_dim',
        'output_dim',
        'max_num_agents',
        'test_size',
        'group_by',
        'save_path',
        'dynamics_seq_len',
        'latent_size',
        'mode'
    ])

DataLoadParams = namedtuple(
    'DataLoadParams', 
    [
        'batch_size',
        'sequence_length',
        'mode',
        'data_type',
        'dataset_names'
    ])