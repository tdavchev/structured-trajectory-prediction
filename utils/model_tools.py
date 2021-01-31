from models.basic_lstm import BasicLSTM
from models.dynamic_mdnlstm import DynamicMDNLSTM
from models.just_lstm import JustLSTM
from models.lstm_map_conditioned3 import MapCond3LSTM
from models.social_lstm import SocialLSTM

from utils.params import DynamicsHyperParams, HyperParams


class ModelTool(object):
    def __init__(self, rnd):
        super(ModelTool, self).__init__()
        self.rnd = rnd

    @property
    def hps(self):
        """The data dictionary we will use."""
        return self.__hps

    @property
    def dynamics_hps(self):
        """The data dictionary we will use."""
        return self.__dynamics_hps

    @hps.setter
    def hps(self, args):
        self.__hps = HyperParams(
            neighbourhood_size=args.neighbourhood_size,
            model_type=args.model_type,
            dimensions=args.dimensions,
            max_num_agents=args.max_num_agents,
            extra_dim=self.assign_extra_dim(args.save_name, args.dynamics_num_units, args.l_size),
            sequence_length=args.sequence_length,
            observed_length=args.observed_length,
            predicted_length=args.predicted_length,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            l_size=args.l_size,
            grid_size=args.grid_size,
            num_units=args.num_units,
            embedding_size=args.embedding_size,
            learning_rate=args.learning_rate,
            decay_rate=args.decay_rate,
            grad_clip=args.grad_clip,
            action_include=args.action_include,
            extra_linear_layer=args.extra_linear_layer,
            oned_vae=args.oned_vae,
            social_grid_include=args.social_grid_include,
            mode=args.mode,
            temperature=args.temperature,
            num_classes=args.num_classes,
            label_include=args.label_include)

        self.__dynamics_hps = DynamicsHyperParams(
            batch_size=args.dynamics_batch_size,
            sequence_length=args.dynamics_sequence_length,
            observed_length=args.observed_length,
            predicted_length=args.predicted_length,
            mode=args.dynamics_mode,
            max_num_agents=args.max_num_agents,
            extra_dim=self.assign_extra_dim(args.save_name, args.dynamics_num_units, args.l_size),
            decay_rate=args.decay_rate,
            input_dim=args.l_size,
            output_dim=args.dynamics_output_dim,
            num_units=args.dynamics_num_units,
            num_mixtures=args.num_mixtures,
            embedding_size=args.max_num_agents * 2,
            learning_rate=args.dynamics_learning_rate,
            grad_clip=args.dynamics_grad_clip,
            action_include=args.action_include,
            model_type=args.dynamics_model_type,
            temperature=args.temperature
        )

    @staticmethod
    def assign_extra_dim(name, num_units, input_dim):
        if '_d_' in name:
             return num_units
        elif '_next_' in name and '_rd_' not in name:
            return input_dim
        elif '_r_' in name:
            return input_dim
        else:
            return num_units + input_dim

    @property
    def model(self):
        """The data dictionary we will use."""
        return self.__model

    @model.setter
    def model(self, name):
        if self.hps.model_type in ['per_agent_LSTM', 'per_frame_LSTM', 'per_frame_social_LSTM', 'frame_classify_LSTM']:
            self.reset_graph()

        self.__model = self.chooseModel(name)

    @property
    def dynamics_model(self):
        """The data dictionary we will use."""
        return self.__dynamics_model

    @dynamics_model.setter
    def dynamics_model(self, name):
        if self.hps.model_type == 'map_cond_per_frame_LSTM' and name is not None:
            self.reset_graph()

        self.__dynamics_model = self.chooseDynamicsModel(name)

    def chooseModel(self, name):
        if name == 'per_frame_LSTM':
            return JustLSTM(self.hps, self.rnd)
        elif name == 'per_agent_LSTM':
            return BasicLSTM(self.hps, self.rnd)
        elif name == 'map_cond_per_frame_LSTM':
            return MapCond3LSTM(self.hps, self.rnd)
        elif name == 'map_cond_per_frame_LSTM_softmax':
            return MapCond3LSTMSoftmax(self.hps, self.rnd)
        elif name == 'per_frame_social_LSTM':
            return SocialLSTM(self.hps, self.rnd)
        elif name == 'per_frame_classify_LSTM':
            return JustLSTMClassify(self.hps, self.rnd)
        elif name == 'per_frame_vc_LSTM':
            return VcLSTM(self.hps, self.rnd)
        elif name == 'per_frame_foxy_vc_LSTM':
            return FoxyVcLSTM(self.hps, self.rnd)
        # elif name == 'dilated_conv':
        #     return Model(self.hps, self.rnd)

    def chooseDynamicsModel(self, name):
        if name == 'mdn_rnn':
            return DynamicMDNLSTM(self.dynamics_hps, self.rnd)

    @staticmethod
    def reset_graph():
        from tensorflow.python.framework import ops

        if 'sess' in globals() and sess:
            print("********THERE WAS A SESSION IN GLOBALS!!!!!!!!!!!!!!!!!!!!!!!!!")
            sess.close()

        ops.reset_default_graph()
        print("graph successfully reset")
