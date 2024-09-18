import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.data_src.dataset_src.dataset_create import create_dataset
from src.models.base_model import Base_Model
from src.models.model_utils.U_net_CNN import UNet
from src.losses import MSE_loss, Goal_BCE_loss
from src.metrics import ADE_best_of, KDE_negative_log_likelihood, \
    FDE_best_of_goal, FDE_best_of_goal_world, FDE_best_of
from src.models.model_utils.sampling_2D_map import sampling, \
    TTST_test_time_sampling_trick
from src.models.model_utils.channel_pooling import my_channel_pooling
from src.models.model_utils.channel_pooling import create_gaussian_heatmap_template, get_patch
from torch.nn import MultiheadAttention
import torchvision.transforms as TF
#from STTL.fusion_transformer import TransformerModel
from Ablation_Study.sttl_linear_spt_and_temp.fusion_transformer import TransformerModel


class ZJL_GOAL(Base_Model):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.dataset = create_dataset(self.args.dataset)

        ##################
        # MODEL PARAMETERS
        ##################

        # Set parameters for network architecture
        self.input_size = 2  # size of the input 2: (x,y)
        self.embedding_size = 64  # embedding dimension
        self.nhead = 8  # number of heads in multi-head attentions TF(transformer)
        self.d_hidden = 2048  # hidden dimension in the TF encoder layer
        self.n_layers = 1  # number of TransformerEncoderLayers
        self.dropout_prob = 0  # the dropout probability value
        self.add_noise_traj = self.args.add_noise_traj
        self.noise_size = 16  # size of random noise vector
        self.output_size = 2  # output size
        self.dropout_TF_prob = 0.1  # dropout in transformer encoder layer

        # Goal module parameters
        #self.num_image_channels = 6

        # Extra information to concat goals: time, last position, prediction final positions,and distance to predicted goals
        #self.extra_features = 4

        # U-net encoder channels
        #self.enc_chs = (self.num_image_channels + self.args.obs_length, 32, 32, 64, 64, 64)

        # U-net decoder channels
        #self.dec_chs = (64, 64, 64, 32, 32)

        # #######################
        # # Upper branch of Model
        # #######################
        #
        # # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(self.input_size, self.embedding_size)
        self.relu = nn.ReLU()
        self.dropout_input_temporal = nn.Dropout(self.dropout_prob)

        self.input_embedding_layer_spatial = nn.Linear(self.input_size, self.embedding_size)
        self.relu = nn.ReLU()
        self.dropout_input_spatial = nn.Dropout(self.dropout_prob)

        self.project = nn.Linear(2, 64)


        #########################
        # fusion and output layer
        #########################
        self.fusion_layer = nn.Linear(64, 32)
        if self.add_noise_traj:
            self.output_layer = nn.Linear(80, self.output_size)
        else:
            self.output_layer = nn.Linear(80, self.output_size)

    def forward(self, inputs, ground_truth, num_samples=1, if_test=False):

        gt_traj = ground_truth.permute(1, 0, 2)
        batch_coords = inputs["abs_pixel_coord"]
        # Number of agent in current batch abs_world
        seq_length, num_agents, _ = batch_coords.shape
        if self.args.shift_last_obs:
            shift = batch_coords[self.args.obs_length - 1]
        else:
            shift = torch.zeros_like(batch_coords[0])
        batch_coords = (batch_coords - shift) / self.args.traj_normalization


        outputs = torch.zeros(seq_length, num_agents, self.output_size).to(self.device)

        # Add observation as first output
        outputs[0:self.args.obs_length] = batch_coords[0:self.args.obs_length]

        # Add noise
        noise = torch.randn((1, self.noise_size)).to(self.device)

        for frame_idx in range(self.args.obs_length, self.args.obs_length + self.args.pred_length):

            all_outputs = []
            #all_aux_outputs = []
            current_agents = batch_coords[:frame_idx]

            ##################
            # RECURRENT MODULE
            ##################

            # Input embedding
            input_embedded = self.dropout_input_temporal(self.relu(
                self.input_embedding_layer_temporal(current_agents)))  # (frame, 32, 32)

            ############################################
                            #STTL#
            ############################################
            fusion_transformer = TransformerModel(input_embedded,
                                                  self.embedding_size,
                                                  self.nhead,
                                                  self.d_hidden,
                                                  self.n_layers,
                                                  self.dropout_TF_prob)
            fusion_transformer = fusion_transformer.to(self.device)
            fusion_transformer_out = fusion_transformer(input_embedded, mask=False)

            # fusion_feat
            if self.add_noise_traj:
                # Concatenate noise to fusion output
                noise_to_cat = noise.repeat(fusion_transformer_out.shape[0], 1)  # (32, 16)
                fusion_feat = torch.cat((fusion_transformer_out, noise_to_cat), dim=1)  # (32, 80)

            # Output predict coordinates
            outputs_current = self.output_layer(fusion_feat)

            # Append to outputs
            outputs[frame_idx] = outputs_current

        # shift normalize back
        outputs = outputs * self.args.traj_normalization + shift
        all_outputs.append(outputs)

        # stack predictions
        all_outputs = torch.stack(all_outputs)
        # from list of dict to dict of list (and then tensors)
        all_aux_outputs = []

        return all_outputs, all_aux_outputs