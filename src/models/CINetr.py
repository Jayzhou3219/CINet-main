from operator import ge
from turtle import forward
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
from STTL.fusion_transformer import TransformerModel
#from Ablation_Study.sttl_linear_spt_and_temp.fusion_transformer import TransformerModel


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
        self.embedding_size = 32  # embedding dimension
        self.nhead = 8  # number of heads in multi-head attentions TF(transformer)
        self.d_hidden = 2048  # hidden dimension in the TF encoder layer
        self.n_layers = 1  # number of TransformerEncoderLayers
        self.dropout_prob = 0  # the dropout probability value
        self.add_noise_traj = self.args.add_noise_traj
        self.noise_size = 32  # size of random noise vector
        self.output_size = 2  # output size
        self.dropout_TF_prob = 0.1  # dropout in transformer encoder layer

        # Goal module parameters
        self.num_image_channels = 6

        # Extra information to concat goals: time, last position, prediction final positions,and distance to predicted goals
        self.extra_features = 4

        # U-net encoder channels
        self.enc_chs = (self.num_image_channels + self.args.obs_length, 32, 32, 64, 64, 64)

        # U-net decoder channels
        self.dec_chs = (64, 64, 64, 32, 32)

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
        #
        # # Temporal Transformer layer to extract temporal information of pedestrians
        # self.temporal_encoder_layer = TransformerEncoderLayer(
        #     d_model=self.embedding_size + self.extra_features * self.nhead,
        #     nhead=self.nhead,
        #     dim_feedforward=self.d_hidden,
        #     dropout=self.dropout_TF_prob)
        # self.temporal_encoder = TransformerEncoder(self.temporal_encoder_layer, num_layers=self.n_layers_temporal)
        #
        # # Spatial Transformer layer to extract spatial information of pedestrians
        # self.spatial_encoder_layer = TransformerEncoderLayer(
        #     d_model=self.embedding_size + self.extra_features * self.nhead,
        #     nhead=self.nhead,
        #     dim_feedforward=self.d_hidden,
        #     dropout=self.dropout_TF_prob)
        # self.spatial_encoder = TransformerEncoder(self.spatial_encoder_layer, num_layers=self.n_layers_temporal)

        #######################
        # Lower branch of Model
        #######################
        self.goal_module = UNet(
            enc_chs=(self.num_image_channels + self.args.obs_length,
                     32, 32, 64, 64, 64),
            dec_chs=(64, 64, 64, 32, 32),
            out_chs=self.args.pred_length)

        #######################
        # Cross attention
        #######################
        self.cross_atten = MultiheadAttention(embed_dim=64, num_heads=self.nhead, dropout=self.dropout_TF_prob)

        #########################
        # fusion and output layer
        #########################
        if self.add_noise_traj:
            self.output_layer = nn.Linear(96, self.output_size)
        else:
            self.output_layer = nn.Linear(64, self.output_size)

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

        ##################
        # PREDICT GOAL
        ##################

        # Extract precomputed map for goal goal_idx
        tensor_image = inputs["tensor_image"].unsqueeze(0).repeat(num_agents, 1, 1, 1)
        obj_traj_maps = inputs["input_traj_maps"][:, 0:self.args.obs_length]
        input_goal_module = torch.cat((tensor_image, obj_traj_maps), dim=1)  # trajectory_on_scene

        # Compute goal maps
        goal_logit_map_start = self.goal_module(input_goal_module)  # (32, 12, 64, 64)
        goal_prob_map = torch.sigmoid(goal_logit_map_start[:, -1].unsqueeze(1) / self.args.sampler_temperature)  # (32,1,64,64)

        # Sampling
        if self.args.use_ttst and num_samples > 3:  # use_ttst:Use Test Time Sampling Trick
            goal_point_start = TTST_test_time_sampling_trick(
                goal_prob_map,
                num_goals=num_samples,
                device=self.device)
            goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2)
        else:
            goal_point_start = sampling(goal_prob_map, num_samples=num_samples)  # (32, 1, 1, 2)
            goal_point_start = goal_point_start.squeeze(1)  # (32, 1, 2)
            goal_point_start_ = self.project(goal_point_start)

        # Start samples loop
        all_outputs = []
        all_aux_outputs = []
        for sample_idx in range(num_samples):
            # output tensor of shape (seq_length, N, 2)
            outputs = torch.zeros(seq_length, num_agents, self.output_size).to(self.device)

            # add observation as first output
            outputs[0:self.args.obs_length] = batch_coords[0:self.args.obs_length]

            # create noise vector to promote different trajectorirs
            noise = torch.randn((1, self.noise_size)).to(self.device)

            if if_test:
                goal_point = goal_point_start[:, sample_idx]
            else:
                goal_point = batch_coords[-1]

            # list of auxiliary outputs
            aux_outputs = {
                "goal_logit_map": goal_logit_map_start,
                "goal_point": goal_point}

            ###################################
            # loop over seq_length-1 frames, strating from frame 8
            ###################################
            for frame_idx in range(self.args.obs_length, self.args.seq_length):
                # If testing phase and frame >= obs_length (prediction)
                if if_test and frame_idx >= self.args.obs_length:
                    # Get current agents positions: from 0 to obs_length
                    # Take GT, then previous predicted positions
                    current_agents = torch.cat(
                        (batch_coords[:self.args.obs_length],
                         outputs[self.args.obs_length:frame_idx]))
                else:  # Train phase or frame < obs_length (observation)
                    # Current agents positions
                    current_agents = batch_coords[:frame_idx]

                ##################
                # RECURRENT MODULE
                ##################

                # Input embedding
                temporal_input_embedded = self.dropout_input_temporal(self.relu(
                    self.input_embedding_layer_temporal(current_agents)))

                # Additional informatioan for ZJL_GOAL
                last_positions = current_agents[-1]  # (32, 2)
                current_time_step = torch.full(size=(last_positions.shape[0], 1),  # (32, 1)
                                               fill_value=frame_idx).to(self.device)
                distance_to_goal = goal_point_start.squeeze(1) - last_positions  # (32, 2)

                last_positions_to_cat = last_positions.repeat(frame_idx, 1, self.nhead // 2)  # (frame_idx, 32, 8)
                current_time_step_to_cat = current_time_step.repeat(frame_idx, 1, self.nhead)  # (frame_idx, 32, 8)
                distance_to_goal_to_cat = distance_to_goal.repeat(frame_idx, 1, self.nhead // 2)  # (frame_idx, 32, 8)
                goal_point_start_to_cat = goal_point_start.squeeze(1).repeat(frame_idx, 1,
                                                                             self.nhead // 2)  # (frame_idx, 32, 8)

                additional_info_upper = torch.cat((temporal_input_embedded,
                                                   last_positions_to_cat,
                                                   current_time_step_to_cat,
                                                   distance_to_goal_to_cat,
                                                   goal_point_start_to_cat), dim=2)  # (frame_idx, 32, 64)

                ############################################
                fusion_transformer = TransformerModel(additional_info_upper,
                                                      self.embedding_size,
                                                      self.nhead,
                                                      self.d_hidden,
                                                      self.n_layers,
                                                      self.dropout_TF_prob)
                fusion_transformer = fusion_transformer.to(self.device)
                fusion_transformer_out = fusion_transformer(additional_info_upper, mask=False)  # (32,64)

                # convert Cross-attention to concatenate
                # attn_output_up, _ = self.cross_atten(
                #     goal_point_start_,
                #     fusion_transformer_out.unsqueeze(1),
                #     fusion_transformer_out.unsqueeze(1))  # (32,1,64)
                # attn_output_final = attn_output_up.squeeze(1)

                # fusion_feat
                if self.add_noise_traj:
                    # Concatenate noise to fusion output
                    noise_to_cat = noise.repeat(fusion_transformer_out.shape[0], 1)  # (1,32) -->  (32, 32)
                    fusion_feat = torch.cat((fusion_transformer_out, noise_to_cat), dim=1)  # (32, 96)

                # Output predict coordinates
                outputs_current = self.output_layer(fusion_transformer_out)

                # Append to outputs
                outputs[frame_idx] = outputs_current

                # shift normalize back
            outputs = outputs * self.args.traj_normalization + shift
            all_outputs.append(outputs)
            all_aux_outputs.append(aux_outputs)

            # stack predictions
            all_outputs = torch.stack(all_outputs)
            # from list of dict to dict of list (and then tensors)
            all_aux_outputs = {k: torch.stack([d[k] for d in all_aux_outputs])
                               for k in all_aux_outputs[0].keys()}

            return all_outputs, all_aux_outputs

    def prepare_inputs(self, batch_data, batch_id):
        """
        Prepare inputs to be fed to a generic model.
        """
        input_traj_maps = batch_data['input_traj_maps']
        input_traj_maps_crop = TF.CenterCrop((64, 64))(input_traj_maps)

        tensor_image = batch_data['tensor_image']
        tensor_image_crop = TF.CenterCrop((64, 64))(tensor_image)

        entries_to_remove = ['input_traj_maps']
        for key in entries_to_remove:
            if key in batch_data:
                del batch_data[key]

        entries_to_remove_ = ['tensor_image']
        for value in entries_to_remove_:
            if value in batch_data:
                del batch_data[value]

        batch_data["input_traj_maps"] = input_traj_maps_crop
        batch_data["tensor_image"] = tensor_image_crop

        # we need to remove first dimension which is added by torch.DataLoader
        # float is needed to convert to 32bit float
        selected_inputs = {k: v.squeeze(0).float().to(self.device) if \
            torch.is_tensor(v) else v for k, v in batch_data.items()}
        # extract seq_list
        seq_list = selected_inputs["seq_list"]
        # decide which is ground truth
        ground_truth = selected_inputs["abs_pixel_coord"]

        scene_name = batch_id["scene_name"][0]
        scene = self.dataset.scenes[scene_name]
        selected_inputs["scene"] = scene

        return selected_inputs, ground_truth, seq_list

    def init_losses(self):
        losses = {
            "traj_MSE_loss": 0,
            "goal_BCE_loss": 0,
        }
        return losses

    def set_losses_coeffs(self):
        losses_coeffs = {
            "traj_MSE_loss": 1,
            "goal_BCE_loss": 1e6,
        }
        return losses_coeffs

    def init_train_metrics(self):
        train_metrics = {
            "ADE": [],
            "FDE": [],
            "ADE_world": [],
            "FDE_world": [],
        }
        return train_metrics

    def init_test_metrics(self):
        test_metrics = {
            "ADE": [],
            "FDE": [],
            "ADE_world": [],
            "FDE_world": [],
            # "NLL": [],
        }
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            # "ADE": 1e9,
            # "FDE": 1e9,
            "ADE_world": 1e9,
            "FDE_world": 1e9,
            "goal_BCE_loss": 1e9,
        }
        return best_metrics

    def best_valid_metric(self):
        return "FDE_world"

    # def compute_model_losses(self,
    #                          outputs,
    #                          ground_truth,
    #                          loss_mask,
    #                          inputs,
    #                          aux_outputs):
    #     """
    #     Compute loss for a generic model.
    #     """
    #     out_maps_GT_goal = inputs["input_traj_maps"][:, self.args.obs_length:]
    #     goal_logit_map = aux_outputs["goal_logit_map"]
    #     goal_BCE_loss = Goal_BCE_loss(
    #         goal_logit_map, out_maps_GT_goal, loss_mask)
    #
    #     losses = {
    #         "traj_MSE_loss": MSE_loss(outputs, ground_truth, loss_mask),
    #         "goal_BCE_loss": goal_BCE_loss,
    #     }
    #
    #     return losses

    def compute_model_losses(self,
                             outputs,
                             ground_truth,
                             loss_mask,
                             inputs,
                             aux_outputs):
        #gt_future_map = aux_outputs["gt_future_map"]

       # goal_logit_map = aux_outputs["goal_logit_map"]

        out_maps_GT_goal = inputs["input_traj_maps"][:, self.args.obs_length:]
        goal_logit_map = aux_outputs["goal_logit_map"]

        goal_BCE_loss = Goal_BCE_loss(goal_logit_map, out_maps_GT_goal, loss_mask)

        losses = {
            "traj_MSE_loss": MSE_loss(outputs, ground_truth, loss_mask),
            "goal_BCE_loss": goal_BCE_loss, }

        return losses

    def compute_model_metrics(self,
                              metric_name,
                              phase,
                              predictions,
                              ground_truth,
                              metric_mask,
                              all_aux_outputs,
                              inputs,
                              obs_length=8):
        """
        Compute model metrics for a generic model.
        Return a list of floats (the given metric values computed on the batch)
        """
        if phase == 'test':
            compute_nll = self.args.compute_test_nll
            num_samples = self.args.num_test_samples
        elif phase == 'valid':
            compute_nll = self.args.compute_valid_nll
            num_samples = self.args.num_valid_samples
        else:
            compute_nll = False
            num_samples = 1

        # scale back to original dimension
        predictions = predictions.detach() * self.args.down_factor
        ground_truth = ground_truth.detach() * self.args.down_factor

        # convert to world coordinates
        scene = inputs["scene"]
        scene_name = scene.name
        pred_world = []
        for i in range(predictions.shape[0]):
            pred_world.append(scene.make_world_coord_torch(predictions[i]))
        pred_world = torch.stack(pred_world)

        GT_world = scene.make_world_coord_torch(ground_truth)

        if metric_name == 'ADE':
            return ADE_best_of(
                phase, scene_name, predictions, ground_truth, metric_mask, obs_length)
        elif metric_name == 'FDE':
            # return FDE_best_of_goal(all_aux_outputs, ground_truth,
            #                         metric_mask, self.args)
            return FDE_best_of_goal(phase, scene_name, predictions, ground_truth,
                                     metric_mask, self.args)
        if metric_name == 'ADE_world':
            return ADE_best_of(
                phase, scene_name, pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'FDE_world':
            return FDE_best_of_goal_world(phase, scene_name, pred_world, scene, GT_world, metric_mask, self.args)
        elif metric_name == 'NLL':
            if compute_nll and num_samples > 1:
                return KDE_negative_log_likelihood(
                    predictions, ground_truth, metric_mask, obs_length)
            else:
                return [0, 0, 0]
        else:
            raise ValueError("This metric has not been implemented yet!")
