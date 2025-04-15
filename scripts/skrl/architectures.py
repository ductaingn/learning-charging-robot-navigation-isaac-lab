from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class FDM(nn.Module):
    def __init__(self, lidar_dim, output_dim, *args, **kwargs):
        super(FDM, self).__init__(*args, **kwargs)

        self.history_embedding = nn.Linear(lidar_dim, 128)
        self.current_embedding = nn.Linear(lidar_dim, 128)

        self.tfm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, nhead=4, dropout=0.1, dim_feedforward=512, batch_first=True
            ),
            num_layers=1,
        )

        self.layer_norm = nn.LayerNorm(128)

        self.encode = nn.Linear(128, output_dim)

    def forward(self, history_observation, current_observation):
        """
        history_observation: shape = (batch_size, n_history_frame, lidar_dim)
        current_observation: shape = (batch_size, 1, lidar_dim)

        out: shape = (batch_size, output_dim)
        """
        history_observation = self.history_embedding(history_observation)

        # hs, self.hidden_his_state = self.lstm(history_observation)
        hs = F.leaky_relu(history_observation)

        co = F.leaky_relu(self.current_embedding(current_observation))

        concat = torch.cat([hs, co], dim=1)

        out = self.tfm(concat)

        out = self.encode(out)

        return out


class KineticModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_waypoints, *args, **kwargs):
        super(KineticModel, self).__init__(*args, **kwargs)

        self.kinetic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim // 2),
        )

        self.waypoint_encoder = nn.Linear(n_waypoints * 2, output_dim // 4)

        self.goal_encoder = nn.Linear(2, output_dim // 4)

    def forward(self, kinetic_state, waypoints, goal):
        n_envs = kinetic_state.shape[0]
        kinetic_state = self.kinetic(kinetic_state)

        waypoints = self.waypoint_encoder(waypoints)

        goal = self.goal_encoder(goal)

        return torch.cat([kinetic_state, waypoints, goal], dim=-1)


class BackBone(nn.Module):
    def __init__(
        self,
        lidar_dim,
        kin_input_dim,
        kin_output_dim,
        latent_dim,
        n_history_frame,
        n_waypoints,
        *args,
        **kwargs
    ):
        super(BackBone, self).__init__(*args, **kwargs)
        self.lidar_dim = lidar_dim
        self.kin_input_dim = kin_input_dim
        self.n_history_frame = n_history_frame
        self.latent_dim = latent_dim

        self.fdm = FDM(lidar_dim, kin_output_dim)
        self.kinetic_model = KineticModel(kin_input_dim, kin_output_dim, n_waypoints)

        self.mha = nn.MultiheadAttention(
            kin_output_dim, 4, dropout=0.1, batch_first=True
        )

        self.encode = nn.Sequential(
            nn.Linear(kin_output_dim, latent_dim),
            nn.LeakyReLU(),
        )

        self.apply(weights_init_)

    def forward(self, observation: torch.Tensor):
        """
        Processes the input observation tensor to compute a latent representation for the robot's navigation.

        Parameters:
            observation (torch.Tensor): A tensor with shape
                (batch_size, kin_input_dim + lidar_dim*(n_history_frame+1) + 2 + 2*n_waypoints).
                The tensor is organized as follows:
                  - The first kin_input_dim elements represent the robot kinetic state.
                  - The next lidar_dim elements correspond to the current lidar scan.
                  - The following (lidar_dim * n_history_frame) elements are historical lidar scans.
                  - The subsequent 2 elements encode goal coordinates.
                  - The remaining elements represent the waypoints (each waypoint defined by 2 values).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_dim) representing the latent features.
            The latent features are obtained by processing the kinetic state, lidar data, goals, and waypoints
            through dedicated network modules (fdm, kinetic_model), integrating them via multi-head attention,
            and finally encoding the combined output.
        """
        kinetic_state_start_index = 0
        current_lidar_start_index = kinetic_state_start_index + self.kin_input_dim
        history_lidar_start_index = current_lidar_start_index + self.lidar_dim
        goal_start_index = (
            history_lidar_start_index + self.n_history_frame * self.lidar_dim
        )
        waypoints_start_index = goal_start_index + 2

        kinetic_state = observation[
            :, kinetic_state_start_index:current_lidar_start_index
        ]
        current_lidar = observation[
            :, current_lidar_start_index:history_lidar_start_index
        ]
        lidar_observation = observation[:, history_lidar_start_index:goal_start_index]
        goals = observation[:, goal_start_index:waypoints_start_index]
        waypoints = observation[:, waypoints_start_index:]

        kinetic_state = kinetic_state.reshape(-1, 1, self.kin_input_dim)
        lidar_observation = lidar_observation.reshape(
            -1, self.n_history_frame, self.lidar_dim
        )
        current_lidar = current_lidar.reshape(-1, 1, self.lidar_dim)
        n_envs = kinetic_state.shape[0]
        waypoints = waypoints.reshape(n_envs, 1, -1)
        goals = goals.reshape(n_envs, 1, 2)

        env_rep = self.fdm.forward(lidar_observation, current_lidar)
        robot_rep = self.kinetic_model.forward(kinetic_state, waypoints, goals)

        concat = torch.cat([env_rep, robot_rep], dim=1)

        out, _ = self.mha(concat, concat, concat)

        latent = self.encode(out)
        latent = torch.sum(latent, dim=1).reshape(-1, self.latent_dim)

        return latent


class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space:Box, action_space:Box, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.latent_dim = 256
        self.feature_extractor = BackBone(lidar_dim=360, kin_input_dim=7, kin_output_dim=128, 
                            latent_dim=self.latent_dim, n_history_frame=16, n_waypoints=10)
        self.mu = nn.Linear(self.latent_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        feature = self.feature_extractor(inputs["states"])
        mu = self.mu(F.leaky_relu(feature))
        return mu, self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space:Box, action_space:Box, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.latent_dim = 256
        self.feature_extractor = BackBone(lidar_dim=360, kin_input_dim=7, kin_output_dim=128, 
                            latent_dim=self.latent_dim, n_history_frame=16, n_waypoints=10)
        
        self.action_dim = int(np.prod(action_space.shape))

        self.q_net = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256,1)
        )

        self.q_net.to(self.device)

    def compute(self, inputs, role):
        feature = self.feature_extractor(inputs["states"])
        qvalue_input = torch.cat([feature, inputs["taken_actions"]], dim=1)
        return self.q_net(qvalue_input), {}
    