import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GaussianDistribution(Normal):
    def __init__(self, mean, std, action_space=None):
        super().__init__(mean, std)
        self.action_space = action_space

    def get_actions(self, deterministic=False):
        if deterministic:
            actions = self.mean
        else:
            actions = self.sample()

        # apply reshape
        if hasattr(self, 'action_space') and self.action_space is not None:
            actions = actions.reshape((-1, *self.action_space.shape))

        return actions


class GaussianPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
    ):
        super(GaussianPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )

        # generate mu and sigma
        action_dim = self.action_space.shape[0]
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim * 2)

        # net on correct device
        self.action_net.to(self.device)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):

        latent_pi = latent_pi.to(self.device)

        # generate mu and sigma
        mean_actions = self.action_net(latent_pi)

        action_dim = self.action_space.shape[0]
        mu = mean_actions[..., :action_dim]  # first half of the output is mu
        log_std = mean_actions[..., action_dim:]  # second half is log_std

        # constraint the range for stability
        log_std = th.clamp(log_std, -20, 2)
        std = th.exp(log_std)

        # pass action space to the distribution
        return GaussianDistribution(mu, std, action_space=self.action_space)

    def forward(self, obs, deterministic=False):
        obs = obs.to(self.device)

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # sample from distribution
        actions = distribution.get_actions(deterministic=deterministic)


        log_prob = distribution.log_prob(actions).sum(dim=1)

        # clipping interval [-1, 1]
        actions = th.clamp(actions, -1.0, 1.0)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):

        obs = obs.to(self.device)

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return values, log_prob, entropy