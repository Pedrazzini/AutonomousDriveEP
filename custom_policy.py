import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Beta
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BetaDistribution(Beta):
    def __init__(self, concentration1, concentration0, action_space=None):
        super().__init__(concentration1, concentration0)
        self.action_space = action_space

    def get_actions(self, deterministic=False):
        if deterministic:
            # for the Beta distribution the mode is:
            # (alpha - 1) / (alpha + beta - 2) with alpha and beta > 1
            alpha = self.concentration1
            beta = self.concentration0

            mode = th.where(
                (alpha > 1) & (beta > 1),
                (alpha - 1) / (alpha + beta - 2),
                th.full_like(alpha, 0.5)
            )
            mode = mode * 2.0 - 1.0

            # apply reshape as done in forward function
            if hasattr(self, 'action_space') and self.action_space is not None:
                mode = mode.reshape((-1, *self.action_space.shape))

            return mode
        else:
            actions = self.sample()
            scaled_actions = (actions * 2.0 - 1.0).reshape((-1, *self.action_space.shape))
            return actions


class BetaPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
    ):
        super(BetaPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )

        # the final layer of the CNN are the neurons alpha and beta
        action_dim = self.action_space.shape[0]
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim * 2)

        # go to the right device
        self.action_net.to(self.device)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        latent_pi = latent_pi.to(self.device)


        mean_actions = self.action_net(latent_pi)

        action_dim = self.action_space.shape[0]
        alpha = th.log(1 + th.exp(mean_actions[..., :action_dim])) + 1.0  #  alpha > 1
        beta = th.log(1 + th.exp(mean_actions[..., action_dim:])) + 1.0  # beta > 1


        return BetaDistribution(alpha, beta, action_space=self.action_space)

    def forward(self, obs, deterministic=False):

        obs = obs.to(self.device)

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # sample action from the distribution
        actions = distribution.sample()

        log_prob = distribution.log_prob(actions).sum(dim=1)

        # rescale to [-1, 1]
        scaled_actions = (actions * 2.0 - 1.0).reshape((-1, *self.action_space.shape))

        return scaled_actions, values, log_prob

    def evaluate_actions(self, obs, actions):

        obs = obs.to(self.device)

        # rescale to [0, 1]
        rescaled_actions = (actions + 1.0) / 2.0

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(rescaled_actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return values, log_prob, entropy