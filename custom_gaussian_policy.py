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

        # Applica il reshape se necessario
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

        # Reinizializziamo action_net per generare mu e sigma
        action_dim = self.action_space.shape[0]
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim * 2)

        # Assicuriamoci che la rete sia sul dispositivo corretto
        self.action_net.to(self.device)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        # Assicuriamoci che latent_pi sia sul dispositivo corretto
        latent_pi = latent_pi.to(self.device)

        # Genera i parametri mu e sigma
        mean_actions = self.action_net(latent_pi)

        action_dim = self.action_space.shape[0]
        mu = mean_actions[..., :action_dim]  # Prima metà degli output è mu
        log_std = mean_actions[..., action_dim:]  # Seconda metà è log_std

        # Limita il range di log_std per stabilità
        log_std = th.clamp(log_std, -20, 2) # prova anche togliendolo per essere più fair
        std = th.exp(log_std)

        # Passa anche l'action_space alla distribuzione
        return GaussianDistribution(mu, std, action_space=self.action_space)

    def forward(self, obs, deterministic=False):
        # Assicuriamoci che le osservazioni siano sul dispositivo corretto
        obs = obs.to(self.device)

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Campiona dalla distribuzione
        actions = distribution.get_actions(deterministic=deterministic)

        # Già nella forma corretta grazie a get_actions
        log_prob = distribution.log_prob(actions).sum(dim=1)

        # Clipping delle azioni nell'intervallo [-1, 1]
        actions = th.clamp(actions, -1.0, 1.0) # potrebbe però generare del bias rispetto alla distribuzione Beta

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        # Assicuriamoci che le osservazioni siano sul dispositivo corretto
        obs = obs.to(self.device)

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return values, log_prob, entropy