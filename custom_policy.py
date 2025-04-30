import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Beta
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BetaDistribution(Beta):
    def get_actions(self, deterministic=False):
        if deterministic:
            # Per la distribuzione Beta, la moda è:
            # (alpha - 1) / (alpha + beta - 2) quando sia alpha che beta sono > 1
            alpha = self.concentration1
            beta = self.concentration0
            # Gestisci il caso in cui alpha o beta siano <= 1
            mode = th.where(
                (alpha > 1) & (beta > 1),
                (alpha - 1) / (alpha + beta - 2),
                th.full_like(alpha, 0.5)  # Valore di default se non si può calcolare la moda
            )
            return mode
        else:
            return self.sample()


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

        # Reinizializziamo action_net per generare alpha e beta
        action_dim = self.action_space.shape[0]
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim * 2)

        # Assicuriamoci che la rete sia sul dispositivo corretto
        self.action_net.to(self.device)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        # Assicuriamoci che latent_pi sia sul dispositivo corretto
        latent_pi = latent_pi.to(self.device)

        # Genera i parametri alpha e beta
        mean_actions = self.action_net(latent_pi)

        action_dim = self.action_space.shape[0]
        alpha = th.log(1 + th.exp(mean_actions[..., :action_dim])) + 1.0  # Assicura che alpha > 1
        beta = th.log(1 + th.exp(mean_actions[..., action_dim:])) + 1.0  # Assicura che beta > 1

        return BetaDistribution(alpha, beta)

    def forward(self, obs, deterministic=False):
        # Assicuriamoci che le osservazioni siano sul dispositivo corretto
        obs = obs.to(self.device)

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Campiona dalla distribuzione
        actions = distribution.sample()

        log_prob = distribution.log_prob(actions).sum(dim=1)

        # La distribuzione Beta restituisce valori tra 0 e 1, riscaliamo a [-1, 1]
        scaled_actions = (actions * 2.0 - 1.0).reshape((-1, *self.action_space.shape))

        return scaled_actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        # Assicuriamoci che le osservazioni siano sul dispositivo corretto
        obs = obs.to(self.device)

        # Riscala le azioni da [-1, 1] a [0, 1] per la distribuzione Beta
        rescaled_actions = (actions + 1.0) / 2.0

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(rescaled_actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return values, log_prob, entropy