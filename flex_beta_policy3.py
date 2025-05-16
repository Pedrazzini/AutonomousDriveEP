import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FlexibleBetaDistribution(Distribution):
    def __init__(self, alpha1, alpha2, tau, p, action_space=None):
        """
        Distribuzione FlexibleBeta: Y ∼ p*Beta(α1+τ,α2) + (1−p)*Beta(α1, α2+τ)

        Args:
            alpha1: Primo parametro di concentrazione della distribuzione Beta
            alpha2: Secondo parametro di concentrazione della distribuzione Beta
            tau: Parametro di modificazione per le distribuzioni Beta
            p: Peso per la combinazione lineare delle due distribuzioni Beta
            action_space: Spazio d'azione
        """
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.tau = tau
        self.p = p
        self.action_space = action_space
        self.device = alpha1.device

    def sample(self):
        """
        Campiona dalla distribuzione FlexibleBeta.
        """
        # Crea i parametri per le due distribuzioni Beta
        beta1_alpha = self.alpha1 + self.tau
        beta1_beta = self.alpha2

        beta2_alpha = self.alpha1
        beta2_beta = self.alpha2 + self.tau

        # Campiona dalle due distribuzioni Beta
        beta1_dist = th.distributions.Beta(beta1_alpha, beta1_beta)
        beta2_dist = th.distributions.Beta(beta2_alpha, beta2_beta)

        beta1_sample = beta1_dist.sample()
        beta2_sample = beta2_dist.sample()

        # Combinazione lineare delle due distribuzioni
        samples = self.p * beta1_sample + (1 - self.p) * beta2_sample

        return samples

    def log_prob(self, value):
        """
        Calcola il logaritmo della densità di probabilità per il valore dato.

        Args:
            value: Valore per cui calcolare la densità di probabilità
        """
        # Calcola i parametri per le due distribuzioni Beta
        beta1_alpha = self.alpha1 + self.tau
        beta1_beta = self.alpha2

        beta2_alpha = self.alpha1
        beta2_beta = self.alpha2 + self.tau

        # Calcola le densità di probabilità per le due distribuzioni Beta
        beta1_dist = th.distributions.Beta(beta1_alpha, beta1_beta)
        beta2_dist = th.distributions.Beta(beta2_alpha, beta2_beta)

        # Calcola il logaritmo della densità di probabilità per la distribuzione FlexibleBeta
        # Utilizziamo la formula della PDF fornita

        # Calcolo dei componenti della formula
        log_gamma_sum = th.lgamma(self.alpha1 + self.alpha2 + self.tau)
        log_gamma_alpha1 = th.lgamma(self.alpha1)
        log_gamma_alpha2 = th.lgamma(self.alpha2)

        # Parte comune della formula
        log_common = log_gamma_sum - log_gamma_alpha1 - log_gamma_alpha2
        log_common += (self.alpha1 - 1) * th.log(value + 1e-10)
        log_common += (self.alpha2 - 1) * th.log(1 - value + 1e-10)

        # Parte specifica per le due componenti
        log_term1 = th.log(self.p + 1e-10) + log_gamma_alpha1 - th.lgamma(self.alpha1 + self.tau) + self.tau * th.log(
            value + 1e-10)
        log_term2 = th.log(1 - self.p + 1e-10) + log_gamma_alpha2 - th.lgamma(
            self.alpha2 + self.tau) + self.tau * th.log(1 - value + 1e-10)

        # Combina i termini usando LogSumExp per stabilità numerica
        max_term = th.maximum(log_term1, log_term2)
        log_sum = max_term + th.log(th.exp(log_term1 - max_term) + th.exp(log_term2 - max_term))

        log_prob = log_common + log_sum

        return log_prob

    def pdf(self, value):
        """
        Calcola la densità di probabilità (PDF) per il valore dato.

        Args:
            value: Valore per cui calcolare la densità di probabilità
        """
        return th.exp(self.log_prob(value))

    def entropy(self):
        """
        Calcola l'entropia della distribuzione FlexibleBeta come approssimazione
        del valore atteso di -log(pdf), campionando punti tra 0 e 1.

        Utilizziamo una griglia uniforme di punti tra 0 e 1 per calcolare l'approssimazione.
        """
        # Numero di punti da campionare per l'approssimazione
        n_samples = 100

        # Crea una griglia uniforme di punti tra 0 e 1
        # Assicurandoci di avere le dimensioni corrette per il batch
        batch_size = self.alpha1.shape[0]
        action_dim = 1 if len(self.alpha1.shape) == 1 else self.alpha1.shape[1]

        # Crea una griglia di campioni
        samples = th.linspace(0.01, 0.99, n_samples).to(self.device)

        # Espandi la griglia per coprire tutte le dimensioni del batch e dell'azione
        samples = samples.unsqueeze(0).unsqueeze(-1)
        samples = samples.expand(batch_size, n_samples, action_dim)

        # Appiattisci per il calcolo del log_prob
        flat_samples = samples.reshape(-1, action_dim)

        # Ripeti i parametri della distribuzione per ogni campione
        alpha1_expanded = self.alpha1.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, action_dim)
        alpha2_expanded = self.alpha2.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, action_dim)
        tau_expanded = self.tau.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, action_dim)
        p_expanded = self.p.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, action_dim)

        # Crea una distribuzione temporanea per calcolare log_prob per tutti i campioni
        temp_dist = FlexibleBetaDistribution(alpha1_expanded, alpha2_expanded, tau_expanded, p_expanded)

        # Calcola log_prob per ogni campione
        log_probs = temp_dist.log_prob(flat_samples)

        # Riforma per ottenere log_probs per batch, sample
        log_probs = log_probs.reshape(batch_size, n_samples, action_dim)

        # La PDF normalizzata (assicurandoci che l'integrale sia 1)
        probs = th.exp(log_probs)
        normalizer = probs.sum(dim=1, keepdim=True) / n_samples
        normalized_probs = probs / (normalizer + 1e-10)

        # Calcola l'entropia come E[-log(p(x))]
        entropy_values = -normalized_probs * log_probs

        # Media sull'asse dei campioni e moltiplica per l'intervallo (1/n_samples)
        entropy = entropy_values.sum(dim=1) / n_samples

        return entropy

    def get_actions(self, deterministic=False):
        """
        Restituisce le azioni, in modo deterministico o stocastico.

        Args:
            deterministic: Se True, restituisce la moda della distribuzione invece di campionare
        """
        if deterministic:
            # Calcola la moda della distribuzione FlexibleBeta
            # campionando diversi punti e trovando quello con la più alta densità di probabilità

            # Numero di punti da campionare per trovare la moda
            n_samples = 1000

            # Crea una griglia uniforme di punti tra 0 e 1
            # Assicurandoci di avere le dimensioni corrette per il batch
            batch_size = self.alpha1.shape[0]
            action_dim = 1 if len(self.alpha1.shape) == 1 else self.alpha1.shape[1]

            # Crea una griglia di campioni
            samples = th.linspace(0.01, 0.99, n_samples).to(self.device)

            # Espandi la griglia per coprire tutte le dimensioni del batch e dell'azione
            samples = samples.unsqueeze(0).unsqueeze(-1)
            samples = samples.expand(batch_size, n_samples, action_dim)

            # Calcola la PDF per ogni campione
            pdf_values = []

            # Utilizziamo un approccio chunk-wise per evitare di esaurire la memoria
            # specialmente per batch grandi
            chunk_size = 100  # Regola in base alla disponibilità di memoria
            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                chunk_samples = samples[:, i:end_idx, :]

                # Appiattisci per il calcolo del log_prob
                flat_chunk = chunk_samples.reshape(-1, action_dim)

                # Ripeti i parametri della distribuzione per ogni campione nel chunk
                alpha1_expanded = self.alpha1.unsqueeze(1).expand(-1, end_idx - i, -1).reshape(-1, action_dim)
                alpha2_expanded = self.alpha2.unsqueeze(1).expand(-1, end_idx - i, -1).reshape(-1, action_dim)
                tau_expanded = self.tau.unsqueeze(1).expand(-1, end_idx - i, -1).reshape(-1, action_dim)
                p_expanded = self.p.unsqueeze(1).expand(-1, end_idx - i, -1).reshape(-1, action_dim)

                # Crea una distribuzione temporanea per calcolare PDF per i campioni del chunk
                temp_dist = FlexibleBetaDistribution(alpha1_expanded, alpha2_expanded, tau_expanded, p_expanded)

                # Calcola PDF per ogni campione nel chunk
                log_probs = temp_dist.log_prob(flat_chunk)
                chunk_pdf = th.exp(log_probs).reshape(batch_size, end_idx - i, action_dim)
                pdf_values.append(chunk_pdf)

            # Concatena tutti i valori PDF
            pdf_values = th.cat(pdf_values, dim=1)

            # Trova l'indice del valore massimo della PDF per ogni batch e dimensione
            max_pdf_indices = th.argmax(pdf_values, dim=1)

            # Estrai i valori corrispondenti dalla griglia di campioni
            modes = th.gather(samples, 1, max_pdf_indices.unsqueeze(1)).squeeze(1)

            # Riscala tra [-1, 1]
            modes = modes * 2.0 - 1.0

            # Applica il reshape se necessario
            if hasattr(self, 'action_space') and self.action_space is not None:
                modes = modes.reshape((-1, *self.action_space.shape))

            return modes
        else:
            # Campiona dalla distribuzione
            actions = self.sample()

            # Riscala tra [-1, 1]
            actions = actions * 2.0 - 1.0

            # Applica il reshape se necessario
            if hasattr(self, 'action_space') and self.action_space is not None:
                actions = actions.reshape((-1, *self.action_space.shape))

            return actions


class FlexibleBetaPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
    ):
        super(FlexibleBetaPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )

        # Parametri della distribuzione FlexibleBeta
        action_dim = self.action_space.shape[0]

        # Rete per generare i parametri alpha1, alpha2, tau e p
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim * 4)

        # Assicuriamoci che la rete sia sul dispositivo corretto
        self.action_net.to(self.device)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        """
        Restituisce la distribuzione delle azioni dato l'output latente della policy.

        Args:
            latent_pi: Output latente della policy
            latent_sde: Parametro non utilizzato per questa policy
        """
        # Assicuriamoci che latent_pi sia sul dispositivo corretto
        latent_pi = latent_pi.to(self.device)

        # Genera i parametri alpha1, alpha2, tau e p
        raw_params = self.action_net(latent_pi)

        action_dim = self.action_space.shape[0]



        # Estrai i parametri
        alpha1_raw = raw_params[..., :action_dim]
        alpha2_raw = raw_params[..., action_dim:action_dim * 2]
        tau_raw = raw_params[..., action_dim * 2:action_dim * 3]
        p_raw = raw_params[..., action_dim * 3:]


        # Trasforma i parametri per garantire che siano nel range valido
        alpha1 = th.log(1 + th.exp(alpha1_raw)) + 1.0  # Assicura che alpha1 > 1
        alpha2 = th.log(1 + th.exp(alpha2_raw)) + 1.0  # Assicura che alpha2 > 1
        tau = th.log(1 + th.exp(tau_raw))  # Assicura che tau > 0
        p = th.sigmoid(p_raw)  # Assicura che p sia tra 0 e 1

        # Crea la distribuzione FlexibleBeta
        return FlexibleBetaDistribution(alpha1, alpha2, tau, p, action_space=self.action_space)

    def forward(self, obs, deterministic=False):
        """
        Forward pass per la policy.

        Args:
            obs: Osservazioni
            deterministic: Se True, usa la moda invece di campionare
        """


        # Assicuriamoci che le osservazioni siano sul dispositivo corretto
        obs = obs.to(self.device)

        # Estrai le feature dalle osservazioni
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)



        # Calcola il valore
        values = self.value_net(latent_vf)



        # Ottieni la distribuzione delle azioni
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Ottieni le azioni
        actions = distribution.get_actions(deterministic)

        # Calcola il log delle probabilità
        log_prob = distribution.log_prob(actions / 2.0 + 0.5).sum(dim=1)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Valuta le azioni date le osservazioni.

        Args:
            obs: Osservazioni
            actions: Azioni da valutare
        """
        # Assicuriamoci che le osservazioni siano sul dispositivo corretto
        obs = obs.to(self.device)

        # Riscala le azioni da [-1, 1] a [0, 1] per la distribuzione FlexibleBeta
        rescaled_actions = (actions + 1.0) / 2.0

        # Estrai le feature dalle osservazioni
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Calcola il valore
        values = self.value_net(latent_vf)

        # Ottieni la distribuzione delle azioni
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Calcola il log delle probabilità e l'entropia
        log_prob = distribution.log_prob(rescaled_actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return values, log_prob, entropy