import gym
import time
import yaml
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from scripts.network import NatureCNN
from custom_gaussian_policy import GaussianPolicy

# Load train environment configs
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# Load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Determine input image shape
image_shape = (50, 50, 3)

# Create a DummyVecEnv per il training
train_env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0",
        ip_address="127.0.0.1",
        image_shape=image_shape,
        env_config=env_config["TrainEnv"],
        input_mode=config["train_mode"]
    )
)])

# Wrap env as VecTransposeImage (Channel last to channel first)
train_env = VecTransposeImage(train_env)

policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    # Stessa struttura della rete come nella BetaPolicy
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=th.nn.Tanh,
)

model = PPO(
    GaussianPolicy,  # Usa la custom GaussianPolicy
    train_env,
    batch_size=128,
    clip_range=0.10,
    max_grad_norm=0.5,
    verbose=1,
    seed=1,
    device="cuda",
    tensorboard_log="./tb_logs/",
    policy_kwargs=policy_kwargs,
)

# --- VERIFICA STRUTTURA DELLA POLICY NETWORK ---
print("\n STRUTTURA COMPLETA DELLA POLICY:\n")
print(model.policy)

print("\n Feature extractor (NatureCNN):\n")
print(model.policy.features_extractor)

print("\n MLP extractor:\n")
print(model.policy.mlp_extractor)

print("\n Action net:\n")
print(model.policy.action_net)

print("\n Value net:\n")
print(model.policy.value_net)

# Ambiente per la valutazione (usando TestEnv)
eval_env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:test-env-v0",  # Usa l'ambiente di test registrato
        ip_address="127.0.0.1",
        image_shape=image_shape,
        env_config=env_config["EvalEnv"],  # Configurazione specifica per la valutazione
        input_mode=config["train_mode"],
        test_mode=True  # Parametro aggiuntivo per TestEnv
    )
)])
eval_env = VecTransposeImage(eval_env)

# Evaluation callback con l'ambiente di valutazione
callbacks = []
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=None,
    n_eval_episodes=10,
    best_model_save_path="saved_policy",
    log_path=".",
    eval_freq=1024,
    deterministic=True,  # Usa azioni deterministiche durante la valutazione
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

log_name = "ppo_gaussian_run_" + str(time.time())

model.learn(
    total_timesteps=35000,
    tb_log_name=log_name,
    **kwargs
)