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
from custom_policy import BetaPolicy

# load train environment configs
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# determine input image shape
image_shape = (50, 50, 3)

# create a DummyVecEnv per il training
train_env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0",
        ip_address="127.0.0.1",
        image_shape=image_shape,
        env_config=env_config["TrainEnv"],
        input_mode=config["train_mode"]
    )
)])

# wrap env as VecTransposeImage (Channel last to channel first)
train_env = VecTransposeImage(train_env)

policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=th.nn.Tanh,
)

model = PPO(
    BetaPolicy,
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

# --- VERIFY THE POLICY NETWORK STRUCTURE ---
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

# evaluation environment
eval_env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:test-env-v0",
        ip_address="127.0.0.1",
        image_shape=image_shape,
        env_config=env_config["EvalEnv"],
        input_mode=config["train_mode"],
        test_mode=True
    )
)])
eval_env = VecTransposeImage(eval_env)

# evaluation callback con l'ambiente di valutazione
callbacks = []
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=None,
    n_eval_episodes=4,
    best_model_save_path="saved_policy",
    log_path=".",
    eval_freq=256,
    deterministic=True,  # test it
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

log_name = "ppo_run_" + str(time.time())

model.learn(
    total_timesteps=35000,
    tb_log_name=log_name,
    **kwargs
)

