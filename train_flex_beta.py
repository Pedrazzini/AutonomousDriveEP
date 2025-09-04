import gym
import time
import yaml
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback # Importa CheckpointCallback
from scripts.network import NatureCNN
from flex_beta_policy3 import FlexibleBetaPolicy

# load train environment configs
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# determine input image shape
image_shape = (50, 50, 3)

# create a DummyVecEnv for training
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
    activation_fn=th.nn.Tanh, # to compare it with other models
)

# path to TensorBoard log
tensorboard_log_path = "./tb_logs/"
log_name = "ppo_run_" + str(time.time())

model = PPO(
    FlexibleBetaPolicy,
    train_env,  # use training environment
    batch_size=128,
    clip_range=0.10,
    max_grad_norm=0.5,
    verbose=1,
    seed=1,
    device="cuda",
    tensorboard_log=tensorboard_log_path, # use defined path
    policy_kwargs=policy_kwargs,
)

# --- CHECK POINTS ---
print("\n POLICY STRUCTURE:\n")
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
        env_config=env_config["EvalEnv"],  # special configuration
        input_mode=config["train_mode"],
        test_mode=True  # parameter for test env
    )
)])
eval_env = VecTransposeImage(eval_env)

# evaluation callback
callbacks = []
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="saved_policy_best", # where I want to save the best model
    log_path=".",
    eval_freq=1024,
    deterministic=True,
)
callbacks.append(eval_callback)

# **ADD CHECKPOINT CALLBACK TO SAVE PERIODICALLY**
# save each 2048 steps
checkpoint_callback = CheckpointCallback(
    save_freq=2048,
    save_path="./checkpoints/", # file
    name_prefix="ppo_flexible_beta_model",
)
callbacks.append(checkpoint_callback)


kwargs = {}
kwargs["callback"] = callbacks

model.learn(
    total_timesteps=35000,
    tb_log_name=log_name,
    reset_num_timesteps=False, # IMPORTANT: Do not reset the timesteps if you want to continue a training
    **kwargs
)

# save the final model
model.save("final_ppo_flexible_beta_model")
train_env.close()
eval_env.close()

print("Training completed!")