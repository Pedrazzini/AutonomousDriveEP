# REMEMBER TO CHANGE TEST TYPE IN config.yml
import os
import gym
import yaml
import time
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from scripts.network import NatureCNN

# load train environment configs
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# model name
model_name = "best_model.zip"

# determine input image shape
image_shape = (50, 50, 3)

# create a DummyVecEnv
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:test-env-v0",
        ip_address="127.0.0.1",
        image_shape=image_shape,
        # train and test envs shares same config for the test
        env_config=env_config["EvalEnv"],
        input_mode=config["test_mode"],
        test_mode=config["test_type"]
    )
)])

# wrap env as VecTransposeImage (Channel last to channel first)
env = VecTransposeImage(env)

policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=th.nn.Tanh,
)

# load an existing model
model = PPO.load(
    env=env,
    path=os.path.join("saved_policy", model_name),
    policy_kwargs=policy_kwargs
)

# run the trained policy
obs = env.reset()
start_time = time.time()
collision_detected = False

for i in range(2300):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, info = env.step(action)

    # check if the episode ended (collision)
    if dones[0]:
        collision_detected = True
        elapsed_time = time.time() - start_time
        print("-----------------------------------")
        print(f"> Collision registered after {elapsed_time:.2f} seconds")
        print("-----------------------------------\n")
        break

# if we completed all the 2300 steps without collisions
if not collision_detected:
    elapsed_time = time.time() - start_time
    print("-----------------------------------")
    print(f"> You win! The car traveled {elapsed_time:.2f} seconds long without collisions")
    print("-----------------------------------\n")