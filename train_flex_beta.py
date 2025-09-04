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

# Load train environment configs
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# Load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Determine input image shape
image_shape = (100, 100, 3)

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
    net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=th.nn.Tanh, # per renderlo uguale alla struttura della rete nella Gaussian Policy
)

# Definizione del percorso per il log di TensorBoard
tensorboard_log_path = "./tb_logs/"
# Definizione del nome del run (puoi mantenerlo come prima)
log_name = "ppo_run_" + str(time.time())

model = PPO(
    FlexibleBetaPolicy,
    train_env,  # Usa l'ambiente di training
    batch_size=128,
    clip_range=0.10,
    max_grad_norm=0.5,
    verbose=1,
    seed=1,
    device="cuda",
    tensorboard_log=tensorboard_log_path, # Usa il percorso definito
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
    eval_env,  # Usa l'ambiente di valutazione
    callback_on_new_best=None, # Puoi impostarlo per salvare il miglior modello se vuoi
    n_eval_episodes=5,
    best_model_save_path="saved_policy_best", # Percorso per salvare il modello migliore
    log_path=".",
    eval_freq=1024,
    deterministic=True,
)
callbacks.append(eval_callback)

# **AGGIUNGI IL CHECKPOINT CALLBACK PER SALVARE PERIODICAMENTE**
# Salva il modello ogni 10000 timesteps nel percorso specificato
checkpoint_callback = CheckpointCallback(
    save_freq=2048, # Salva ogni 10000 timesteps
    save_path="./checkpoints/", # Cartella dove salvare i checkpoint
    name_prefix="ppo_flexible_beta_model", # Prefisso per il nome dei file
)
callbacks.append(checkpoint_callback)


kwargs = {}
kwargs["callback"] = callbacks

model.learn(
    total_timesteps=35000,
    tb_log_name=log_name,
    reset_num_timesteps=False, # IMPORTANTE: Non resettare il conteggio dei timesteps se stai riprendendo
    **kwargs
)

# Salva il modello finale alla fine del training (utile anche se non interrotto)
model.save("final_ppo_flexible_beta_model")
train_env.close()
eval_env.close()

print("Training completato e modello finale salvato!")