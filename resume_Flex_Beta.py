import gym
import time
import yaml
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from scripts.network import NatureCNN
from flex_beta_policy3 import FlexibleBetaPolicy

# --- CONFIGURAZIONE ---
ENV_CONFIG_PATH = 'scripts/env_config.yml'
CONFIG_PATH = 'config.yml'
CHECKPOINT_DIR = "./checkpoints/"
CHECKPOINT_PREFIX = "ppo_flexible_beta_model"
PATH_TO_RESUME_MODEL = "./checkpoints/ppo_flexible_beta_model_2048_steps.zip" # Controlla che questo sia il file corretto!
TOTAL_TIMESTEPS_TARGET = 35000

# --- CARICAMENTO CONFIGURAZIONI ---
with open(ENV_CONFIG_PATH, 'r') as f:
    env_config = yaml.safe_load(f)
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

image_shape = (100, 100, 3)

# --- CREAZIONE AMBIENTE DI TRAINING ---
train_env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0",
        ip_address="127.0.0.1",
        image_shape=image_shape,
        env_config=env_config["TrainEnv"],
        input_mode=config["train_mode"]
    )
)])
train_env = VecTransposeImage(train_env)

train_env.reset() # <--- IMPORTANTE: Resetta l'ambiente prima di avviare il training


# --- CARICAMENTO DEL MODELLO ---
print(f"Tentativo di caricare il modello da: {PATH_TO_RESUME_MODEL}")
try:
    model = PPO.load(
        PATH_TO_RESUME_MODEL,
        env=train_env,
        device="cuda",
        custom_objects={
            "policy_class": FlexibleBetaPolicy,
            "features_extractor_class": NatureCNN
        },
    )
    print(f"Modello caricato con successo. Timesteps già percorsi: {model.num_timesteps}")
except FileNotFoundError:
    print(f"ERRORE: File '{PATH_TO_RESUME_MODEL}' non trovato.")
    print("Assicurati che il percorso e il nome del file siano corretti.")
    exit()


# --- AMBIENTE DI VALUTAZIONE ---
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
eval_env.reset() # Anche l'ambiente di valutazione dovrebbe essere resettato

# --- CALLBACKS ---
callbacks = []
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="saved_policy_best",
    log_path=".",
    eval_freq=1024,
    deterministic=True,
)
callbacks.append(eval_callback)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=CHECKPOINT_DIR,
    name_prefix=CHECKPOINT_PREFIX,
)
callbacks.append(checkpoint_callback)

kwargs = {}
kwargs["callback"] = callbacks

# --- CONTINUA IL TRAINING ---
remaining_timesteps = TOTAL_TIMESTEPS_TARGET - model.num_timesteps
if remaining_timesteps <= 0:
    print(f"Il modello ha già raggiunto o superato il numero totale di timesteps ({TOTAL_TIMESTEPS_TARGET}). Training non necessario.")
    remaining_timesteps = 0
else:
    print(f"Continuo il training per ulteriori {remaining_timesteps} timesteps.")

log_name_resume = "ppo_resume_run_" + str(time.time())

model.learn(
    total_timesteps=remaining_timesteps,
    tb_log_name=log_name_resume,
    reset_num_timesteps=False,
    **kwargs
)

model.save("final_ppo_flexible_beta_model_resumed")
train_env.close()
eval_env.close()

print("Training ripreso e completato con successo!")