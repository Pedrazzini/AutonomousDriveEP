import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import airsim
import time
import yaml
from stable_baselines3 import PPO
import gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from scripts.network import NatureCNN
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus
import importlib.metadata

print("🧪 Versione grad-cam in uso:", importlib.metadata.version("grad-cam"))

import inspect

print("📦 GradCAM viene importato da:", inspect.getfile(GradCAM))

from pytorch_grad_cam.utils.image import show_cam_on_image
import math

# Caricamento delle configurazioni
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Impostazione forma dell'immagine per RGB
image_shape = (50, 50, 3)

# Device globale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 CAM verrà eseguito su: {device}")

# Creazione ambiente
env = DummyVecEnv([lambda: gym.make(
    "scripts:airsim-env-v0",
    ip_address="127.0.0.1",
    image_shape=image_shape,
    env_config=env_config["TrainEnv"],
    input_mode="single_rgb"
)])

# Trasformazione ambiente
env = VecTransposeImage(env)

# Caricamento del modello pre-addestrato
model_path = "saved_policy/best_model.zip"
model = PPO.load(model_path, env=env)

# Analizzo la struttura del modello per il debug
for name, module in model.policy.named_modules():
    if isinstance(module, nn.Linear):
        print(f"Layer lineare '{name}': input={module.in_features}, output={module.out_features}")
    elif isinstance(module, nn.Conv2d):
        print(
            f"Layer conv '{name}': input={module.in_channels}, output={module.out_channels}, kernel={module.kernel_size}")


# Classe per estrarre solo la parte CNN dal modello PPO
# Questa classe è semplificata per funzionare solo con GradCAM ed evitare errori di dimensione
class CNNOnly(nn.Module):
    def __init__(self, policy):
        super(CNNOnly, self).__init__()
        # Estrai solamente il modulo CNN
        self.features = policy.features_extractor.cnn

    def forward(self, x):
        # Passa attraverso la CNN e restituisci direttamente le feature maps
        return self.features(x)
# Funzione per ottenere un'immagine da AirSim
def get_airsim_image(client):
    rgb_image_request = airsim.ImageRequest(
        0, airsim.ImageType.Scene, False, False)
    responses = client.simGetImages([rgb_image_request])

    # Utilizza frombuffer invece di fromstring (deprecato)
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))

    return img2d


# Target personalizzato per GradCAM per la classificazione binaria
class BinaryClassificationTarget:
    def __init__(self, category_index):
        self.category_index = category_index

    def __call__(self, model_output):
        # Per un canale specifico (0 per sinistra, 1 per destra)
        if model_output.shape[1] > 1:
            # Se l'output ha più canali
            return model_output[:, self.category_index]
        else:
            # Se l'output è un singolo valore, lo trattiamo in modo speciale
            # Per il canale 0 (sinistra), ritorniamo -1 * output
            # Per il canale 1 (destra), ritorniamo l'output
            return model_output if self.category_index == 1 else -model_output


# Funzione per posizionare l'auto e analizzare l'immagine con GradCAM
def analyze_with_gradcam(x_pos, y_pos, yaw_degrees, methods=None):
    if methods is None:
        methods = [GradCAM]

    # Connessione ad AirSim
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # Reset della simulazione
    client.reset()

    # Conversione gradi in radianti per la rotazione
    yaw_radians = math.radians(yaw_degrees)

    # Creazione della pose per posizionare l'auto
    pose = airsim.Pose(airsim.Vector3r(x_pos, y_pos, 0), airsim.to_quaternion(0, 0, yaw_radians))

    # Posizionamento dell'auto
    client.simSetVehiclePose(pose, ignore_collision=True)

    # Attesa per stabilizzare la simulazione
    time.sleep(1)

    # Acquisizione dell'immagine dalla telecamera
    image = get_airsim_image(client)

    # Ridimensionamento dell'immagine per il modello
    img_resized = cv2.resize(image, (50, 50))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalizzazione immagine per la visualizzazione
    img_display = img_rgb.astype(np.float32) / 255.0

    # Preparazione dell'immagine per il modello
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    # Creazione del wrapper per il modello
    cnn_model = CNNOnly(model.policy)

    # Strato target per GradCAM (ultimo strato convoluzionale)
    # Usiamo l'ultimo layer convoluzionale prima di Flatten
    last_conv_layer = None
    for layer in cnn_model.features:
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer

    target_layers = [last_conv_layer]

    print(f"Layer target per GradCAM: {last_conv_layer}")

    # Preparazione immagine per CAM
    input_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)

    print(f"Debug - input_tensor shape: {input_tensor.shape}")
    print(f"Debug - target_layers: {target_layers}")

    # Esecuzione della predizione con il modello completo
    with torch.no_grad():
        action, _ = model.policy.predict(np.expand_dims(img_chw, axis=0), deterministic=True)
    print(f"Shape of model output: {action.shape}")  # Stampa la forma dell'output del modello
    steering_value = float(action[0])

    # Riepilogo informazioni
    print(f"Posizione: ({x_pos}, {y_pos}), Rotazione: {yaw_degrees}°")
    print(f"Valore di sterzata predetto: {steering_value:.4f} (range: -1 a 1)")
    print(f"  - Interpretazione: {'Sterza a destra' if steering_value > 0 else 'Sterza a sinistra'}")

    # Creazione della figura per visualizzare i risultati
    n_cols = len(methods) + 1
    plt.figure(figsize=(n_cols * 5, 5))

    # Visualizzazione immagine originale
    plt.subplot(1, n_cols, 1)
    plt.imshow(img_display)
    plt.title(f'Immagine Originale\n({x_pos}, {y_pos}, {yaw_degrees}°)\nSterzata: {steering_value:.4f}')
    plt.axis('off')

    # Applicazione dei vari metodi CAM
    for i, method_class in enumerate(methods):
        method_name = method_class.__name__

        # Creiamo la visualizzazione per il target
        try:

            cam = method_class(model=cnn_model, target_layers=target_layers)

            # MOLTO IMPORTANTE: non usare alcun target per GradCAM v1.5.5
            # Questo utilizzerà semplicemente tutte le feature maps
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]  # Prima immagine del batch


            # Visualizzazione risultati
            plt.subplot(1, n_cols, i + 2)

            visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
            plt.imshow(visualization)

            if steering_value >= 0:
                plt.title(f'{method_name}\nAree che influenzano la\nsterzata a DESTRA')
            else:
                plt.title(f'{method_name}\nAree che influenzano la\nsterzata a SINISTRA')

            plt.axis('off')
        except Exception as e:
            print(f"Errore con {method_name}: {e}")
            plt.subplot(1, n_cols, i + 2)
            plt.text(0.5, 0.5, f"Errore con {method_name}:\n{str(e)}",
                     ha='center', va='center', wrap=True)
            plt.axis('off')

    plt.tight_layout()

    # Salviamo l'analisi
    grad_cam_save_path = f"grad_cam_analysis_{x_pos}_{y_pos}_{yaw_degrees}.png"
    plt.savefig(grad_cam_save_path)
    print(f"Analisi GradCAM salvata in: {grad_cam_save_path}")

    plt.show()

    # Disabilitazione del controllo API e chiusura della connessione
    client.enableApiControl(False)

    return steering_value


# Funzione per analizzare multiple posizioni
def analyze_multiple_positions():
    # Definizione dei metodi CAM da utilizzare
    # Utilizziamo solo GradCAM per iniziare, poiché è il più stabile
    methods = [GradCAM]  # Puoi aggiungere altri metodi dopo che GradCAM funziona

    positions = [
        # Lista di posizioni da analizzare: (x, y, yaw)
        (119, 3, 20),  # Posizione centrale, direzione in avanti
        (120, 0, -20),  # Posizione centrale, orientata a -20 gradi
    ]

    results = []
    for i, (x, y, yaw) in enumerate(positions):
        print(f"\n===== Analisi posizione {i + 1}/{len(positions)} =====")
        try:
            steering = analyze_with_gradcam(x, y, yaw, methods)
            results.append((x, y, yaw, steering))
        except Exception as e:
            print(f"Errore durante l'analisi della posizione {i + 1}: {e}")
        time.sleep(2)  # Attesa tra le posizioni

    # Stampa un riepilogo dei risultati
    if results:
        print("\n===== Riepilogo dei risultati =====")
        for i, (x, y, yaw, steering) in enumerate(results):
            print(f"Posizione {i + 1}: ({x}, {y}, {yaw}°) -> Sterzata: {steering:.4f}")


# Esecuzione del programma
if __name__ == "__main__":
    try:
        analyze_multiple_positions()
    except Exception as e:
        print(f"Errore generale: {e}")