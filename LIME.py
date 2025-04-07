import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image
from stable_baselines3 import PPO
import gym
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from scripts.network import NatureCNN
import airsim
import time
import math

# load the configuration
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# load the choice of image format
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# set image shape for RGB
image_shape = (50, 50, 3)

# create environment
env = DummyVecEnv([lambda: gym.make(
    "scripts:airsim-env-v0",
    ip_address="127.0.0.1",
    image_shape=image_shape,
    env_config=env_config["TrainEnv"],
    input_mode="single_rgb"
)])

# transform the environment
env = VecTransposeImage(env)

# load the pre-trained model
model_path = "saved_policy/best_model.zip"
model = PPO.load(model_path, env=env)


# function utilized by lime for predictions
def model_predict_fn(images):

    processed_images = []
    for img in images:
        # reshaping image (50x50)
        img_resized = cv2.resize(img, (50, 50))
        # what PyTorch is expecting
        # what VecTransposeImage does
        img_chw = np.transpose(img_resized, (2, 0, 1))
        processed_images.append(img_chw)

    # convert into array numpy
    batch = np.array(processed_images)

    # get predictions from PPO model

    with np.errstate(all='ignore'):
        actions, _ = model.policy.predict(batch, deterministic=False)

    # convert actions so that LIME can read them
    # map range [-1, 1] to [0, 1] as a probability
    probs = np.zeros((len(actions), 2))  # let's use two classes: right and left
    norm_actions = (actions + 1) / 2  # from [-1, 1] to [0, 1]
    probs[:, 0] = 1 - norm_actions.flatten()  # prob of going left
    probs[:, 1] = norm_actions.flatten()  # prob of going right

    return probs


# get images from AirSim
def get_airsim_image(client):
    # get image from the frontal camera
    rgb_image_request = airsim.ImageRequest(
        0, airsim.ImageType.Scene, False, False)
    responses = client.simGetImages([rgb_image_request])

    # convert to array numpy
    img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))

    return img2d


# Funzione per posizionare l'auto e analizzare l'immagine
def analyze_at_position(x_pos, y_pos, yaw_degrees, save_path=None):
    # Connettiti ad AirSim
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # Reset la simulazione
    client.reset()

    # Converti gradi in radianti per la rotazione
    yaw_radians = math.radians(yaw_degrees)

    # Crea una pose per posizionare l'auto
    pose = airsim.Pose(airsim.Vector3r(x_pos, y_pos, 0), airsim.to_quaternion(0, 0, yaw_radians))

    # Posiziona l'auto
    client.simSetVehiclePose(pose, ignore_collision=True)

    # Attendi un momento per far stabilizzare la simulazione
    time.sleep(1)

    # Acquisizione dell'immagine dalla telecamera
    image = get_airsim_image(client)

    # Salva l'immagine originale se richiesto
    if save_path:
        # Non serve convertire da RGB a BGR perché l'immagine è già in BGR
        cv2.imwrite(save_path, image)
        print(f"Immagine salvata in: {save_path}")

    # Crea una copia dell'immagine per la visualizzazione
    display_image = cv2.resize(image, (200, 150))

    # Predici l'azione con il modello addestrato
    # Prepara l'immagine nel formato corretto per il modello
    img_for_model = cv2.resize(image, (50, 50))

    # IMPORTANTE: NON normalizziamo qui perché lo farà implicitamente VecTransposeImage
    # IMPORTANTE: Trasponiamo qui perché è ciò che fa VecTransposeImage
    img_chw = np.transpose(img_for_model, (2, 0, 1))
    batch = np.array([img_chw])

    # Ottieni l'azione predetta dal modello
    action, _ = model.policy.predict(batch, deterministic=True)
    steering_value = float(action[0])

    print(f"Posizione: ({x_pos}, {y_pos}), Rotazione: {yaw_degrees}°")
    print(f"Valore di sterzata predetto: {steering_value:.4f} (range: -1 a 1)")
    print(f"  - Interpretazione: {'Sterza a destra' if steering_value > 0 else 'Sterza a sinistra'}")

    # Inizializza l'explainer LIME
    explainer = lime_image.LimeImageExplainer()

    # Genera spiegazioni per l'immagine
    # Per la visualizzazione, convertiamo in RGB solo per la visualizzazione con matplotlib
    display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    explanation = explainer.explain_instance(
        display_image,  # Passiamo l'immagine BGR originale
        model_predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=300,
        random_seed=42
    )

    # Crea una figura con più sottografici
    plt.figure(figsize=(15, 5))

    # 1. Immagine originale - convertiamo in RGB solo per la visualizzazione con matplotlib
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Immagine Originale\n({x_pos}, {y_pos}, {yaw_degrees}°)')
    plt.axis('off')

    # 2. Spiegazione LIME per sterzare a destra (label=1)
    temp_right, mask_right = explanation.get_image_and_mask(
        label=1,  # Classe "destra"
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    plt.subplot(1, 3, 2)
    # Convertiamo temp_right in RGB per la visualizzazione con matplotlib
    temp_right_rgb = cv2.cvtColor(temp_right.astype(np.uint8), cv2.COLOR_BGR2RGB)
    temp_right_rgb_norm = temp_right_rgb.astype(float) / 255.0
    plt.imshow(mark_boundaries(temp_right_rgb_norm / 2 + 0.5, mask_right))
    plt.title(f'Aree che favoriscono\nla sterzata a DESTRA')
    plt.axis('off')

    # 3. Spiegazione LIME per sterzare a sinistra (label=0)
    temp_left, mask_left = explanation.get_image_and_mask(
        label=0,  # Classe "sinistra"
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    plt.subplot(1, 3, 3)
    # Convertiamo temp_left in RGB per la visualizzazione con matplotlib
    temp_left_rgb = cv2.cvtColor(temp_left.astype(np.uint8), cv2.COLOR_BGR2RGB)
    temp_left_rgb_norm = temp_left_rgb.astype(float) / 255.0
    plt.imshow(mark_boundaries(temp_left_rgb_norm / 2 + 0.5, mask_left))
    plt.title(f'Aree che favoriscono\nla sterzata a SINISTRA')
    plt.axis('off')

    plt.tight_layout()

    # Salva l'analisi LIME
    lime_save_path = f"lime_analysis_{x_pos}_{y_pos}_{yaw_degrees}.png"
    plt.savefig(lime_save_path)
    print(f"Analisi LIME salvata in: {lime_save_path}")

    plt.show()

    # Disabilita il controllo API e chiudi la connessione
    client.enableApiControl(False)

    return explanation, steering_value


# Funzione per analizzare diverse posizioni
def analyze_multiple_positions():
    positions = [
        # Lista di posizioni da analizzare: (x, y, yaw)
        (119, 3, 20),  # Posizione centrale, direzione in avanti
        (120, 0, -20),  # Posizione centrale, orientata a 45 gradi

    ]

    results = []
    for i, (x, y, yaw) in enumerate(positions):
        print(f"\n===== Analisi posizione {i + 1}/{len(positions)} =====")
        _, steering = analyze_at_position(x, y, yaw)
        results.append((x, y, yaw, steering))
        time.sleep(2)  # Attendi tra le posizioni

    # Stampa un riepilogo dei risultati
    print("\n===== Riepilogo dei risultati =====")
    for i, (x, y, yaw, steering) in enumerate(results):
        print(f"Posizione {i + 1}: ({x}, {y}, {yaw}°) -> Sterzata: {steering:.4f}")


# Se eseguito come script principale
if __name__ == "__main__":
    analyze_multiple_positions()