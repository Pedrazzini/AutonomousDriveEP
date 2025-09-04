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

print("grad-cam version:", importlib.metadata.version("grad-cam"))

import inspect

print("GradCAM imported from:", inspect.getfile(GradCAM))

from pytorch_grad_cam.utils.image import show_cam_on_image
import math

# configuration loading
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# image shape RGB
image_shape = (50, 50, 3)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Grad_CAM will run on: {device}")

# environment creation
env = DummyVecEnv([lambda: gym.make(
    "scripts:airsim-env-v0",
    ip_address="127.0.0.1",
    image_shape=image_shape,
    env_config=env_config["TrainEnv"],
    input_mode="single_rgb"
)])

env = VecTransposeImage(env)

# load pretrained model
model_path = "saved_policy/best_model.zip"
model = PPO.load(model_path, env=env)

# debug to analyze the CNN structure
for name, module in model.policy.named_modules():
    if isinstance(module, nn.Linear):
        print(f"linear Layer '{name}': input={module.in_features}, output={module.out_features}")
    elif isinstance(module, nn.Conv2d):
        print(
            f"conv Layer '{name}': input={module.in_channels}, output={module.out_channels}, kernel={module.kernel_size}")


# class to extract only the CNN part from PPO model
class CNNOnly(nn.Module):
    def __init__(self, policy):
        super(CNNOnly, self).__init__()
        # extract only CNN
        self.features = policy.features_extractor.cnn

    def forward(self, x):
        # pass through CNN and return feature maps
        return self.features(x)
# get AirSim image
def get_airsim_image(client):
    rgb_image_request = airsim.ImageRequest(
        0, airsim.ImageType.Scene, False, False)
    responses = client.simGetImages([rgb_image_request])


    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))

    return img2d


# biary classification semplification: only right or left
class BinaryClassificationTarget:
    def __init__(self, category_index):
        self.category_index = category_index

    def __call__(self, model_output):
        # 0 for the left, 1 for the righ (if the output has two dimensions)
        if model_output.shape[1] > 1:
            return model_output[:, self.category_index]
        else:
            # if the output is a single value
            # for analyzing label 0 (left), return -1 * output
            # for analyzing label 1 (right), return output
            return model_output if self.category_index == 1 else -model_output


# set position and analyze image with Grad-Cam
def analyze_with_gradcam(x_pos, y_pos, yaw_degrees, methods=None):
    if methods is None:
        methods = [GradCAM]

    # AirSim connection
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # reset simulation
    client.reset()

    # conversion of the degrees
    yaw_radians = math.radians(yaw_degrees)

    # set car position
    pose = airsim.Pose(airsim.Vector3r(x_pos, y_pos, 0), airsim.to_quaternion(0, 0, yaw_radians))
    client.simSetVehiclePose(pose, ignore_collision=True)

    # wait a bit to let the car reset
    time.sleep(1)

    # get image from camera
    image = get_airsim_image(client)

    # resize
    img_resized = cv2.resize(image, (50, 50))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # normalization
    img_display = img_rgb.astype(np.float32) / 255.0

    # prepare image for the model
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    # wrapper creation
    cnn_model = CNNOnly(model.policy)

    # for grad-Cam only the last convolutional layer is used
    last_conv_layer = None
    for layer in cnn_model.features:
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer

    target_layers = [last_conv_layer]

    print(f"Layer target per GradCAM: {last_conv_layer}")

    # prepare the image for Grad-Cam
    input_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)

    print(f"Debug - input_tensor shape: {input_tensor.shape}")
    print(f"Debug - target_layers: {target_layers}")

    # prediction of the action with the model
    with torch.no_grad():
        action, _ = model.policy.predict(np.expand_dims(img_chw, axis=0), deterministic=True)
    print(f"Shape of model output: {action.shape}")
    steering_value = float(action[0])

    # informations
    print(f"Position: ({x_pos}, {y_pos}), Rotation: {yaw_degrees}°")
    print(f"Value of steering: {steering_value:.4f} (range: -1 a 1)")
    print(f"  - Interpretation: {'right steering' if steering_value > 0 else 'left steering'}")

    # plot results
    n_cols = len(methods) + 1
    plt.figure(figsize=(n_cols * 5, 5))

    # original image
    plt.subplot(1, n_cols, 1)
    plt.imshow(img_display)
    plt.title(f'original image\n({x_pos}, {y_pos}, {yaw_degrees}°)\nSteering: {steering_value:.4f}')
    plt.axis('off')

    # CAM methods
    for i, method_class in enumerate(methods):
        method_name = method_class.__name__

        # do the CAM algorithm
        try:

            cam = method_class(model=cnn_model, target_layers=target_layers)

            # use all feature maps
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]  # first batch image


            # plot results
            plt.subplot(1, n_cols, i + 2)

            visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
            plt.imshow(visualization)

            if steering_value >= 0:
                plt.title(f'{method_name}\npixels influencing the\nRIGHT steering')
            else:
                plt.title(f'{method_name}\npixels influencing the\nLEFT steering')

            plt.axis('off')
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            plt.subplot(1, n_cols, i + 2)
            plt.text(0.5, 0.5, f"Error with {method_name}:\n{str(e)}",
                     ha='center', va='center', wrap=True)
            plt.axis('off')

    plt.tight_layout()

    # save all
    grad_cam_save_path = f"grad_cam_analysis_{x_pos}_{y_pos}_{yaw_degrees}.png"
    plt.savefig(grad_cam_save_path)
    print(f"GradCAM saved in: {grad_cam_save_path}")

    plt.show()

    # switch off the control
    client.enableApiControl(False)

    return steering_value


# analyze multiple positions
def analyze_multiple_positions():

    methods = [GradCAM]  # you can also add something else

    positions = [
        # List of positions to analyze: (x, y, yaw)
        (0, 0, 0),
        (20, 0, 0),
        (-120,0,0),
    ]

    results = []
    for i, (x, y, yaw) in enumerate(positions):
        print(f"\n===== analyzing position {i + 1}/{len(positions)} =====")
        try:
            steering = analyze_with_gradcam(x, y, yaw, methods)
            results.append((x, y, yaw, steering))
        except Exception as e:
            print(f"Error with position {i + 1}: {e}")
        time.sleep(2)  # wait a bit

    # plots results
    if results:
        print("\n===== results =====")
        for i, (x, y, yaw, steering) in enumerate(results):
            print(f"Position {i + 1}: ({x}, {y}, {yaw}°) -> Steering: {steering:.4f}")


# program execution
if __name__ == "__main__":
    try:
        analyze_multiple_positions()
    except Exception as e:
        print(f"general error: {e}")