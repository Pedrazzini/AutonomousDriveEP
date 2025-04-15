from . import airsim
import os
import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class AirSimCarEnv(gym.Env):
    animals_moved = False
    def __init__(self, ip_address, image_shape, env_config, input_mode):
        self.image_shape = image_shape
        self.sections = env_config["sections"]
        self.input_mode = input_mode

        self.car = airsim.CarClient(ip=ip_address)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.image_shape, dtype=np.uint8)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32) #indica di quando posso curvare (steering) range fra -1 e 1 quindi valori continui


        self.info = {"collision": False}
        self.collision_time = 0
        self.episode_reward = 0
        self.delta_dist = 0
        self.start_time = None
        self.random_start = True
        self.agent_start_X = 0
        self.agent_start_Y = 0
        self.target_X = 0
        self.target_Y = 0
        self.dist_prev = 0
        self.steps_in_episode = 0
        self.animals_moved = False

        self.setup_car()

    def _animals_out(self, neighbourhood=True):

        if neighbourhood:
            animals = ["RaccoonBP2_85", "RaccoonBP3_154", "RaccoonBP4_187", "RaccoonBP_50", "DeerBothBP2_19",
                       "DeerBothBP3_43", "DeerBothBP4_108", "DeerBothBP5_223", "DeerBothBP_12"]
        else:
            animals = []
            objects = self.car.simListSceneObjects()
            for obj in objects:
                if obj[0:7] == "Raccoon" or obj[0:8] == "DeerBoth":
                    animals.append(obj)

        for animal in animals:
            pose = self.car.simGetObjectPose(animal)
            pose.position.x_val += 500
            pose.position.y_val += 500
            self.car.simSetObjectPose(animal, pose)

    def step(self, action):
        self.steps_in_episode += 1
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        # Accumula la reward
        self.episode_reward += reward
        #print(reward)
        # Se l'episodio è terminato, stampa la reward totale
        if done:
            print("> Episode reward: %.2f" % self.episode_reward)
        return obs, reward, done, info

    def reset(self):
        if not AirSimCarEnv.animals_moved:
            self._animals_out()
            AirSimCarEnv.animals_moved = True

        self.episode_reward = 0
        self.steps_in_episode = 0
        self.setup_car()
        obs, _ = self.get_obs()
        self.start_time = time.time()
        return obs

    def render(self):
        return self.get_obs()

    def setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)


        # Get collision time stamp
        self.collision_time = self.car.simGetCollisionInfo().time_stamp

        # Get a random section
        if self.random_start == True:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        self.agent_start_X = section["offset"][0]  # da dove voglio che la macchina parti rispetto a X
        self.target_X = section["target"][0] # dove voglio che la macchina arrivi rispetto a X
        self.agent_start_Y = section["offset"][1] # la Y dalla quale parto
        self.target_Y = section["target"][1]  # dove voglio che la machcina arrivi rispetto a Y

        # Imposta la posizione iniziale
        # Per le auto usiamo y invece di y e z
        y_pos = self.agent_start_Y + np.random.uniform(-1, 1)  # da dove voglio che la macchina parti rispetto a Y (con un certo noise)
        x_pos = self.agent_start_X + np.random.uniform(-1,1)  # add noise
        # Crea una pose iniziale
        yaw_radians = np.deg2rad(section["offset"][2])  # converto in radianti il valore in gradi della rotazione iniziale dell'auto in offset[1], così che venga letto correttamente da quaternion per impostare la rotazione iniziale
        pose = airsim.Pose(airsim.Vector3r(x_pos, y_pos, 0), airsim.to_quaternion(0, 0, yaw_radians))
        self.car.simSetVehiclePose(pose, ignore_collision=True)


        # Get target distance for reward calculation
        x, y, _ = self.car.simGetVehiclePose().position
        self.dist_prev = np.sqrt(np.square(x - self.target_X) + np.square(y - self.target_Y))

    def do_action(self, action):
        steering = float(action[0])  # Controlla la sterzata (valori da -1 a 1)

        # Imposta la velocità target a 10 m/s
        desired_speed = 10  # m/s

        # Leggi la velocità attuale
        car_state = self.car.getCarState()
        current_speed = car_state.speed  # La velocità in m/s

        # Regola il throttle per mantenere la velocità fissa
        throttle = np.clip((desired_speed - current_speed) * 0.1, 0, 1)

        self.car.setCarControls(airsim.CarControls(
            throttle=throttle,
            steering=steering
        ))

        airsim.time.sleep(0.1)  # Attendere un po' per dare tempo all'auto di muoversi

    def get_obs(self):
        # Simile all'implementazione del drone, ma usa il metodo per le auto
        self.info["collision"] = self.is_collision()

        if self.input_mode == "single_rgb":
            obs = self.get_rgb_image()
        elif self.input_mode == "depth":
            obs = self.get_depth_image(thresh=3.4).reshape(self.image_shape)
            obs = ((obs / 3.4) * 255).astype(int)

        return obs, self.info

    def compute_reward(self):
        reward = 0
        done = 0

        elapsed_time = time.time() - self.start_time
        if elapsed_time > 30:
            done = 1

        # 1. Ottieni la posizione attuale dell'auto
        x, y, _ = self.car.simGetVehiclePose().position

        # 4. Calcola la distanza combinata con l'obiettivo 17 in X
        target_dist_curr_2d = np.sqrt(np.square(x - self.target_X) + np.square(y - self.target_Y))

        # 5. Calcola la differenza con la distanza precedente
        self.delta_dist = self.dist_prev - target_dist_curr_2d

        # 6. Aggiorna la distanza precedente
        self.dist_prev = target_dist_curr_2d

        #  reward
        if self.delta_dist > 0:
            reward += np.exp(-(self.dist_prev)/100)  #  10 * delta_dist  # Incentivo se ci si avvicina al target
        else:
            reward += -3*np.exp(-(self.dist_prev)/100)  # Penalità se ci si allontana

        # 6. Penalità per collisione
        if self.is_collision():
            self.episode_reward = self.episode_reward/5  # 0 + 0.1*steps_in_episode
            reward = 0
            done = 1

        if self.dist_prev < 5:
            reward += 100
            done = 1


        return reward, done

    def is_collision(self):
        current_collision_time = self.car.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False

    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False)
        responses = self.car.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))

        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape))




class TestEnv(AirSimCarEnv):
    def __init__(
        self,
        ip_address,
        image_shape,
        env_config,
        input_mode,
        test_mode
    ):
        super(TestEnv, self).__init__(
            ip_address,
            image_shape,
            env_config,
            input_mode
        )

        self.test_mode = test_mode
        self.eps_n = 0
        self.eps_success = 0
        self.episode_reward = 0
        self.random_start = True


    def reset(self):
        self.episode_reward = 0

        if self.random_start:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        if not AirSimCarEnv.animals_moved:
            self._animals_out()
            AirSimCarEnv.animals_moved = True

        self.steps_in_episode = 0
        self.setup_car()
        obs, _ = self.get_obs()
        self.start_time = time.time()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Accumula la reward per questo episodio
        self.episode_reward += reward
        return obs, reward, done, info

    def compute_reward(self):
        reward, done = super().compute_reward()
        if done:
            self.eps_n += 1
            if not self.is_collision():
                self.eps_success += 1
            print("-----------------------------------")
            print("> Total episodes:", self.eps_n)
            print("> Successful runs: %d out of %d" % (self.eps_success, self.eps_n))
            print("> Success rate: %.2f%%" % (self.eps_success * 100 / self.eps_n))
            print("> Episode reward: %.2f" % self.episode_reward)
            print("-----------------------------------\n")
        return reward, done