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
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # the action is constrained between [-1,1]---> -1 is the maximum for the left steering


        self.info = {"collision": False}
        self.collision_time = 0
        self.episode_reward = 0
        self.temp_reward = 0
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
        self.episode_reward += reward
        self.temp_reward = self.episode_reward
        # print(reward)
        # if the episode ended, print the cumulative reward
        if done:
            if not hasattr(self, 'test_mode') or self.test_mode != "inference":
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


        # get collision time stamp
        self.collision_time = self.car.simGetCollisionInfo().time_stamp

        # get a random section
        if self.random_start == True:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        self.agent_start_X = section["offset"][0]  # the starting point of the car with respect to X
        self.target_X = section["target"][0] # the target point of the car with respect to X
        self.agent_start_Y = section["offset"][1] # """ Y
        self.target_Y = section["target"][1]  # """ Y

        # set the starting point
        y_pos = self.agent_start_Y #+ np.random.uniform(-1, 1)  # add noise to the starting position
        x_pos = self.agent_start_X #+ np.random.uniform(-1,1)  # add noise
        # create initial pose
        yaw_radians = np.deg2rad(section["offset"][2])  # convert in radiants the angle of the starting position
        pose = airsim.Pose(airsim.Vector3r(x_pos, y_pos, 0), airsim.to_quaternion(0, 0, yaw_radians))
        self.car.simSetVehiclePose(pose, ignore_collision=True)


        # get target distance for reward calculation
        x, y, _ = self.car.simGetVehiclePose().position
        self.dist_prev = np.sqrt(np.square(x - self.target_X) + np.square(y - self.target_Y))

    def do_action(self, action):
        steering = float(action[0])

        # set the desired speed 10 m/s
        desired_speed = 10  # m/s

        # read the ongoing velocity
        car_state = self.car.getCarState()
        current_speed = car_state.speed  # speed in m/s

        # balance throttle to have a fixed velocity
        throttle = np.clip((desired_speed - current_speed) * 0.1, 0, 1)

        self.car.setCarControls(airsim.CarControls(
            throttle=throttle,
            steering=steering
        ))

        #airsim.time.sleep(0.1)  # wait a bit

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        return obs, self.info

    def compute_reward(self):
        reward = 0
        done = 0

        elapsed_time = time.time() - self.start_time
        if elapsed_time > 30:
            done = 1

        # 1. get the actual position
        x, y, _ = self.car.simGetVehiclePose().position

        # 2. compute the distance with the target
        target_dist_curr_2d = np.sqrt(np.square(x - self.target_X) + np.square(y - self.target_Y))

        # 3. compute the difference with the previous distance
        self.delta_dist = self.dist_prev - target_dist_curr_2d

        # 4. update the previous distance
        self.dist_prev = target_dist_curr_2d

        #  reward
        if self.delta_dist > 0:
            reward += 5*(self.delta_dist) #ragiona sul sostituire += con semplicemente = (forse è per questo che nel momento in cui l'auto procede allontanandosi dal target, ci mette troppo tempo prima di accumulare rewards negative)
        else:
            reward += 5*(self.delta_dist)

        # 5. penality for collision
        if self.is_collision():
            self.temp_reward = self.episode_reward/6
            reward = -(5*self.temp_reward) # da modificare per gestire anche il caso in cui la collisione avvenga durante un episodio in cui l'auto stava proseguendo allontanandosi dall'obiettivo (in quel caso la collisione darebbe un apporto positivo e NON va bene)
            done = 1

        if self.dist_prev < 4:
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

        # sometimes no image returns from api
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
        self.steps_in_episode += 1
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        self.episode_reward += reward
        self.temp_reward = self.episode_reward
        # print(reward)
        
        # Stampa statistiche DOPO aver aggiornato episode_reward
        if done and self.test_mode != "inference":
            self.eps_n += 1
            if not self.is_collision():
                self.eps_success += 1
            print("-----------------------------------")
            print("> Total episodes:", self.eps_n)
            print("> Successful runs: %d out of %d" % (self.eps_success, self.eps_n))
            print("> Success rate: %.2f%%" % (self.eps_success * 100 / self.eps_n))
            print("> Episode reward: %.2f" % self.episode_reward)
            print("-----------------------------------\n")
            
        # if the episode ended, print the cumulative reward
        if done:
            if not hasattr(self, 'test_mode') or self.test_mode != "inference":
                print("> Episode reward: %.2f" % self.episode_reward)
                
        return obs, reward, done, info

    def compute_reward(self):
        reward, done = super().compute_reward()

        # in inference mode ignore the 30 seconds threshold
        if self.test_mode == "inference" and done == 1:
            if not self.is_collision():
                done = 0

        # NON stampare più le statistiche qui
        return reward, done