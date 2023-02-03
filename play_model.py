from enviroments.SliderEnv import SliderEnv
import time
import os
import glob
import numpy as np 

from stable_baselines3 import PPO

env = SliderEnv()

model = PPO("MlpPolicy", env, verbose=1, learning_rate = 0.0005, 
      tensorboard_log="./trained_models/tensorboard")
# n_steps = int(8192 * 0.5),
timesteps = 100_000
total_timesteps = 0

trial_name = "forward-35"
model_save_path = "./trained_models/" + trial_name


model =  PPO.load(model_save_path + "/model-18", env=env)

forward = False

speed = 1.0

while True:
    # Reset enviroment
    obs = env.reset()

    # Render things
    for i in range(10000):

        action, _state = model.predict(obs, deterministic=True)

        # if i > 50:
        #     speed = 1.0

        # if i > 2 * np.pi * 150:
        #     speed = 0.0

        # env.v_ref = [-0.5, 0]

        # if forward:
        #     env.v_ref = [0.8, 0]
        # else:
        #     env.v_ref = [0.0, 0]
        # env.v_ref = [(np.sin(i / 100))/2.0, 0.0, 0.0]
        # env.v_ref = [(np.cos(i / 150))/2.0 * speed, (np.sin(i / 150))/2.0 * speed, 0.0]
            #print("SWITCH")

        obs, reward, done, info = env.step(action)
        env.render()

        #quat = np.array([obs[10], obs[11], obs[12], obs[13]])
        #print(quat)

        # print(round(obs[-2], 2))
        # print(obs[1])
        # print(obs[2])
        # print(reward)
        # print(reward)
        # print(0.1 + 200 * (action[10:15] + 1) * 0.5 + 50)

        # if(done):
        #     env.reset()

        time.sleep(0.015)