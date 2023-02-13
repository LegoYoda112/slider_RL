from enviroments.SliderEnv import SliderEnv
import time
import os
import glob
import numpy as np 

from stable_baselines3 import PPO

env = SliderEnv()

model = PPO("MlpPolicy", env, verbose=1, learning_rate = 0.003, 
      tensorboard_log="./trained_models/tensorboard")
# n_steps = int(8192 * 0.5),
timesteps = 100_000
total_timesteps = 0

trial_name = "new-feet-21-omni-back-to-old-feet"
model_save_path = "./trained_models/" + trial_name


model =  PPO.load(model_save_path + "/model-96", env=env)

forward = False

speed = 1.0

target_x = 5.0
target_y = 5.0

i = 0


# import csv

# # open the file in the write mode
# f = open('test.csv', 'w')

# # create the csv writer
# writer = csv.writer(f)



while True:
    # Reset enviroment
    obs = env.reset()

    # Render things
    for i in range(10000):
        i+=1

        action, _state = model.predict(obs, deterministic=True)

        i = 0
        for value in obs:
            print(round(value, 2), i)
            i += 1

        # print()

        # action = [-1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0]

        # print(action)

        # pos = [
        #           obs[18],
        #           obs[22],
        #           obs[14],
        #           obs[30],
        #           obs[26],

        #           obs[20],
        #           obs[24],
        #           obs[16],
        #           obs[32],
        #           obs[28],
        # ]
        # # write a row to the csv file
        # # writer.writerow(action)
        # print(pos)
        # writer.writerow(pos)

        # if i > 50:
        #     speed = 1.0

        # if i > 2 * np.pi * 150:
        #     speed = 0.0

        env.v_ref = [0.0, 0.0]

        # if forward:
        #     env.v_ref = [0.8, 0]
        # else:
        #     env.v_ref = [0.0, 0]
        # p_x = env.data.qpos[0]
        # p_y = env.data.qpos[1]

        # print(p_x, p_y)

        # env.v_ref = [(np.sin(i / 100)) * 0.4 + 0.4, 0.0, 0.0

        # if(abs(p_x - target_x) < 0.1 and abs(p_y - target_y) < 0.1):
        #     target_x = np.random.uniform(-5, 5)
        #     target_y = np.random.uniform(-5, 5)

        
        # print((target_x - p_x), (target_y - p_y))

        # env.v_ref = [max(-0.6, min(0.6, (target_x - p_x) * 1.0)), max( -0.4, min(0.4, (target_y - p_y) * 1.0)), 0.0]
        # env.v_ref = [0.2, 0, 0]
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
        # input()
        # time.sleep(0.1)

writer.close()