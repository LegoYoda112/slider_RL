from enviroments.SliderEnv import SliderEnv
import time
import os
import glob

from stable_baselines3 import PPO

env = SliderEnv()

model = PPO("MlpPolicy", env, verbose=1, learning_rate = 0.0003, 
      tensorboard_log="./trained_models/tensorboard")
# n_steps = int(8192 * 0.5),
timesteps = 100_000
total_timesteps = 0

trial_name = "position-3"
model_save_path = "./trained_models/" + trial_name


model =  PPO.load(model_save_path + "/model-75", env=env)

while True:
    # Reset enviroment
    obs = env.reset()

    # Render things1
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        
        #  print(reward)
        # print(0.1 + 200 * (action[10:15] + 1) * 0.5 + 50)

        # if(done):
        #     env.reset()

        time.sleep(0.018)