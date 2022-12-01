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

trial_name = "initial_testing_10"
model_save_path = "./trained_models/" + trial_name

# Make save path
try:
    os.mkdir(model_save_path)
except FileExistsError:
    pass

model =  PPO.load(model_save_path + "/model-11", env=env)

while True:
    # Reset enviroment
    obs = env.reset()

    # Render things1
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        

        print(reward)

        # if(done):
        #     env.reset()

        time.sleep(0.015)