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

model =  PPO.load(model_save_path + "/model-19", env=env)

# Make save path
try:
    os.mkdir(model_save_path)
except FileExistsError:
    pass

while True:
    total_timesteps += timesteps
    model.learn(total_timesteps=timesteps, tb_log_name = trial_name, reset_num_timesteps = False)
    model.save("trained_models/" + trial_name + "/" "model-" + str(int(total_timesteps / timesteps)))

    # Reset enviroment
    obs = env.reset()

    # Render things1
    for i in range(300):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        print(reward)

        if(done):
            env.reset()

        time.sleep(0.01)