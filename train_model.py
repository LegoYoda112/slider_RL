from enviroments.SliderEnv import SliderEnv
import time
import os
import glob

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback



timesteps = 500_000
total_timesteps = 0

trial_name = "model_v17-forward9"
model_save_path = "./trained_models/" + trial_name


env = SliderEnv(trial_name)

model = PPO("MlpPolicy", env, verbose=1, learning_rate = 0.0002, 
      tensorboard_log="./trained_models/tensorboard", n_steps = int(8192 * 0.5))

# n_steps = int(8192 * 0.5)

load = True

if(load): 
    trial_load_name = "model_v17-forward7"
    model_save_path_load = "./trained_models/" + trial_load_name

    model =  PPO.load(model_save_path_load + "/model-39", env=env, learning_rate = 0.00005)

# Make save path
try:
    os.mkdir(model_save_path)
except FileExistsError:
    pass

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        cost_dict = env.cost_dict

        for key in cost_dict:
            self.logger.record("cost/" + key, cost_dict[key])

        # self.logger.record("reward", value)
        return True

#Seed the enviroment
env.seed(422)

# env.purtrub_max = [500, 500, 500]

while True:


    total_timesteps += timesteps
    model.learn(total_timesteps=timesteps, tb_log_name = trial_name, reset_num_timesteps = False, callback=TensorboardCallback())
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