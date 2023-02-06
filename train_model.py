from enviroments.SliderEnv import SliderEnv
import time
import os
import glob

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

env = SliderEnv()

# 0.00005

model = PPO("MlpPolicy", env, verbose=1, learning_rate = 0.0003, 
      tensorboard_log="./trained_models/tensorboard", n_steps = int(8192 * 0.5))
# n_steps = int(8192 * 0.5),
timesteps = 100_000
total_timesteps = 0

trial_name = "weaker-pid-step-time-0-6"
model_save_path = "./trained_models/" + trial_name

# trial_load_name = "forward-36"
# model_save_path_load = "./trained_models/" + trial_load_name

# model =  PPO.load(model_save_path_load + "/model-83", env=env)

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