from enviroments.SliderEnv import SliderEnv
import time
import os
import glob
import numpy as np 

from matplotlib import pyplot as plt

from stable_baselines3 import PPO


trial_name = "model_v15-forward3-1"
model_save_path = "./trained_models/" + trial_name

env = SliderEnv(trial_name)

model =  PPO.load(model_save_path + "/model-25", env=env)

def trial_force(force, render = False):
    # Reset enviroment
    obs = env.reset()

    env.purtrub_max = [0,0,0]
    env.step_time = 0.8
    env.max_ep_time = 100 # seconds

    offset = int(np.random.random() * 30)

    for i in range(300):


        if(i == 100 + offset):
            env.apply_force(force)

        action, _state = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        if render:
            env.render()

        # Fail
        if done:
            return False

    # Sucsess    
    return True

trials_per = 10
force_values = np.linspace(-750, 750, 100)
sucsess_nums = np.zeros(force_values.shape)

for i in range(force_values.shape[0]):
    force_value = force_values[i]

    sucsess_num_trial = 0
    for num in range(trials_per):
        sucsess = trial_force([0, force_value, 0], False)

        sucsess_num_trial += int(sucsess)

    sucsess_nums[i] = sucsess_num_trial

    print(force_value)
    print(sucsess_num_trial)


plt.plot(force_values, sucsess_nums)
plt.show()