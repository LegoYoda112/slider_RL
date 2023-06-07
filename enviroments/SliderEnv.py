from gym import Env, spaces
from gym.utils import seeding

import numpy as np

import mujoco as mj

class SliderEnv(Env):
    def __init__(self, trial_name):
        super(SliderEnv, self).__init__()

        self.trial_name = trial_name

        # ======= PARAMS ======
        # Sim params
        self.sim_steps = 10
        self.max_ep_time = 200 # Seconds

        # Gait params
        self.step_time = 0.8 # s - time per step
        self.stance_time = self.step_time/2.0 # time per stance
        self.phase_offset = 0.5 # percent offset between leg phases

        self.cycle_clock = 0

        self.cost_dict = {}

        self.action_noise_scale = 0.01
        self.action_offset_noise_scale = 0.01

        self.purtrub_max = [10, 10, 10] # Newtons
        self.purtrub_prob = 0.001 # Probability per timestep
        self.purtrub_count = 100
        self.purtrub_timesteps = 5

        self.v_ref_change_prob = 0.001
        self.v_ref = [0,0,0]

        # ======= STATE HISTORY =====
        self.state_size = 31
        self.state_history_length = 3 # states 
        self.state_history = np.zeros(self.state_size * self.state_history_length)

        # ======= MUJOCO INIT =======
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

         # Init mujoco glfw
        mj.glfw.glfw.init()
        self.window = mj.glfw.glfw.create_window(1500, 750, str(self.trial_name), None, None)
        mj.glfw.glfw.make_context_current(self.window)
        mj.glfw.glfw.swap_interval(1)

        mj.glfw.glfw.set_key_callback(self.window, self.key_callback)

        # Create camera
        mj.mjv_defaultCamera(self.cam)
        self.cam.distance = 3
        self.cam.lookat = (1, 0, 1)
        mj.mjv_defaultOption(self.opt)

        # Load a Mujoco model from the xml path
        xml_path = "enviroments/models/flat_world_2022_feet.xml"
        # xml_path2 = "enviroments/models/flat_world_new_feet.xml"
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        # Make a new scene and visual context
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # ====== GYM ==== 
        # State space
        observation_max = 20
        observation_min = -observation_max

        # Auto generate observation off of returned vector from the observe method
        self.observation_shape = self.observe().shape
        self.observation_space = spaces.Box(low = np.ones(self.observation_shape) * observation_min,
                                            high = np.ones(self.observation_shape) * observation_max,
                                            dtype = np.float32)

        # Action space
        # 0-4 leg 0 positions
        # 5-9 leg 1 positions
        # 10-14 leg position gains
        num_actions = 10
        self.action_space = spaces.Box(low = -np.ones(num_actions) * 1, high= np.ones(num_actions) * 1, dtype = np.float32)
        self.action = np.zeros(10)

    # Gym reset method
    def reset(self):
        # Reset enviroment
        mj.mj_resetData(self.model, self.data)

        # Reset desired reference velocity
        # x, y, theta
        # self.v_ref = (np.random.uniform(0.5, -0.5), np.random.uniform(0.0, 0.0), np.random.uniform(-0.0, 0.0))
        self.v_ref = (0.5, 0.0, 0.0)

        self.action_offset_noise = np.random.normal(size=(10)) * self.action_offset_noise_scale

        # Randomize starting position and velocity
        self.data.qpos[0] = np.random.uniform(-2, 2)
        self.data.qpos[2] = 0.75
        self.data.qpos[0] = np.random.uniform(-1.0, -0.5)
        self.data.qvel[0] = np.random.uniform(-0.2, 0.2)

        # self.data.qpos[1] = np.random.uniform(-1.0, 1.0)
        self.data.qvel[1] = np.random.uniform(-0.2, 0.2)

        self.state_history = np.zeros(self.state_size * self.state_history_length)


        # Reset torso position
        self.torso_state_history = []
        self.lf_state_history = []
        self.rf_state_history = []
        self.state_max_length = 2
        self.state_hist_frame_skip = 3
        self.state_hist_frame_counter = 0

        #robot_starting_height = 0.4
        #self.data.qpos[2] = robot_starting_height
        
        # Joint randomization
        # TODO: Do this properly
        # self.data.qpos[7:16] = np.random.uniform(-0.05, 0.05, size = 9)
        # self.data.qvel[7:16] = np.random.uniform(-0.1, 0.1, size = 9)

        observation = self.observe()

        return observation

    def draw_pos_history(self, scene, history, width, color):
        for i in range(1, len(history)):
            current_pos = history[i].copy()
            prev_pos = history[i-1]
            
            mj.mjv_initGeom(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), color)
            mj.mjv_makeConnector(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_CAPSULE, width,
                                prev_pos[0], prev_pos[1], prev_pos[2],
                                current_pos[0], current_pos[1], current_pos[2])
            scene.ngeom+=1

    def render(self):
        # Set camera location
        torso_pos = self.data.body("base_link").xpos
        torso_x = torso_pos[0]
        torso_y = torso_pos[1]
        self.cam.lookat = (torso_x, torso_y, 0.75)
        self.cam.azimuth = self.data.time * 10

        # self.cam.lookat = (2, 0, 0.7)
        self.cam.distance = 10
        self.cam.azimuth = 90
        self.cam.elevation = 0


        self.cam.azimuth = 30
        self.cam.elevation = -20



        # self.data.time * 10
        # self.cam.azimuth = 45

        # Render out an image of the enviroment
        viewport = mj.MjrRect(0, 0, int(3000/1.0), int(1500/1.0))
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)


        # Draw traces
        if(self.state_hist_frame_counter % self.state_hist_frame_skip == 0):
            self.torso_state_history.append(torso_pos.copy())
            self.lf_state_history.append(self.data.body("Left_Foot").xpos.copy())
            self.rf_state_history.append(self.data.body("Right_Foot").xpos.copy())

        self.state_hist_frame_counter += 1

        print(len(self.torso_state_history))

        if(len(self.torso_state_history) > self.state_max_length):
            self.torso_state_history.pop(0)
            self.lf_state_history.pop(0)
            self.rf_state_history.pop(0)

        width = 0.005
        torso_color = np.array([1.0, 0.0, 0.0, 1.0])
        lf_color = np.array([0.0, 1.0, 0.0, 1.0])
        rf_color = np.array([0.0, 0.0, 1.0, 1.0])

        self.draw_pos_history(self.scene, self.torso_state_history, width, torso_color)
        self.draw_pos_history(self.scene, self.lf_state_history, width, lf_color)
        self.draw_pos_history(self.scene, self.rf_state_history, width, rf_color)

        mj.mjr_render(viewport, self.scene, self.context)

        mj.glfw.glfw.swap_buffers(self.window)
        mj.glfw.glfw.poll_events()

        return

    def step(self, action):

        # Apply a purturbation
        self.data.xfrc_applied[2] = [0,0,0,  0,0,0]
        if(np.random.rand() < self.purtrub_prob):
            # print("BONK")
            F_x = np.random.normal() * self.purtrub_max[0]
            F_y = np.random.normal() * self.purtrub_max[1]
            F_z = np.random.normal() * self.purtrub_max[2]
            self.data.xfrc_applied[2] = [F_x,F_y,F_z,  0,0,0]

        # Perform an action
        self.act(action)

        # Advance simulation
        for i in range(self.sim_steps):
            mj.mj_step(self.model, self.data)

        # Take observation
        observation = self.observe()

        # Compute reward
        reward = self.compute_reward()

        # If we are over time, return done
        done = False
        if(self.data.time > self.max_ep_time):
            done = True
        
        # If we've fallen over, stop the episode6
        if(self.data.body("base_link").xpos[2] < 0.4):
            reward -= 200.0
            done = True
        
        # if(np.random.random() < self.v_ref_change_prob):
        #     # print("vref change!")
        #     self.v_ref = (np.random.uniform(1.0, 0.0), np.random.uniform(-0.0, 0.0), np.random.uniform(-0.0, 0.0))

        # if(self.purtrub_count < self.purtrub_timesteps):
        #     self.purtrub_count += 1
        # else:
        #     self.data.xfrc_applied[2] = [0,0,0,  0,0,0]

        info = {}

        return observation, reward, done, info

    def apply_force(self, force):
        F_x = force[0]
        F_y = force[1]
        F_z = force[2]
        self.data.xfrc_applied[2] = [F_x,F_y,F_z,  0,0,0]

        self.purtrub_count = 0

    # Apply an action
    def act(self, action):
        action_noise_flag = 1

        self.action = action

        # Apply noise and constant offsets to actions
        action += (np.random.normal(size=(len(action))) * self.action_noise_scale + self.action_offset_noise) * action_noise_flag

        scale = 1.0

        # Force control

        # self.data.actuator("Left_Slide").ctrl = action[0] * 300
        # self.data.actuator("Right_Slide").ctrl = action[1] * 300

        # self.data.actuator("Left_Roll").ctrl = action[2] * 100
        # self.data.actuator("Right_Roll").ctrl = action[3] * 100

        # self.data.actuator("Left_Pitch").ctrl = action[4] * 65
        # self.data.actuator("Right_Pitch").ctrl = action[5] * 65

        # self.data.actuator("Left_Foot_Pitch").ctrl = action[6] * 15
        # self.data.actuator("Right_Foot_Pitch").ctrl = action[7] * 15

        # self.data.actuator("Left_Foot_Pitch").ctrl = action[8] * 15
        # self.data.actuator("Right_Foot_Pitch").ctrl = action[9] * 15

        # self.data.actuator("Left_Slide").ctrl = -100

        # ====== Left foot
        # Roll Pitch
        self.data.ctrl[0] = action[0] * 0.5 * scale
        self.data.ctrl[2] = action[1] * 0.8 * scale
        
        # Slide
        self.data.ctrl[4] = action[2] * 0.1 * scale

        # Foot Roll Pitch
        self.data.ctrl[6] = action[3] * 0.5 * scale
        self.data.ctrl[8] = action[4] * 0.5 * scale

        # ====== Right foot
        # Roll Pitch
        self.data.ctrl[10] = action[5] * 0.4 * scale
        self.data.ctrl[12] = action[6] * 0.8 * scale
        
        # Slide
        self.data.ctrl[14] = action[7] * 0.1 * scale

        # Foot Roll Pitch
        self.data.ctrl[16] = action[8] * 0.5 * scale
        self.data.ctrl[18] = action[9] * 0.5 * scale

    # Calculate current actuator power
    def actuator_power(self, actuator_name):
        return abs(self.data.actuator(actuator_name).force[0] * self.data.actuator(actuator_name).velocity[0] * 1.0)

    def actuator_force(self, actuator_name):
        return abs(self.data.actuator(actuator_name).force[0])
        
    def compute_reward(self):
        cost = 0

        self.cycle_clock = self.data.time % self.step_time


        # Calculate left and right foot drag costs
        # "drag" is defined as force and velocity
        lf_vel = self.data.sensor("left-foot-vel").data
        rf_vel = self.data.sensor("right-foot-vel").data

        left_force = self.data.sensor("left-foot-touch").data
        right_force = self.data.sensor("right-foot-touch").data

        lf_drag_cost = np.linalg.norm([lf_vel[0], lf_vel[1]]) * left_force[0] 
        rf_drag_cost = np.linalg.norm([rf_vel[0], rf_vel[1]]) * right_force[0]
        self.cost_dict['foot_vel'] = (lf_drag_cost + rf_drag_cost) * 0.015
        cost += self.cost_dict['foot_vel']


        # cc = self.cycle_clock

        # ground_factor = 10.0

        # lf_vel = self.data.sensor("left-foot-vel").data
        # rf_vel = self.data.sensor("right-foot-vel").data

        # self.cost_dict['foot_vel'] = 0
        # # == Left Leg
        # if(cc > self.stance_time):
        #     # STANCE
        #     self.cost_dict['foot_vel'] += ground_factor * np.linalg.norm(lf_vel)
        #     pass
        # else:
        #     # SWING
        #     pass
        
        # # == Right foot
        # if(cc < self.stance_time):
        #     # STANCE
        #     self.cost_dict['foot_vel'] += ground_factor * np.linalg.norm(rf_vel)
        #     pass
        # else:
        #     # SWING
        #     pass

        # cost += self.cost_dict['foot_vel']
        
        # Adjust slide effort compared to other actuator effort
        slide_factor = 0.1
        roll_factor = 1.0

        # Lower ankle effort compared to other actuator effort
        ankle_factor = 1.0

        actuator_effort = self.actuator_power("Left_Slide") ** 2 * slide_factor
        actuator_effort += self.actuator_power("Right_Slide") ** 2 * slide_factor

        actuator_effort += self.actuator_power("Left_Roll") ** 2 * roll_factor
        actuator_effort += self.actuator_power("Right_Roll") ** 2 * roll_factor

        actuator_effort += self.actuator_power("Left_Pitch") ** 2 * 1.0
        actuator_effort += self.actuator_power("Right_Pitch") ** 2 * 1.0

        actuator_effort += self.actuator_power("Left_Foot_Roll") ** 2 * ankle_factor
        actuator_effort += self.actuator_power("Right_Foot_Roll") ** 2 * ankle_factor
        actuator_effort += self.actuator_power("Left_Foot_Pitch") ** 2 * ankle_factor
        actuator_effort += self.actuator_power("Right_Foot_Pitch") ** 2 * ankle_factor
        
        self.cost_dict["effort"] = actuator_effort / 200000.0
        cost += self.cost_dict["effort"]
        
        # actuator_force = 0
        # actuator_force += (self.actuator_force("Left_Slide") / 600.0) ** 2
        # actuator_force += (self.actuator_force("Left_Slide") / 600.0) ** 2

        # actuator_force += (self.actuator_force("Left_Roll") / 100.0) ** 2
        # actuator_force += (self.actuator_force("Right_Roll") / 100.0) ** 2
        # actuator_force += (self.actuator_force("Left_Pitch") / 100.0) ** 2
        # actuator_force += (self.actuator_force("Right_Pitch") / 100.0) ** 2

        # actuator_force += (self.actuator_force("Left_Foot_Roll") / 100.0) ** 2
        # actuator_force += (self.actuator_force("Right_Foot_Roll") / 100.0) ** 2
        # actuator_force += (self.actuator_force("Left_Foot_Pitch") / 100.0) ** 2
        # actuator_force += (self.actuator_force("Right_Foot_Pitch") / 100.0) ** 2


        # self.cost_dict["force"] = actuator_force * 0.0
        # print(self.cost_dict["force"])
        # cost += self.cost_dict["force"]

        # Body velocity cost
        self.cost_dict["body_vel"] = 5.0 * (self.v_ref[0] - self.data.qvel[0]) ** 2 + 2.0 * (self.v_ref[1] - self.data.qvel[1]) ** 2

        # if(self.cost_dict["body_vel"] < 0.001):
        #     self.cost_dict["body_vel"] = 0.0
        cost += self.cost_dict["body_vel"]

        # Orientation reward
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, self.data.site("Torso").xmat)
        up = np.array([0, 0, 1])
        forward = np.array([1, 0, 0])

        # Generate upwards and forward relative
        up_rel = np.zeros(3)
        mj.mju_rotVecQuat(up_rel, up, quat)
        forward_rel = np.zeros(3)
        mj.mju_rotVecQuat(forward_rel, forward, quat)

        self.cost_dict["body_orientation"] = 2.5 * np.linalg.norm([up_rel[0], up_rel[1]])
        self.cost_dict["body_orientation"] += 0.2 * np.linalg.norm([forward_rel[1], forward_rel[2]])
        cost += self.cost_dict["body_orientation"]
        
        # Body movement cost
        self.cost_dict["body_movement"] = 0.02 * np.linalg.norm(self.data.sensor("body-gyro").data)
        self.cost_dict["body_movement"] += 0.01 * np.linalg.norm(self.data.sensor("body-accel").data - np.array([0,0,9.8]))
        cost += self.cost_dict["body_movement"]

        # Add a constant offset to prevent early termination
        reward = (2.0 - cost)

        # Return reward
        return reward

    def observe(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        pos_noise_scale = np.random.normal(size=(15)) * 0.01
        vel_noise_scale = np.random.normal(size=(16)) * 0.01

        # === Full state === (minus x and y position)
        state = np.concatenate((qpos[2:] + pos_noise_scale, qvel + vel_noise_scale))

        #print(state)
        #print()

        # Shuffle state history around
        self.state_history[0:self.state_size] = self.state_history[self.state_size:2*self.state_size]
        self.state_history[1*self.state_size:2*self.state_size] = self.state_history[2*self.state_size:3*self.state_size]
        self.state_history[2*self.state_size:3*self.state_size] = state

        # === CLOCK ===
        clock = []

        # clock.append(10 * np.cos(0.5 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(0.6 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(0.7 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(0.8 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(0.9 * self.cycle_clock * 2 * np.pi / self.step_time))
        clock.append(10 * np.cos(1.0 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.1 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.2 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.3 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.4 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.5 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.6 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.7 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.8 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(1.9 * self.cycle_clock * 2 * np.pi / self.step_time))
        # clock.append(10 * np.cos(2.0 * self.cycle_clock * 2 * np.pi / self.step_time))

        # clock.append(10 * np.sin(1 * self.cycle_clock * 2 * np.pi / self.step_time))

        #clock.append(10 * np.cos(1 * self.cycle_clock * 4 * np.pi / self.step_time))
        #clock.append(10 * np.sin(1 * self.cycle_clock * 4 * np.pi / self.step_time))

        # ==== VREF ====
        vref = []

        vref.append(self.v_ref[0])
        vref.append(self.v_ref[1])

        # print(vref)
        observation = np.array(np.concatenate((self.state_history, clock, vref)), dtype = np.float16)

        return observation

    # Handle keyboard callbacks (for teleop)
    def key_callback(self, window, key, scancode, action, mods):
        up = 265
        down = 264
        left = 263
        right = 262
        space = 32

        if(key == up):
            self.v_ref = (0.5, 0.0, 0.0)
            pass

        if(key == down):
            self.v_ref = (-0.2, 0.0, 0.0)
            pass

        if(key == left):
            self.v_ref = (0.0, 0.2, 0.0)
            pass
        
        if(key == right):
            self.v_ref = (0.0, -0.2, -0.0)
            pass

        if(key == space):
            self.v_ref = (0.0, 0.0, 0.0)
            pass

        print(self.v_ref)

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        mj.glfw.glfw.terminate()