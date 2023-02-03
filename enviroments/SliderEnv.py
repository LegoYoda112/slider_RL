from gym import Env, spaces
from gym.utils import seeding

import numpy as np

import mujoco as mj

class SliderEnv(Env):
    def __init__(self):
        super(SliderEnv, self).__init__()

        # ======= PARAMS ======
        # Sim params
        self.sim_steps = 10
        self.max_ep_time = 20 # Seconds

        # Gait params
        self.step_time = 0.6 # s - time per step
        self.stance_time = self.step_time/2.0 # time per stance
        self.phase_offset = 0.5 # percent offset between leg phases

        self.cycle_clock = 0

        self.cost_dict = {}

        self.action_noise_scale = 0.01
        self.action_offset_noise_scale = 0.01

        self.purtrub_max = [50,50,50] # Newtons
        self.purtrub_prob = 0.001 # Probability per timestep

        self.v_ref_change_prob = 0.005
        self.v_ref = [0,0,0]

        # ======= MUJOCO INIT =======
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

         # Init mujoco glfw
        mj.glfw.glfw.init()
        self.window = mj.glfw.glfw.create_window(1500, 750, "Demo", None, None)
        mj.glfw.glfw.make_context_current(self.window)
        mj.glfw.glfw.swap_interval(1)

        mj.glfw.glfw.set_key_callback(self.window, self.key_callback)

        # Create camera
        mj.mjv_defaultCamera(self.cam)
        self.cam.distance = 3
        self.cam.lookat = (1, 0, 1)
        mj.mjv_defaultOption(self.opt)

        # Load a Mujoco model from the xml path
        xml_path = "enviroments/models/flat_world.xml"
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
        self.action_space = spaces.Box(-np.ones(num_actions) * 1, np.ones(num_actions) * 1, dtype = np.float32)

    # Gym reset method
    def reset(self):
        # Reset enviroment
        mj.mj_resetData(self.model, self.data)

        # Reset desired reference velocity
        # x, y, theta
        # self.v_ref = (np.random.uniform(0.5, -0.5), np.random.uniform(0.5, -0.5), np.random.uniform(-0.0, 0.0))
        self.v_ref = (1.0, 0, 0)

        # self.target_torso_height = 0.4

        self.action_offset_noise = np.random.normal(size=(10)) * self.action_offset_noise_scale

        # Randomize starting position and velocity
        self.data.qpos[0] = np.random.uniform(-3, -2)+ 2
        self.data.qvel[0] = np.random.uniform(0.0, 0.4)

        self.data.qpos[1] = np.random.uniform(-2, 2)
        self.data.qvel[1] = np.random.uniform(-0.2, 0.2)

        #robot_starting_height = 0.4
        #self.data.qpos[2] = robot_starting_height
        
        # Joint randomization
        # TODO: Do this properly
        self.data.qpos[7:16] = np.random.uniform(-0.05, 0.05, size = 9)
        self.data.qvel[7:16] = np.random.uniform(-0.1, 0.1, size = 9)

        observation = self.observe()

        return observation


    def render(self):
        # Set camera location
        torso_pos = self.data.body("base_link").xpos
        torso_x = torso_pos[0]
        torso_y = torso_pos[1]
        self.cam.lookat = (torso_x, torso_y, 0.75)
        # self.cam.azimuth = self.data.time * 10
        # self.data.time * 10
        self.cam.azimuth = 45
        self.cam.elevation = -20

        # Render out an image of the enviroment
        viewport = mj.MjrRect(0, 0, int(3000/1.0), int(1500/1.0))
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        mj.glfw.glfw.swap_buffers(self.window)
        mj.glfw.glfw.poll_events()

        return

    def step(self, action):

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

        # print(self.data.body("base_link").xpos[2])
        
        # If we've fallen over, stop the episode6
        if(self.data.body("base_link").xpos[2] < 0.4):
            reward -= 200.0
            # print("fall")
            done = True
        
        # if(np.random.random() < self.v_ref_change_prob):
        #     # print("vref change!")
        #     self.v_ref = (np.random.uniform(1.0, 0.0), np.random.uniform(-0.0, 0.0), np.random.uniform(-0.0, 0.0))

        info = {}

        return observation, reward, done, info

    # Apply an action
    def act(self, action):
        # action = -np.ones(15) * 1
        # print(self.data.actuator)
        # print(self.data.gain)

        # action = action

        # print(self.data.actuator("Left_Slide").ctrl)
        action_noise_flag = 1

        # Apply noise and constant offsets to actions
        action += (np.random.normal(size=(len(action))) * self.action_noise_scale + self.action_offset_noise) * action_noise_flag

        # action[7]

        # ====== Left foot
        # Roll Pitch
        self.data.ctrl[0] = action[0] * 0.3
        self.data.ctrl[2] = action[1] * 0.8
        
        # Slide
        self.data.ctrl[4] = action[2] * 0.2 + 0.1

        # Foot Roll Pitch
        self.data.ctrl[6] = action[3] * 0.5
        self.data.ctrl[8] = action[4] * 0.5

        # ====== Right foot
        # Roll Pitch
        self.data.ctrl[10] = action[5] * 0.3
        self.data.ctrl[12] = action[6] * 0.8
        
        # Slide
        self.data.ctrl[14] = action[7] * 0.2 + 0.1

        # Foot Roll Pitch
        self.data.ctrl[16] = action[8] * 0.5
        self.data.ctrl[18] = action[9] * 0.5


        if(np.random.rand() < self.purtrub_prob):
            F_x = np.random.normal() * self.purtrub_max[0]
            F_y = np.random.normal() * self.purtrub_max[1]
            F_z = np.random.normal() * self.purtrub_max[2]
            self.data.xfrc_applied[2] = [F_x,F_y,F_z,  0,0,0]

            # print(self.data.xfrc_applied[2])
        else:
            self.data.xfrc_applied[2] = [0,0,0,  0,0,0]

        # TORQUE CONTROL
        # self.data.actuator("Left_Slide").ctrl = action[0] * 0.2 + 0.1
        # self.data.actuator("Right_Slide").ctrl = action[1] * 0.2 + 0.1

        # self.data.actuator("Left_Roll").ctrl = action[2] * 0.3
        # self.data.actuator("Right_Roll").ctrl = action[3] * 0.3

        # self.data.actuator("Left_Pitch").ctrl = action[4] * 0.8
        # self.data.actuator("Right_Pitch").ctrl = action[5] * 0.8

        # self.data.actuator("Left_Foot_Pitch").ctrl = action[6] * 0.5
        # self.data.actuator("Right_Foot_Pitch").ctrl = action[7] * 0.5

        # self.data.actuator("Left_Foot_Pitch").ctrl = action[8] * 0.5
        # self.data.actuator("Right_Foot_Pitch").ctrl = action[9] * 0.5

        # Set gain params
        # self.set_actuator_kp_gains(action[10:15])

    def compute_reward(self):
        cost = 0

        self.cycle_clock = self.data.time % self.step_time
        cc = self.cycle_clock

        ground_factor = 1.0

        lf_vel = self.data.sensor("left-foot-vel").data
        rf_vel = self.data.sensor("right-foot-vel").data

        left_force = self.data.sensor("left-foot-touch").data
        right_force = self.data.sensor("right-foot-touch").data

        lf_drag_cost = np.linalg.norm(lf_vel) * left_force[0] 
        rf_drag_cost = np.linalg.norm(rf_vel) * right_force[0]

        # print(lf_drag_cost)
        # print(rf_drag_cost)

        

        self.cost_dict['foot_vel'] = 0
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

        self.cost_dict['foot_vel'] = (lf_drag_cost + rf_drag_cost) * 0.01

        cost += self.cost_dict['foot_vel']
        # print(self.cost_dict['foot_vel'])
        
        # Adjust slide effort compared to other actuator effort
        slide_factor = 1.0
        roll_factor = 1.0
        # Increase ankle effort compared to other actuator effort
        ankle_factor = 1.0

        actuator_effort = (self.data.actuator("Left_Slide").force[0] ** 2  / 200.0) * slide_factor
        actuator_effort += (self.data.actuator("Right_Slide").force[0] ** 2 / 200.0) * slide_factor

        actuator_effort += self.data.actuator("Left_Roll").force[0] ** 2 / 65.0 * roll_factor
        actuator_effort += self.data.actuator("Right_Roll").force[0] ** 2 / 65.0 * roll_factor

        actuator_effort += self.data.actuator("Left_Pitch").force[0] ** 2 / 65.0
        actuator_effort += self.data.actuator("Right_Pitch").force[0] ** 2 / 65.0

        actuator_effort += self.data.actuator("Left_Foot_Roll").force[0] ** 2 / 15.0 * ankle_factor
        actuator_effort += self.data.actuator("Right_Foot_Roll").force[0] ** 2 / 15.0 * ankle_factor
        actuator_effort += self.data.actuator("Left_Foot_Pitch").force[0] ** 2 / 15.0 * ankle_factor
        actuator_effort += self.data.actuator("Right_Foot_Pitch").force[0] ** 2 / 15.0 * ankle_factor
        
        self.cost_dict["effort"] = actuator_effort / 1000.0
        cost += self.cost_dict["effort"]

        # Body velocity tracking cost
        # We split out both to ensure
        # body_vel = self.data.sensor("body-vel").data
        # self.cost_dict["body_vel"] = 0.0 * (self.v_ref[0] - body_vel[0]) ** 2 + 0.0 * (self.v_ref[1] - body_vel[1]) ** 2
        # cost += self.cost_dict["body_vel"]

        # Velocity tracking cost
        forward_v = self.data.qvel[0]
        # if abs(forward_v) < 0.2:
        #     forward_v = 0
        self.cost_dict["body_vel"] = 3.0 * (self.v_ref[0] - self.data.qvel[0]) ** 2 + 3.0 * (self.v_ref[1] - self.data.qvel[1]) ** 2
        cost += self.cost_dict["body_vel"]

        # print(forward_v)

        # Orientation reward
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, self.data.site("Torso").xmat)
        up = np.array([0, 0, 1])
        forward = np.array([1, 0, 0])

        # Generate upwards and forward relative
        up_rel = np.zeros(3)
        mj.mju_rotVecQuat(up_rel, up, quat)
        forward_rel = np.zeros(3)
        mj.mju_rotVecQuat(forward_rel, up, quat)

        self.cost_dict["body_orientation"] = 0.2 * np.linalg.norm([up_rel[0], up_rel[1]])
        self.cost_dict["body_orientation"] = 0.5 * np.linalg.norm([forward_rel[0], forward_rel[1]])
        cost += self.cost_dict["body_orientation"]
        
        # print(self.cost_dict["body_orientation"])
        # self.cost_dict["body_movement"] = np.linalg.norm(self.data.sensor("body-gyro").data))
        self.cost_dict["body_movement"] = 0.02 * np.linalg.norm(self.data.sensor("body-gyro").data)
        self.cost_dict["body_movement"] += 0.01 * np.linalg.norm(self.data.sensor("body-accel").data - np.array([0,0,9.8]))
        cost += self.cost_dict["body_movement"]

        # Add a constant offset to prevent early termination
        reward = (2.0 - cost)

        # Return reward
        return reward

    def observe(self):
        observation = []

        qpos = self.data.qpos
        qvel = self.data.qvel
        body_accel = self.data.sensor("body-accel").data
        body_gyro = self.data.sensor("body-gyro").data

        left_slide = self.data.actuator("Left_Slide")
        right_slide = self.data.actuator("Right_Slide")

        left_roll = self.data.actuator("Left_Roll")
        right_roll = self.data.actuator("Right_Roll")

        left_pitch = self.data.actuator("Left_Pitch")
        right_pitch = self.data.actuator("Right_Pitch")
        
        left_foot_roll = self.data.actuator("Left_Foot_Roll")
        right_foot_roll = self.data.actuator("Right_Foot_Roll")

        left_foot_pitch = self.data.actuator("Left_Foot_Pitch")
        right_foot_pitch = self.data.actuator("Right_Foot_Pitch")
        

        # ======= Body sensors ======
        # Body height
        observation.append(qpos[2])

        # Body velocity
        observation.append(qvel[0])
        observation.append(qvel[1])
        observation.append(qvel[2])

        # Body acceleration
        observation.append(body_accel[0] / 2.0)
        observation.append(body_accel[1] / 2.0)
        observation.append(body_accel[2] / 2.0)

        # Body gyro
        observation.append(body_gyro[0] / 2.0)
        observation.append(body_gyro[1] / 2.0)
        observation.append(body_gyro[2] / 2.0)

        # Body orientation
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, self.data.body("base_link").xmat)
        # print(quat)
        
        observation.append(quat[0])
        observation.append(quat[1])
        observation.append(quat[2])
        observation.append(quat[3])

        # print(self.data.sensor("dist1").data[0])
        # observation.append(self.data.sensor("dist1").data[0])
        # observation.append(self.data.sensor("dist2").data[0])
        # observation.append(self.data.sensor("dist3").data[0])
        # observation.append(self.data.sensor("dist4").data[0])
        # observation.append(self.data.sensor("dist5").data[0])

        # ====== Actuator states ====
        observation.append(left_slide.length)
        observation.append(left_slide.velocity)
        # observation.append(left_slide.force / 200.0)

        observation.append(right_slide.length)
        observation.append(right_slide.velocity)
        # observation.append(right_slide.force / 200.0)

        observation.append(left_roll.length)
        observation.append(left_roll.velocity)
        # observation.append(left_roll.force / 144.0)

        observation.append(right_roll.length)
        observation.append(right_roll.velocity)
        # observation.append(right_roll.force / 144.0)

        observation.append(left_pitch.length)
        observation.append(left_pitch.velocity)
        # observation.append(left_pitch.force / 65.0)

        observation.append(right_pitch.length)
        observation.append(right_pitch.velocity)
        # observation.append(right_pitch.force / 65.0)

        # ===== FOOT =====
        observation.append(left_foot_pitch.length)
        observation.append(left_foot_pitch.velocity)
        # observation.append(left_foot_pitch.force / 15.0)

        observation.append(right_foot_pitch.length)
        observation.append(right_foot_pitch.velocity)
        # observation.append(right_foot_pitch.force / 15.0)

        observation.append(left_foot_roll.length)
        observation.append(left_foot_roll.velocity)
       # observation.append(left_foot_roll.force / 15.0)

        observation.append(right_foot_roll.length)
        observation.append(right_foot_roll.velocity)
        # observation.append(right_foot_roll.force / 15.0)

        twopi = 2 * np.pi

        # === CLOCK ===
        # for i in np.linspace(0.5, 2, 10):
        #     observation.append(10 * np.cos(i * self.cycle_clock * 2 * np.pi / self.step_time))
        #     observation.append(10 * np.sin(i * self.cycle_clock * 2 * np.pi / self.step_time))

        observation.append(10 * np.cos(1 * self.cycle_clock * 2 * np.pi / self.step_time))
        observation.append(10 * np.sin(1 * self.cycle_clock * 2 * np.pi / self.step_time))


        # ==== VREF ====
        observation.append(self.v_ref[0])
        observation.append(self.v_ref[1])

        # print(np.max(observation))

        observation = np.array(observation, dtype = np.float16)

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
            self.v_ref = (-0.5, 0.0, 0.0)
            pass

        if(key == left):
            self.v_ref = (0.0, 0.5, 0.0)
            pass
        
        if(key == right):
            self.v_ref = (0.0, -0.5, -0.0)
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