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

        # ======= MUJOCO INIT =======
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

         # Init mujoco glfw
        mj.glfw.glfw.init()
        self.window = mj.glfw.glfw.create_window(1500, 750, "Demo", None, None)
        mj.glfw.glfw.make_context_current(self.window)
        mj.glfw.glfw.swap_interval(1)

        # mj.glfw.glfw.set_key_callback(self.window, self.key_callback)

        # Create camera
        mj.mjv_defaultCamera(self.cam)
        self.cam.distance = 5
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

        # Action space, both slide positions
        self.action_space = spaces.Box(-np.ones(10) * 1, np.ones(10) * 1, dtype = np.float32)

    # Gym reset method
    def reset(self):
        # Reset enviroment
        mj.mj_resetData(self.model, self.data)

        # Reset desired reference velocity
        # x, y, theta
        self.v_ref = (np.random.uniform(0.5, 0.0), np.random.uniform(0.0, 0.0), np.random.uniform(-0.0, 0.0))
        self.v_ref = (0.8, 0, 0)

        # Randomize starting position and velocity
        self.data.qpos[0] = np.random.uniform(-3, -2)
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
        self.cam.lookat = (torso_x, torso_y, 1.0)
        self.cam.azimuth = self.data.time * 10
        # self.cam.azimuth = 90
        self.cam.elevation = -15

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
            reward -= 100.0
            # print("fall")
            done = True
        
        info = {}

        return observation, reward, done, info

    # Apply an action
    def act(self, action):
        self.data.actuator("Left_Slide").ctrl = action[0] * 0.2 + 0.1
        self.data.actuator("Right_Slide").ctrl = action[1] * 0.2 + 0.1

        self.data.actuator("Left_Roll").ctrl = action[2] * 0.3
        self.data.actuator("Right_Roll").ctrl = action[3] * 0.3

        self.data.actuator("Left_Pitch").ctrl = action[4] * 0.8
        self.data.actuator("Right_Pitch").ctrl = action[5] * 0.8

        self.data.actuator("Left_Foot_Pitch").ctrl = action[6] * 0.5
        self.data.actuator("Right_Foot_Pitch").ctrl = action[7] * 0.5

        self.data.actuator("Left_Foot_Pitch").ctrl = action[8] * 0.5
        self.data.actuator("Right_Foot_Pitch").ctrl = action[9] * 0.5

    def compute_reward(self):
        cost = 0

        self.cycle_clock = self.data.time % self.step_time
        cc = self.cycle_clock

        ground_factor = 1.0

        lf_vel = self.data.sensor("left-foot-vel").data
        rf_vel = self.data.sensor("right-foot-vel").data

        self.cost_dict['foot_vel'] = 0

        # == Left Leg
        if(cc > self.stance_time):
            # STANCE
            self.cost_dict['foot_vel'] += ground_factor * np.linalg.norm(lf_vel)
            pass
        else:
            # SWING
            pass
        
        # == Right foot
        if(cc < self.stance_time):
            # STANCE
            self.cost_dict['foot_vel'] += ground_factor * np.linalg.norm(rf_vel)
            pass
        else:
            # SWING
            pass

        cost += self.cost_dict['foot_vel']
        

        actuator_effort = self.data.actuator("Left_Slide").force[0] ** 2
        actuator_effort += self.data.actuator("Right_Slide").force[0] ** 2
        actuator_effort = self.data.actuator("Left_Roll").force[0] ** 2
        actuator_effort += self.data.actuator("Right_Roll").force[0] ** 2
        actuator_effort += self.data.actuator("Left_Pitch").force[0] ** 2
        actuator_effort += self.data.actuator("Right_Pitch").force[0] ** 2
        actuator_effort += self.data.actuator("Left_Foot_Roll").force[0] ** 2
        actuator_effort += self.data.actuator("Right_Foot_Roll").force[0] ** 2
        actuator_effort += self.data.actuator("Left_Foot_Pitch").force[0] ** 2
        actuator_effort += self.data.actuator("Right_Foot_Pitch").force[0] ** 2
        
        self.cost_dict["effort"] = actuator_effort / 10000.0
        cost += self.cost_dict["effort"]
        
        # Velocity tracking cost
        self.cost_dict["body_vel"] = 1.0 * (self.v_ref[0] - self.data.qvel[0]) ** 2 + 1.0 * (self.v_ref[1] - self.data.qvel[1]) ** 2
        cost += self.cost_dict["body_vel"]

        # print(self.data.qvel[0])

        # Add a constant offset to prevent early termination
        reward = 1.0 - cost

        # print(reward)

        # Return reward
        return reward

    def observe(self):
        observation = []

        qpos = self.data.qpos
        qvel = self.data.qvel

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
        
        # Body height
        observation.append(qpos[2])

        # Body velocity
        observation.append(qvel[0])
        observation.append(qvel[1])
        observation.append(qvel[2])

        # Body orientation
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, self.data.body("base_link").xmat)
        
        observation.append(quat[0])
        observation.append(quat[1])
        observation.append(quat[2])
        observation.append(quat[3])

        # Actuator states
        observation.append(left_slide.length)
        observation.append(left_slide.velocity)
        # observation.append(left_slide.force)

        observation.append(right_slide.length)
        observation.append(right_slide.velocity)
        # observation.append(right_slide.force)

        observation.append(left_roll.length)
        observation.append(left_roll.velocity)
        # observation.append(left_roll.force)

        observation.append(right_roll.length)
        observation.append(right_roll.velocity)
        # observation.append(right_roll.force)

        observation.append(left_pitch.length)
        observation.append(left_pitch.velocity)
        # observation.append(left_pitch.force)

        observation.append(right_pitch.length)
        observation.append(right_pitch.velocity)
        # observation.append(right_pitch.force)

        # ===== FOOT =====
        observation.append(left_foot_pitch.length)
        observation.append(left_foot_pitch.velocity)
        # observation.append(left_foot_pitch.force)

        observation.append(right_foot_pitch.length)
        observation.append(right_foot_pitch.velocity)
        # observation.append(right_foot_pitch.force)

        observation.append(left_foot_roll.length)
        observation.append(left_foot_roll.velocity)
        # observation.append(left_foot_roll.force)

        observation.append(right_foot_roll.length)
        observation.append(right_foot_roll.velocity)
        # observation.append(right_foot_roll.force)

        # === CLOCK ===
        observation.append(10 * np.cos(self.cycle_clock * 2 * np.pi / self.step_time))
        observation.append(10 * np.cos(self.cycle_clock * 2 * np.pi / self.step_time + self.phase_offset * 2 * np.pi))


        # print(np.max(observation))

        observation = np.array(observation, dtype = np.float16)

        return observation

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        mj.glfw.glfw.terminate()