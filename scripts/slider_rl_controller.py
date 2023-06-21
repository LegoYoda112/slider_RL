#!/usr/bin/env python3 


print("importing rclpy")
import rclpy

print("importing ros packages")
from rclpy.node import Node
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

import time

print("importing PPO")
from stable_baselines3 import PPO

print("importing numpy")
import numpy as np

print("done importing")

class SliderRLController(Node):

    def __init__(self):
        super().__init__('slider_rl_controller')

        self.get_logger().info("Starting RL controller")

        # ======= STATE HISTORY =====
        self.qpos = np.zeros(17) # body quaternion, body position, joint position
        self.qvel = np.zeros(16) # body angular vel, body vel, joint position
        self.sin_clock = np.zeros(1)
        self.vref = np.zeros(2)

        self.state_size = 31

        self.state_history = np.zeros(3 * self.state_size)

        observation_length = (len(self.sin_clock) +
                              len(self.vref) + 
                              len(self.state_history))
        
        self.observation = np.zeros(observation_length)

        model_save_path = "/home/thoma/Documents/ros2_ws/src/slider_RL/rl_models"

        self.model = PPO.load(model_save_path + "/model-40")
        # self.model = PPO.load(model_save_path + "/model-39-slow")

        self.position_goal_publisher = self.create_publisher(
            Float32MultiArray, '/slider/control/joint/position_goals', 10)

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/slider/sensors/imu/imu',
            self.imu_callback,
            10
        )

        self.body_velocity_subscriber = self.create_subscription(
            Vector3Stamped,
            '/slider/state_estimation/body_velocity',
            self.body_velocity_callback,
            10
        )

        self.body_position_subscriber = self.create_subscription(
            PoseStamped,
            '/slider/state_estimation/body_pose',
            self.body_position_callback,
            10
        )

        self.joy_subscriber = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        self.joint_states = None

        # Wait until we have a joint state
        while(self.joint_states == None):
            rclpy.spin_once(self)
            print("waiting")
            
            pass

        for i in range(10):
            rclpy.spin_once(self)
            # time.sleep(0.001)
        
        print(self.joint_states.position)
        
        self.move_time = 2.0 # seconds
        self.move_steps = int(100.0 * self.move_time)

        self.initial_position = np.array(self.joint_states.position)

        starting_position = np.array([0.0, # Right_Roll
            0.0, # Right_Pitch
            0.0, # Right_Slide
            0.0, # Right_Foot_Roll
            0.0, # Right_Foot_Pitch

            0.0, # Left_Roll
            0.0, # Left_Pitch
            0.0, # Left_slide
            0.0, # Left_Foot_Pitch
            0.0]) # check for user input

        for t in range(0, self.move_steps):
            factor = 1 - t / self.move_steps
            new_states = (1.0 - factor) * starting_position + factor * self.initial_position
            print(new_states)

            self.pub_joint_states(new_states)
            time.sleep(0.01)

        self.action_scale = 0.0

        self.clock_value = 0
        self.dt = 0.01 # wrong don't do this thomas please

        self.abort = False

        # /Users/thomasg/Documents/nogithub/ros2-ws/src/slider_RL/trained_models
        # /home/thoma/Documents/ros2_ws/src/slider_RL/trained_models

        self.create_timer(self.dt, self.run_policy)

    def pub_joint_states(self, array):
        msg = Float32MultiArray()
        msg.data = list(array)

        self.position_goal_publisher.publish(msg)

    def run_policy(self):

        self.step_time = 0.9
        self.clock_value += self.dt  # highly incorrect oh my god
        self.cycle_clock = self.clock_value


        # Make full state
        self.state = np.concatenate((self.qpos[2:], self.qvel))

        # Shuffle state history around
        self.state_history[0:self.state_size] = self.state_history[self.state_size:2*self.state_size]
        self.state_history[self.state_size:2*self.state_size] = self.state_history[2*self.state_size:3*self.state_size]
        self.state_history[2*self.state_size:3*self.state_size] = self.state

        # Make clock signal
        self.sin_clock = np.array([10 * np.cos(1 * self.cycle_clock * 2 * np.pi / self.step_time)])

        # Set up target velocity
        self.vref = np.array([0.0, 0.0])

        #print(self.state)
        #print()

        # Build observation vector and append
        self.observation = np.array(np.concatenate((self.state_history, self.sin_clock, self.vref)), dtype = np.float16)
        action = self.model.predict(self.observation, deterministic=True)[0]

        # print(self.observation)
        goal_msg = Float32MultiArray()
        goal_msg.data = np.zeros(10).tolist()

        # i = 0
        # for value in self.observation:
        #     print(round(value, 2), i)
        #     i += 1

        #print()
        #print(action)
        #print()

        goal_msg.data[0] = action[0] * 0.3 * self.action_scale
        goal_msg.data[1] = action[1] * 0.8 * self.action_scale
        goal_msg.data[2] = action[2] * 0.1 * self.action_scale * 1.0
        goal_msg.data[3] = action[3] * 0.5 * self.action_scale
        goal_msg.data[4] = action[4] * 0.5 * self.action_scale - 0.00

        goal_msg.data[5] = action[5] * 0.3 * self.action_scale
        goal_msg.data[6] = action[6] * 0.8 * self.action_scale
        goal_msg.data[7] = action[7] * 0.1 * self.action_scale * 1.0 
        goal_msg.data[8] = action[8] * 0.5 * self.action_scale + 0.0 # Left Foot Pitch
        goal_msg.data[9] = action[9] * 0.5 * self.action_scale + 0.00

        print(goal_msg.data)
        print()

        if(self.abort == False):
            self.position_goal_publisher.publish(goal_msg)
            pass
        else:
            self.get_logger().error("STOPPED")


    def joint_state_callback(self, msg):
        
        self.joint_states = msg

        # 0 "left_roll"
        # 1 "left_pitch"
        # 2 "left_slide"
        # 3 "left_foot_roll"
        # 4 "left_foot_pitch"

        # 5 "right_roll"
        # 6 "right_pitch"
        # 7 "right_slide"
        # 8 "right_foot_roll"
        # 9 "right_foot_pitch"
        
        self.qpos[7:17] = np.array(msg.position)
        self.qvel[6:16] = np.array(msg.velocity)

        pass

    def imu_callback(self, msg):

        # Set body angular velocity
        self.qvel[3] = msg.angular_velocity.x
        self.qvel[4] = msg.angular_velocity.y
        self.qvel[5] = msg.angular_velocity.z

        # Set body orientation
        self.qpos[3] = msg.orientation.w 
        self.qpos[4] = msg.orientation.x
        self.qpos[5] = msg.orientation.y 
        self.qpos[6] = msg.orientation.z

    def body_velocity_callback(self, msg):
        
        self.qvel[0] = msg.vector.x
        self.qvel[1] = msg.vector.y
        self.qvel[2] = msg.vector.z

        pass

    def body_position_callback(self, msg):
        
        # Set body height
        self.qpos[0] = msg.pose.position.x
        self.qpos[1] = msg.pose.position.y
        self.qpos[2] = msg.pose.position.z
        pass

    def joy_callback(self, msg):
        self.action_scale = 1 - (msg.axes[5] + 1) / 2.0

        # print(self.action_scale)

        if(msg.buttons[1]):
            self.abort = True

def main(args=None):
    rclpy.init(args=args)

    slider_rl_controller = SliderRLController()

    rclpy.spin(slider_rl_controller)

    slider_rl_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
