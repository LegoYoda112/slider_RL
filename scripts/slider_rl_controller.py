#!/usr/bin/env python3 

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from stable_baselines3 import PPO

import numpy as np

class SliderRLController(Node):

    def __init__(self):
        super().__init__('slider_rl_controller')

        self.get_logger().info("Starting RL controller")

        self.position_goal_publisher = self.create_publisher(
            Float32MultiArray, '/slider/control/motor/position_goals', 10)

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

        self.body_pose = self.create_subscription(
            PoseStamped,
            '/slider/state_estimation/body_pose',
            self.body_pose,
            10
        )

        self.clock_value = 0
        self.dt = 0.01 # wrong don't do this thomas please

        self.observation = np.zeros(40)

        # /Users/thomasg/Documents/nogithub/ros2-ws/src/slider_RL/trained_models

        trial_name = "new-feet-21-omni-back-to-old-feet"
        model_save_path = "/Users/thomasg/Documents/nogithub/ros2-ws/src/slider_RL/trained_models/" + trial_name

        self.model = PPO.load(model_save_path + "/model-96")

        self.create_timer(self.dt, self.run_policy)


    def run_policy(self):

        self.step_time = 0.6

        # self.clock_value += self.dt  # highly incorrect oh my god

        self.observation[34] = 10 * np.cos(1 * self.clock_value * 2 * np.pi / self.step_time)
        self.observation[35] = 10 * np.sin(1 * self.clock_value * 2 * np.pi / self.step_time)
        self.observation[36] = 10 * np.cos(1 * self.clock_value * 4 * np.pi / self.step_time)
        self.observation[37] = 10 * np.sin(1 * self.clock_value * 4 * np.pi / self.step_time)

        self.observation[38] = 0.0 # vref X
        self.observation[39] = 0.0 # vref Y

        action = self.model.predict(self.observation, deterministic=True)[0]

        goal_msg = Float32MultiArray()
        goal_msg.data = np.zeros(10).tolist()

        i = 0
        # action = [-1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0]
        for value in self.observation:
            print(round(value, 2), i)
            i += 1

        print()

        goal_msg.data[0] = action[0]
        goal_msg.data[1] = action[1]
        goal_msg.data[2] = action[2]
        goal_msg.data[3] = action[3]
        goal_msg.data[4] = action[4]

        goal_msg.data[5] = action[5]
        goal_msg.data[6] = action[6]
        goal_msg.data[7] = action[7]
        goal_msg.data[8] = action[8]
        goal_msg.data[9] = action[9]

        self.position_goal_publisher.publish(goal_msg)
        # print(action)
        # do the thing


    def joint_state_callback(self, msg):

        self.clock_value += 0.005

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
        
        self.observation[14] = msg.position[2] # Left slide length
        self.observation[15] =  msg.velocity[2] # Left slide velocity

        self.observation[16] = msg.position[7] # Right slide length
        self.observation[17] = msg.velocity[7] # Right slide velocity

        self.observation[18] = msg.position[0] # Left roll length
        self.observation[19] = msg.velocity[0] # Left roll velocity

        self.observation[20] = msg.position[5] # Right roll length
        self.observation[21] = msg.velocity[5] # Right roll velocity

        self.observation[22] = msg.position[1] # Left pitch length
        self.observation[23] = msg.velocity[1] # Left pitch velocity

        self.observation[24] = msg.position[6] # Right pitch length
        self.observation[25] = msg.velocity[6] # Right pitch velocity


        self.observation[26] = -msg.position[4] # Left foot pitch length
        self.observation[27] = -msg.velocity[4] # Left foot pitch velocity

        self.observation[28] = -msg.position[9] # Right foot pitch length
        self.observation[29] = - msg.velocity[9] # Right foot pitch velocity

        self.observation[30] = msg.position[3] # Left foot roll length
        self.observation[31] = msg.velocity[3] # Left foot roll velocity

        self.observation[32] = -msg.position[8] # Right foot roll length
        self.observation[33] = -msg.velocity[8] # Right foot roll velocity

        pass

    def imu_callback(self, msg):

        # Set body linear acceleration
        self.observation[4] = msg.linear_acceleration.y / 2.0
        self.observation[5] = msg.linear_acceleration.x / 2.0  # might be negative
        self.observation[6] = msg.linear_acceleration.z / 2.0

        # Set body angular velocity
        self.observation[7] = msg.angular_velocity.x / 2.0
        self.observation[8] = msg.angular_velocity.y / 2.0
        self.observation[9] = msg.angular_velocity.z / 2.0

        # Set body orientation
        self.observation[10] = msg.orientation.w * 1
        self.observation[11] = msg.orientation.x * 1
        self.observation[12] = msg.orientation.y * 1
        self.observation[13] = msg.orientation.z * 1

    def body_velocity_callback(self, msg):
        
        self.observation[1] = msg.vector.x * 1.5
        self.observation[2] = msg.vector.y * 1.5
        self.observation[3] = msg.vector.z * 1.5

        pass

    def body_pose(self, msg):
        
        # Set body height
        self.observation[0] = msg.pose.position.z * 0.0
        pass

def main(args=None):
    rclpy.init(args=args)

    slider_rl_controller = SliderRLController()

    rclpy.spin(slider_rl_controller)

    slider_rl_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
