
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from blueboat_interfaces.msg import BlueboatActuatorInputs
from std_msgs.msg import Float32MultiArray

import numpy as np
from gym_asv_ros2.gym_asv.entities import CircularEntity
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer, BG_PMG_PATH
from gym_asv_ros2.gym_asv.environment import RandomDockEnv

class SimulationNode(Node):

    def __init__(self):
        super().__init__("gym_asv_sim_node")

        self.aciton_sub = self.create_subscription(
            BlueboatActuatorInputs,
            "/thruster_input",
            self.thruster_input_callback,
            10
        )

        self.observation_pub = self.create_publisher(
            Float32MultiArray,
            "/observation_space",
            10
        )
        # Initialize env
        self.env = RandomDockEnv(render_mode="human")
        self.env.reset()
        self.env.render()

        self.action = np.array([0.0, 0.0])
        self.wait_for_action_duration = Duration(seconds=5)
        self.last_action_resived = self.get_clock().now() - self.wait_for_action_duration

        self.create_timer(0.01, self.render_callback)
        self.create_timer(0.1, self.publish_observation)

    def __del__(self):
        self.env.close()

    def render_callback(self):

        current_time = self.get_clock().now()
        if current_time > self.last_action_resived + self.wait_for_action_duration:
            self.get_logger().info(
                "It is more than 10 seconds since we got an thrust update.\
                    setting Thrust to 0 for safty reasons")
            self.action[0], self.action[1] = 0.0, 0.0
        
        observation, reward, done, truncated, info = self.env.step(self.action)
        self.last_observation = observation

        if done:
            self.env.reset()
            self.publish_observation()

        self.env.render()

    def publish_observation(self):
        # Publish observation
        observation_msg = Float32MultiArray()
        observation_msg.data = self.last_observation.flatten().tolist()
        self.observation_pub.publish(observation_msg)


    def thruster_input_callback(self, msg: BlueboatActuatorInputs):
        self.get_logger().info(f"got thruster msg: {msg.stb_prop_force, msg.port_prop_force}")
        self.action[0] = msg.stb_prop_force
        self.action[1] = msg.port_prop_force
        self.last_action_resived = self.get_clock().now()


def main(args=None):
    rclpy.init(args=args)
    simulator_node = SimulationNode()
    rclpy.spin(simulator_node)

    simulator_node.destroy_node()
    rclpy.shutdown()
   




