
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from blueboat_interfaces.msg import BlueboatActuatorInputs

import numpy as np
from stable_baselines3 import PPO

class AgentNode(Node):

    def __init__(self) -> None:
        super().__init__("gym_asv_agent_node")

        self.declare_parameter("agent", "")
        agent_file = self.get_parameter("agent").get_parameter_value().string_value
        self.get_logger().info(f"agent file is: {agent_file}")
        if not agent_file:
            raise FileExistsError(f"{agent_file} does not exists.")

        self.observation_sub = self.create_subscription(
            Float32MultiArray,
            "/observation_space",
            self.observation_callback,
            10
        )
        self.action_pub = self.create_publisher(
            BlueboatActuatorInputs,
            "/thruster_input",
            10
        )

        self.agent = PPO.load(agent_file)

    def observation_callback(self, obs_msg: Float32MultiArray):

        self.get_logger().info(f"Recived observation: {obs_msg}")
        observation = np.array([ obs_msg.data ])
        self.get_logger().info(f"Formating observation: {observation}")

        action, _states = self.agent.predict(observation, deterministic=True)

        # Publish action
        action_msg = BlueboatActuatorInputs()
        action_msg.stb_prop_force = float(action[0])
        action_msg.port_prop_force = float(action[1])
        self.action_pub.publish(action_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

