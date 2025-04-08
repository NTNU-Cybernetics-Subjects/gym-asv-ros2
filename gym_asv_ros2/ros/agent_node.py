
from gym_asv_ros2.gym_asv.environment import RandomGoalWithDockObstacle
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from blueboat_interfaces.msg import BlueboatActuatorInputs, BlueboatState

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


        # Lidar
        self.lidar_sub = self.create_subscription(
            Float32MultiArray,
            "/blueboat/lidar",
            self.lidar_sub_callback,
            10
        )
        self.real_lidar_messurments = np.ones((41,))
        self.last_lidar_recived = self.get_clock().now()

        # Vessel
        self.vessel_state_sub = self.create_subscription(
            BlueboatState,
            "/blueboat/state",
            self.state_sub_callback,
            10
        )
        self.real_vessel_state = np.zeros((6,))
        self.last_vessel_state_recived = self.get_clock().now()

        # Action
        self.action_pub = self.create_publisher(
            BlueboatActuatorInputs,
            "/blueboat/u",
            10
        )

        self.agent = PPO.load(agent_file)

        # Environemnt
        # self.env = RandomGoalWithDockObstacle(render_mode=None)
        # self.simulated_observation, _ = self.env.reset()

        self.run_fequency = 0.01
        self.create_timer(self.run_fequency, self.run)


    # FIXME: This needs to be modified according to the actual sensor data
    def lidar_sub_callback(self, msg: Float32MultiArray):

        self.real_lidar_messurments = np.array(msg.data)

        self.last_lidar_recived = self.get_clock().now()

    def state_sub_callback(self, msg: BlueboatState):

        self.real_vessel_state[0] = msg.x
        self.real_vessel_state[1] = msg.y
        self.real_vessel_state[2] = msg.yaw
        self.real_vessel_state[3] = msg.surge
        self.real_vessel_state[4] = msg.sway
        self.real_vessel_state[5] = msg.yaw_r

        self.last_vessel_state_recived = self.get_clock().now()


    def run(self):

        # TODO: Merge with Env?
        observation = np.concatenate([self.real_vessel_state, self.real_lidar_messurments])

        # Predict from real messurments?
        action, _states = self.agent.predict(observation, deterministic=True)
        action = action.flatten()

        # self.simulated_observation, reward, done, truncated, info = self.env.step(action)
        # if done or truncated:
        #     self.get_logger().info("Simulations reached goal")

        # Publish action
        action_msg = BlueboatActuatorInputs(
            stb_prop_force=float(action[0]),
            port_prop_force=float(action[1])
        )
        self.action_pub.publish(action_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

