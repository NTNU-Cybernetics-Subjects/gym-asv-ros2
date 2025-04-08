
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from blueboat_interfaces.msg import BlueboatActuatorInputs, BlueboatState
from std_msgs.msg import Float32MultiArray

import numpy as np
from gym_asv_ros2.gym_asv.entities import CircularEntity
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer, BG_PMG_PATH
from gym_asv_ros2.gym_asv.environment import RandomGoalWithDockObstacle

class SimulationNode(Node):

    def __init__(self):
        super().__init__("gym_asv_sim_node")

        self.action_sub = self.create_subscription(
            BlueboatActuatorInputs,
            "/blueboat/u",
            self.thruster_input_callback,
            10
        )

        self.vessel_state_pub = self.create_publisher(
            BlueboatState,
            "/blueboat/state",
            10
        )

        self.lidar_pub = self.create_publisher(
            Float32MultiArray,
            "/blueboat/lidar",
            10
        )

        # Initialize env
        self.env = RandomGoalWithDockObstacle(render_mode="human")
        self.env.reset()
        self.env.render()

        # Action
        self.action = np.array([0.0, 0.0])

        render_frequence = 0.01
        observation_pub_frequence = 0.1
        self.create_timer(render_frequence, self.render_callback)
        self.create_timer(observation_pub_frequence, self.publish_observation)

    def __del__(self):
        self.env.close()

    def render_callback(self):
        
        observation, reward, done, truncated, info = self.env.step(self.action)
        self.last_observation = observation

        if done or truncated:
            self.env.reset()
            self.publish_observation()

        self.env.render()

    def publish_observation(self):

        # Publish state
        sim_state = self.last_observation.flatten()[:6]
        sim_state_msg = BlueboatState(
            x=sim_state[0],
            y=sim_state[1],
            yaw=sim_state[2],
            surge=sim_state[3],
            sway=sim_state[4],
            yaw_r=sim_state[5]
        )
        self.vessel_state_pub.publish(sim_state_msg)

        # Publish lidar
        lidar_messurments = self.last_observation.flatten()[6:]
        lidar_msg = Float32MultiArray(
            data=lidar_messurments.tolist()
        )
        self.lidar_pub.publish(lidar_msg)


    def thruster_input_callback(self, msg: BlueboatActuatorInputs):
        self.get_logger().debug(f"got thruster msg: {msg.stb_prop_force, msg.port_prop_force}")

        self.action[0] = msg.stb_prop_force
        self.action[1] = msg.port_prop_force


def main(args=None):
    rclpy.init(args=args)
    simulator_node = SimulationNode()
    rclpy.spin(simulator_node)

    simulator_node.destroy_node()
    rclpy.shutdown()
   




