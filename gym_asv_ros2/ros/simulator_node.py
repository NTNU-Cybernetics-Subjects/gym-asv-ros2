
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from microamp_interfaces.msg import ThrusterInputs, BoatState
from std_msgs.msg import Float32MultiArray

import numpy as np
from gym_asv_ros2.gym_asv.entities import CircularEntity
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer, BG_PMG_PATH
from gym_asv_ros2.gym_asv.environment import BaseEnvironment, RandomGoalBlindEnv, RandomGoalWithDockObstacle


class SimulationNode(Node):

    def __init__(self):
        super().__init__("gym_asv_sim_node")
        self.action_sub = self.create_subscription(

            ThrusterInputs,
            "/microampere/control/thrust",
            self.thruster_input_callback,
            1
        )

        self.vessel_state_pub = self.create_publisher(
            BoatState,
            "/microampere/state_est/pos_vel_kalman",
            1
        )

        # self.lidar_pub = self.create_publisher(
        #     Float32MultiArray,
        #     "/ouster/scan",
        #     1
        # )

        self.waypoint_sub = self.create_subscription(
            Float32MultiArray,
            "/gym_asv_ros2/internal/waypoint",
            self.waypoint_callback,
            1
        )

        # Initialize env
        # self.env = RandomGoalWithDockObstacle(render_mode="human")
        self.env = BaseEnvironment(render_mode="human")
        self.env.reset()
        self.env.render()

        # Action
        self.action = np.array([0.0, 0.0])

        simulation_frequency = 0.1
        observation_pub_frequence = 0.01
        self.create_timer(simulation_frequency, self.render_callback)
        self.create_timer(observation_pub_frequence, self.publish_state)

    def __del__(self):
        self.env.close()

    def waypoint_callback(self, msg: Float32MultiArray):
        waypoint = msg.data

        # self.env.reset()
        self.env.goal.position = np.array([waypoint[0], waypoint[1]])
        self.env.goal.angle = waypoint[2]
 

    def render_callback(self):
        
        observation, reward, done, truncated, info = self.env.step(self.action)
        # self.action[0], self.action[1] = 0.0, 0.0 # Reset action when we have used it
        self.last_observation = observation

        # if done or truncated:
        #     self.env.reset()
        #     self.publish_state()

        self.env.render()
        self.get_logger().info(f"Vessel state is: {self.env.vessel._state}")

    def publish_state(self):

        # Publish state
        # sim_state = self.last_observation.flatten()[:6]
        sim_state = self.env.vessel._state
        sim_state_msg = BoatState(
            x=sim_state[0],
            y=sim_state[1],
            yaw=sim_state[2],
            surge=sim_state[3],
            sway=sim_state[4],
            yaw_r=sim_state[5]
        )
        self.vessel_state_pub.publish(sim_state_msg)

        # Publish lidar
        # lidar_messurments = self.last_observation.flatten()[6:]
        # lidar_msg = Float32MultiArray(
        #     data=lidar_messurments.tolist()
        # )
        # self.lidar_pub.publish(lidar_msg)


    def thruster_input_callback(self, msg: ThrusterInputs):

        # self.get_logger().info(f"got thruster msg: {msg.stb_prop_in, msg.port_prop_in}")

        pwm_zero = 1500
        pwm_high = 1900
        pwm_low = 1100

        def pwm_to_action(pwm):
            if pwm >= pwm_zero:
                percentage = (pwm - pwm_zero) / (pwm_high - pwm_zero)
            else: 
                percentage = (pwm - pwm_zero) / (pwm_zero - pwm_low)
            
            return max(min(percentage, 1.0), -1.0)

        self.action[0] = pwm_to_action(msg.stb_prop_in)
        self.action[1] = pwm_to_action(msg.port_prop_in)

        # self.action[0] = msg.stb_prop_in
        # self.action[1] = msg.port_prop_in


def main(args=None):
    rclpy.init(args=args)
    simulator_node = SimulationNode()
    rclpy.spin(simulator_node)

    simulator_node.destroy_node()
    rclpy.shutdown()

