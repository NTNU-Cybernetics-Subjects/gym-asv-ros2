from gym_asv_ros2.gym_asv.environment import RandomGoalBlindEnv, BaseEnvironment
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Float32MultiArray
from microamp_interfaces.msg import ThrusterInputs, BoatState

import numpy as np
from stable_baselines3 import PPO

# from threading import Lock
from gym_asv_ros2.gym_asv.vessel import Vessel

class RosVessel(Vessel):

    def __init__(self, init_state: np.ndarray, width: float, length: float) -> None:
        super().__init__(init_state, width, length)


    def step(self, action: np.ndarray, h: float):
        pass

        
    def set_state(self, state):
        self._state = state


class AgentNode(Node):

    def __init__(self) -> None:
        super().__init__("gym_asv_agent_node")
    
        self.declare_parameter("agent", "")
        agent_file = self.get_parameter("agent").get_parameter_value().string_value
        self.get_logger().info(f"agent file is: {agent_file}")
        if not agent_file:
            raise FileExistsError(f"{agent_file} does not exists.")

        # Lidar
        # self.lidar_sub = self.create_subscription(
        #     Float32MultiArray,
        #     "/ouster/scan",
        #     self.lidar_sub_callback,
        #     10
        # )
        # self.real_lidar_messurments = np.ones((41,))

        # Vessel
        self.vessel_state_sub = self.create_subscription(
            BoatState,
            "/microampere/state_est/pos_vel_kalman",
            self.state_sub_callback,
            1
        )
        self.real_vessel_state = np.zeros((6,))
        # self.navtigation_features = np.zeros((5,))

        # Action
        self.action_pub = self.create_publisher(
            ThrusterInputs,
            "/microampere/control/thurst",
            1
        )

        # send waypoint to simulator
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray,
            "/gym_asv_ros2/internal/waypoint",
            1
        )

        self.agent = PPO.load(agent_file)

        # Set up the env and override the vessel model
        self.real_env = RandomGoalBlindEnv(render_mode=None)
        self.real_env.vessel = RosVessel(np.zeros(6,), 1, 1)
        self.real_env.reset()
        # self.env_initialzied = False

        self.wait_for_data_duration = Duration(seconds=1)
        self.last_vessel_state_recived = self.get_clock().now() - self.wait_for_data_duration

        # The frequency the controller is running on
        self.run_fequency = 0.01 # NOTE: Should mabye run on 0.2 in real time, due to trained on that step size
        self.create_timer(self.run_fequency, self.run)


    def state_sub_callback(self, msg: BoatState):
        """Make the navigation part of the observation when we can state update."""

        # Update the vessel model to the new states
        vessel_state = np.array([ # TODO: check if we need ot do any transformations
            msg.x,
            msg.y,
            msg.yaw,
            msg.surge,
            msg.sway,
            msg.yaw_r,
        ])

        # self.get_logger().info(f"vessel state: {vessel_state}")
        self.real_env.vessel.set_state(vessel_state)

        self.last_vessel_state_recived = self.get_clock().now()


    def run(self):

        # Check if have not gotten state data
        time_now = self.get_clock().now()
        if time_now > self.last_vessel_state_recived + self.wait_for_data_duration:
            action_msg = ThrusterInputs(
                stb_prop_in=float(0.0),
                port_prop_in=float(0.0)
            )
            self.action_pub.publish(action_msg)
            # self.get_logger().info("Did not recive sensor data, setting 0 thrust")
            return

        dummy_action = np.array([0.0, 0.0])
        observation, reward, done, truncated, info = self.real_env.step(dummy_action)
        print(reward)
        # observation = self.real_env.observe()
        action, _states = self.agent.predict(observation, deterministic=True)

        if done:
            self.get_logger().info(f"Reached goal at, {self.real_env.goal.position}")
            self.real_env.reset()
            self.get_logger().info(f"New goal is at: {self.real_env.goal.position}")
            self.waypoint_pub.publish(
                Float32MultiArray(
                    data=[
                        self.real_env.goal.position[0],
                        self.real_env.goal.position[1],
                        self.real_env.goal.angle
                    ]
                )
            )

        action_msg = ThrusterInputs(
            stb_prop_in=float(action[0]),
            port_prop_in=float(action[1])
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










