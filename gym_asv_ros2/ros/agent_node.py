from gym_asv_ros2.gym_asv.environment import BaseEnvironment, RandomGoalWithDockObstacle
from gym_asv_ros2.ros import simulator_node
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Float32MultiArray, Bool
from microamp_interfaces.msg import ThrusterInputs, BoatState, RlLogMessage, Waypoint
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy

# from gym_asv_ros2.gym_asv.entities import BaseEntity


import numpy as np
from stable_baselines3 import PPO
# from typing import Sequence

# from threading import Lock
from gym_asv_ros2.gym_asv.vessel import Vessel

class RosVessel(Vessel):

    def __init__(self, init_state: np.ndarray, width: float, length: float) -> None:
        super().__init__(init_state, width, length)


    def step(self, action: np.ndarray, h: float):
        """Overlads the step function to do nothing. We are not simulating the
        vessel when in real mode"""
        pass

        
    def set_state(self, msg: BoatState):
        """Update the vessel to the recived state."""

        vessel_state = np.array([
            msg.x,
            msg.y ,
            msg.yaw,
            msg.surge,
            msg.sway,
            msg.yaw_r,
        ])

        self._state = vessel_state

class RosLidar():

    def __init__(self, max_range: float, num_rays: int):
        self.max_range = max_range
        self.num_rays = num_rays


        # This is the last proceesed lidar scan
        self.last_lidar_scan = np.full((num_rays,), max_range)


    def index_interpolate_scan(self, msg: LaserScan):

        # min_angle = msg.angle_min
        # max_angle = msg.angle_max
        # angle_increment = msg.angle_increment

        raw_scan = np.array(msg.ranges)
        n_scans = len(raw_scan)
        # shited_raw_scan = np.roll(raw_scan, -int(n_scans/2))

        # reversed_scan = raw_scan[::-1]

        orig_idx = np.arange(n_scans)
        new_idx = np.linspace(0, n_scans -1, self.num_rays)

        reduced_scan = np.interp(new_idx, orig_idx, raw_scan)
        reduced_scan = np.clip(reduced_scan, 0.0, 30.0)

        self.last_lidar_scan = reduced_scan


    def sense(self, *args):
        """Returns the last lidar scan that is proceesed."""
        return self.last_lidar_scan


class AgentNode(Node):

    def __init__(self) -> None:
        super().__init__("gym_asv_agent_node")
    
        self.logger = self.get_logger()

        ## Get paramters

        # Agent file
        self.declare_parameter("agent", "")
        agent_file = self.get_parameter("agent").get_parameter_value().string_value

        # N perception
        self.declare_parameter("n_perception", 0)
        n_perception_features = self.get_parameter("n_perception").get_parameter_value().integer_value

        self.declare_parameter("simulated_lidar", True)
        simulated_lidar = self.get_parameter("simulated_lidar").get_parameter_value().bool_value

        # Dump all paramters
        self.logger.info(f"""
        Loading paramters:
            agent_file: {agent_file}
            n_perception: {n_perception_features}
            simulated_lidar: {simulated_lidar} \
        """
        )

        # Check if we have an valid agent file
        if not agent_file:
            raise FileExistsError(f"{agent_file} does not exists.")

        ## Subscription/ publishers

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # Lidar
        self.lidar_sub = self.create_subscription(
            LaserScan,
            "/ouster/scan",
            self.lidar_sub_callback,
            qos_profile
        )
        # self.real_lidar_messurments = np.ones((41,))

        # Vessel
        self.vessel_state_sub = self.create_subscription(
            BoatState,
            "/microampere/state_est/pos_vel_kalman",
            self.state_sub_callback,
            1
        )

        # Get waypoint
        self.waypoint_sub = self.create_subscription(
            Waypoint,
            "/gym_asv_ros2/internal/waypoint",
            self.waypoint_callback,
            1
        )

        # Action
        self.action_pub = self.create_publisher(
            ThrusterInputs,
            "/microampere/control/thrust",
            1
        )

        # Log topic
        self.log_pub = self.create_publisher(
            RlLogMessage,
            "/gym_asv_ros2/internal/log_data",
            1
        )

        ## Variables

        self.run_state = False # Set the state of the controller

        # Rl related
        self.agent = PPO.load(agent_file)
        self.real_env = BaseEnvironment(render_mode=None, n_perception_features=n_perception_features)
        # self.real_env = RandomGoalWithDockObstacle(render_mode=None, n_perception_features=n_perception_features)
        # self.real_env.init_level = self.real_env.level3
        self.real_env.vessel = RosVessel(np.zeros(6,), 1, 1)
        if not simulated_lidar:
            self.real_env.lidar_sensor = RosLidar(30.0, 41)

        # self.reached_goal_timer_iteration = 0

        # Log data
        self.last_vessel_state_msg = BoatState()
        self.last_waypoint_msg = Waypoint()
        self.last_thrust_msg = ThrusterInputs()

        # timers
        self.wait_for_data_duration = Duration(seconds=1)
        self.last_time_state_recived = self.get_clock().now() - self.wait_for_data_duration

        # The frequency the controller is running on.  NOTE: trained on 0.2
        self.run_fequency = 0.2
        self.create_timer(self.run_fequency, self.run)

        self.logger.info("Node Initialized")


    def lidar_sub_callback(self, msg: LaserScan):

        if not isinstance(self.real_env.lidar_sensor, RosLidar):
            return

        self.real_env.lidar_sensor.index_interpolate_scan(msg)
        scan = self.real_env.lidar_sensor.sense()

        # self.logger.info(f"{scan}\nlen: {len(scan)}")
        
        # scan = self.real_env.lidar_sensor.process_lidar_scan(msg)
        # self.logger.info(f"Got {len(scan)} scan, \n{scan}")

    def waypoint_callback(self, msg: Waypoint):

        # Save the waypoint
        self.last_waypoint_msg = msg

        # Set run_state to true, TODO: set the run_state form the start_autonmous topic
        self.run_state = True
        self.logger.info(f"Run status is set to: {self.run_state}")

        # Set the waypoint
        self.logger.info(f"Recived waypoint. x: {msg.xn}, y: {msg.yn}, psi: {msg.psi_n}")
        self.real_env.goal.position = np.array([msg.xn, msg.yn])
        self.real_env.goal.angle = msg.psi_n

        # Init state for the agent
        self.real_env.vessel._init_state = self.real_env.vessel._state
        self.real_env.reset()

        # self.sim_object_hack_setup()

    def sim_object_hack_setup(self):

        self.helper_env = RandomGoalWithDockObstacle(render_mode=None)
        self.helper_env.level3(False)

        self.real_env.obstacles = self.helper_env.obstacles
        self.logger.info(f"Simulating obstacles at: {[obst.position for obst in self.real_env.obstacles]}")

    def publish_log_data(self, last_reward, reached_goal):

        log_msg = RlLogMessage()
        log_msg.boat_state = self.last_vessel_state_msg
        log_msg.thrust_input = self.last_thrust_msg
        log_msg.reward = last_reward
        log_msg.target_waypoint = self.last_waypoint_msg
        log_msg.operational_state = self.run_state
        log_msg.reached_goal = reached_goal

        self.log_pub.publish(log_msg)


    def state_sub_callback(self, msg: BoatState):
        """Make the navigation part of the observation when we can state update."""

        # Update vessel state from msg
        self.real_env.vessel.set_state(msg)

        self.last_time_state_recived = self.get_clock().now()

        # Save state msg
        self.last_vessel_state_msg = msg


    def pub_action_to_pwm(self, action_stb: float, action_port: float):

        pwm_zero = 1500
        pwm_high = 1700
        pwm_low = 1300
        
        def action_to_pwm(per):

            if per >= 0.0:
                pwm = pwm_zero + per * (pwm_high - pwm_zero)
            else:
                pwm = pwm_zero + per * (pwm_zero - pwm_low)
        
            return pwm

        pwm_stb = action_to_pwm(action_stb)
        pwm_port = action_to_pwm(action_port)

        thruster_msg = ThrusterInputs(
            stb_prop_in = float(pwm_stb),
            port_prop_in = float(pwm_port),
            pwm = True
        )
        # Save thrust msg
        self.last_thrust_msg = thruster_msg
        self.action_pub.publish(thruster_msg)

        
    def run(self):

        if self.run_state == 0:
            self.pub_action_to_pwm(0.0, 0.0)
            return
        
        # Check if have not gotten state data
        time_now = self.get_clock().now()
        if time_now > self.last_time_state_recived + self.wait_for_data_duration:
            self.pub_action_to_pwm(0.0, 0.0)
            self.get_logger().info("Did not recive sensor data, setting 0 thrust")
            return

        dummy_action = np.array([0.0, 0.0])
        observation, reward, done, truncated, info = self.real_env.step(dummy_action)

        # print(reward)
        # observation = self.real_env.observe()
        action, _states = self.agent.predict(observation, deterministic=True)

        # self.get_logger().info(f"state is: {self.real_env.vessel._state}, action: {action}")

        if done:
            if info["reached_goal"]:
                self.logger.info(f"Reached goal at, {self.real_env.goal.position}")

            elif info["collision"]:
                self.logger.info("Collision detected")
                self.run_state = False

            # self.run_state = False
            # if self.reached_goal_timer_iteration >= 50:
                # self.run_state = False
                # self.reached_goal_timer_iteration = 0
            # self.reached_goal_timer_iteration += 1


            # self.real_env.reset()
            # self.get_logger().info(f"New goal is at: {self.real_env.goal.position}")
            # self.waypoint_pub.publish(
            #     Float32MultiArray(
            #         data=[
            #             self.real_env.goal.position[0],
            #             self.real_env.goal.position[1],
            #             self.real_env.goal.angle
            #         ]
            #     )
            # )

        self.pub_action_to_pwm(
            action[0],
            action[1]
        )

        # Publish log data
        self.publish_log_data(reward, done)


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()










