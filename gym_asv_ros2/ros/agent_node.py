from gym_asv_ros2.gym_asv.environment import BaseEnvironment, RandomGoalWithDockObstacle
# from gym_asv_ros2.ros import simulator_node
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool, String
from microamp_interfaces.msg import ThrusterInputs, BoatState, RlLogMessage, Waypoint
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy

from rclpy.logging import LoggingSeverity

from gym_asv_ros2.ros.ros_helpers import RosVessel, RosLidar

import pickle, base64

# from gym_asv_ros2.gym_asv.entities import BaseEntity


import numpy as np
from stable_baselines3 import PPO
# from typing import Sequence

# from threading import Lock
# from gym_asv_ros2.gym_asv.vessel import Vessel



class AgentNode(Node):

    def __init__(self) -> None:
        super().__init__("gym_asv_agent_node")
    
        self.logger = self.get_logger()
        self.logger.set_level(LoggingSeverity.DEBUG)

        ## Get paramters

        # Agent file
        self.declare_parameter("agent", "")
        agent_file = self.get_parameter("agent").get_parameter_value().string_value

        # N perception
        self.declare_parameter("n_perception", 0)
        n_perception_features = self.get_parameter("n_perception").get_parameter_value().integer_value

        self.declare_parameter("simulated_lidar", True)
        simulated_lidar = self.get_parameter("simulated_lidar").get_parameter_value().bool_value

        self.declare_parameter("env_sim_level", 0)
        self.env_sim_level = self.get_parameter("env_sim_level").get_parameter_value().integer_value

        # Prosentage of max trhust to use
        self.declare_parameter("thrust_cap", 0.2)
        self.thrust_cap = self.get_parameter("thrust_cap").get_parameter_value().double_value


        # Dump all paramters
        self.logger.info(f"""
        Loading paramters:
            agent_file: {agent_file}
            n_perception: {n_perception_features}
            simulated_lidar: {simulated_lidar}
            env_sim_level: {self.env_sim_level}
            thrust_cap: {self.thrust_cap}\
        """
        )

        # Check if we have an valid agent file
        if not agent_file:
            raise FileExistsError(f"{agent_file} does not exists.")

        ## Subscription/ publishers

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # Start stop signal
        self.operation_state_sub = self.create_subscription(
            Bool,
            "/start_operation",
            self.operation_state_sub_callback,
            1
        )
        # To publish STOP signal when algorithm is finished
        self.operation_state_pub = self.create_publisher(
            Bool,
            "/start_operation",
            1
        )

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

        self.obstacle_pub = self.create_publisher(
            String,
            "gym_asv_ros2/internal/virtual_obstacles",
            1,
        )

        ## Variables

        self.run_state = False # Set the state of the controller

        # Rl related
        self.agent = PPO.load(agent_file)
        self.real_env = BaseEnvironment(render_mode=None, n_perception_features=n_perception_features)

        self.real_env.vessel = RosVessel(np.zeros(6,), 1, 1)

        if not simulated_lidar:
            self.logger.info(f"Using real lidar with range {30.0} and {n_perception_features} rays")
            self.real_env.lidar_sensor = RosLidar(30.0, n_perception_features) # Overrides the simulated Lidar
        self.simulated_lidar = simulated_lidar

        self.reached_goal_timer_iteration = 0

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

        self.logger.info(f"Node Initialized. Operation state is: {self.run_state}")


    def lidar_sub_callback(self, msg: LaserScan):
        """Recive raw 2d lidarscan, and downscale it to match the agents observation space."""

        if not isinstance(self.real_env.lidar_sensor, RosLidar):
            return

        # self.real_env.lidar_sensor.index_interpolate_scan(msg)
        self.real_env.lidar_sensor.min_pooling_scan(msg)
        scan = self.real_env.lidar_sensor.sense()

        self.logger.debug(f"{scan}\nlen: {len(scan)}")
        
        # scan = self.real_env.lidar_sensor.process_lidar_scan(msg)
        # self.logger.info(f"Got {len(scan)} scan, \n{scan}")
    
    def operation_state_sub_callback(self, msg: Bool):
        """Toogle signal to start/stop the RL-Controller
            if True, controller is allowed to run,
            if False, controller is not allowed to run
        """

        self.run_state = msg.data
        self.logger.info(f"Recived operation state: {self.run_state}.")

        # Set init state for the agent as current state.
        if self.run_state:
            self.real_env.vessel._init_state = self.real_env.vessel._state
            self.real_env.reset()
            self.logger.info("Resting Environment")
        

    def _stop_opertaion(self):
        """Force stop operation."""

        self.run_state = False

        msg = Bool()
        msg.data = self.run_state

        self.logger.debug(f"Sending {self.run_state} signal on /start_operation")
        self.operation_state_pub.publish(msg)

    def waypoint_callback(self, msg: Waypoint):
        """Recives a waypoint, and update the agents goal to this waypoint."""

        # Save the waypoint
        self.last_waypoint_msg = msg

        # Set the waypoint
        self.logger.info(f"Recived waypoint. x: {msg.xn}, y: {msg.yn}, psi: {msg.psi_n}")
        self.real_env.goal.position = np.array([msg.xn, msg.yn])
        self.real_env.goal.angle = msg.psi_n

        # Set init state for the agent as current state.
        self.real_env.vessel._init_state = self.real_env.vessel._state
        self.real_env.reset()

        # TODO: set up simulated obstacles here
        self.sim_object_hack_setup()


    def sim_object_hack_setup(self):
        """Create virtual obstacles, if not using simulated lidar the obstacles are ignored."""


        if self.env_sim_level == 0:
            return
        
        if not self.simulated_lidar:
            return

        self.helper_env = RandomGoalWithDockObstacle(render_mode=None)
        self.helper_env.obstacles.clear()
        self.helper_env.vessel._state = self.real_env.vessel._state

        self.logger.info(f"Using sim_lvl for virtual obstacles: {self.env_sim_level}")

        # set goal in helper_env to get correct obstacle positions
        self.helper_env.goal = self.real_env.goal

        if self.env_sim_level == 2:
            self.helper_env.level2(False)

        elif self.env_sim_level == 3:
            self.helper_env.level3(False)

        elif self.env_sim_level == 23:
            self.helper_env.level2_n_3(False)

        else:
            self.helper_env.level1(False)

        self.real_env.obstacles = self.helper_env.obstacles

        # Publish the obstacle information

        pickled_obst_list = [base64.b64encode(pickle.dumps(obst)).decode("ascii") for obst in self.real_env.obstacles]
        obst_string = "||".join(pickled_obst_list)

        obst_string_msg = String()
        obst_string_msg.data = obst_string
        self.obstacle_pub.publish(obst_string_msg)
        self.logger.debug(f"Using virtual obstacles: {obst_string}")
        # # self.logger.info(f"Simulating obstacles at (pos, vertecies): {[[ obst.position, obst.boundary ] for obst in self.real_env.obstacles]}")

    def publish_log_data(self, last_reward, reached_goal, collision, observation):
        """Publish log data."""

        log_msg = RlLogMessage()
        log_msg.boat_state = self.last_vessel_state_msg
        log_msg.thrust_input = self.last_thrust_msg
        log_msg.reward = last_reward
        log_msg.target_waypoint = self.last_waypoint_msg
        log_msg.operational_state = self.run_state
        log_msg.reached_goal = reached_goal
        log_msg.collision = collision
        log_msg.observation = observation

        log_msg.header.stamp = self.get_clock().now().to_msg()

        self.log_pub.publish(log_msg)

    def state_sub_callback(self, msg: BoatState):
        """Make the navigation part of the observation when we can state update."""

        # Update vessel state from msg
        self.real_env.vessel.set_state(msg)

        self.last_time_state_recived = self.get_clock().now()

        # Save state msg
        self.last_vessel_state_msg = msg


    def pub_action_to_pwm(self, action_stb: float, action_port: float):
        """Convert action (-1, 1) to pwm signal (1100, 1900) and publish it to the thrusters."""

        pwm_zero = 1500

        pwm_max = 1900
        pwm_min = 1100

        pwm_high = 400 * self.thrust_cap + pwm_zero
        pwm_low = pwm_zero - 400 * self.thrust_cap
        
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
        thruster_msg.header.stamp = self.get_clock().now().to_msg()
        # Save thrust msg
        self.last_thrust_msg = thruster_msg
        self.action_pub.publish(thruster_msg)

        
    def run(self):
        """Main loop of the controller. """

        if not self.run_state:
            self.pub_action_to_pwm(0.0, 0.0)
            return

        # Check if have not gotten state data, TODO: add check for lidar data aswell
        time_now = self.get_clock().now()
        if time_now > self.last_time_state_recived + self.wait_for_data_duration:
            self.pub_action_to_pwm(0.0, 0.0)
            self.get_logger().info("Did not recive sensor data, setting 0 thrust")
            return

        dummy_action = np.array([0.0, 0.0])
        observation, reward, done, truncated, info = self.real_env.step(dummy_action)

        # observation = self.real_env.observe()
        action, _states = self.agent.predict(observation, deterministic=True)
        # action, _states = self.agent.predict(observation, deterministic=False)

        # TODO: check if these values makes sense
        reached_goal = bool(info["reached_goal"])
        collision = bool(info["collision"])
        observation = info["observation"].tolist()

        # self.get_logger().info(f"state is: {self.real_env.vessel._state}, action: {action}")

        # TODO: Figure out what to do when reaching the goal.
        if done:
            if reached_goal:
                self.logger.info(f"Reached goal at, {self.real_env.goal.position}")
                self._stop_opertaion()
                # self.real_env.reset()

            elif collision:
                self.logger.info("Collision detected")
                self._stop_opertaion()

            # self.run_state = False
            # if self.reached_goal_timer_iteration >= 50:
                # self.run_state = False
                # self.reached_goal_timer_iteration = 0
            # self.reached_goal_timer_iteration += 1

        self.pub_action_to_pwm(
            action[0],
            action[1]
        )

        # Publish log data
        self.publish_log_data(reward, reached_goal, collision, observation)


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()










