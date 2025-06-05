
import shapely
from gym_asv_ros2.gym_asv.sensors import LidarSimulator
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from microamp_interfaces.msg import ThrusterInputs, BoatState, Waypoint, RlLogMessage
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

import numpy as np
from gym_asv_ros2.gym_asv.entities import CircularEntity
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer, BG_PMG_PATH
from gym_asv_ros2.gym_asv.environment import BaseEnvironment, RandomGoalBlindEnv, RandomGoalWithDockObstacle
from rclpy.qos import QoSProfile, ReliabilityPolicy

from rclpy.logging import LoggingSeverity

from gym_asv_ros2.ros.ros_helpers import RosLidar, RosVessel
# from gym_asv_ros2.ros.agent_node import RosVessel

import pickle, base64


class Logger:

    def __init__(self, out_file) -> None:
        pass


class SimulationNode(Node):

    def __init__(self):
        super().__init__("gym_asv_sim_node")

        self.logger = self.get_logger()
        self.logger.set_level(LoggingSeverity.DEBUG)

        ## Get Paramters
        self.declare_parameter("simulate_vessel", True)
        self.simulate_vessel = self.get_parameter("simulate_vessel").get_parameter_value().bool_value

        self.declare_parameter("simulate_lidar", True)
        self.simulate_lidar = self.get_parameter("simulate_lidar").get_parameter_value().bool_value

        self.declare_parameter("number_lidar_rays", 64)
        n_lidar_rays = self.get_parameter("number_lidar_rays").get_parameter_value().integer_value

        self.declare_parameter("lidar_max_range", 30.0)
        lidar_max_range = self.get_parameter("lidar_max_range").get_parameter_value().double_value

        self.declare_parameter("number_real_lidar_rays", 512)
        self.n_real_lidar_rays = self.get_parameter("number_real_lidar_rays").get_parameter_value().integer_value

        self.logger.info(f"""
            Loading paramters:
                simulate_vessel: {self.simulate_vessel}
                simulate_lidar: {self.simulate_lidar}
                number of lidar rays: {n_lidar_rays}
                real number of lidar rays: {self.n_real_lidar_rays}
                lidar max range: {lidar_max_range}\
        """
        )


        # Publish vessel state if simulating, else subscribe to vessel state

        self.action_sub = self.create_subscription(

            ThrusterInputs,
            "/microampere/control/thrust",
            self.thruster_input_callback,
            1
        )

        # Info from agent
        self.agent_info_sub = self.create_subscription(
            RlLogMessage,
            "/gym_asv_ros2/internal/log_data",
            self.agent_info_sub_callback,
            1
        )

        self.waypoint_sub = self.create_subscription(
            Waypoint,
            "/gym_asv_ros2/internal/waypoint",
            self.waypoint_callback,
            1
        )

        self.obstacle_sub = self.create_subscription(
            String,
            "gym_asv_ros2/internal/virtual_obstacles",
            self.obstacle_sub_callback,
            1
        )


        # self.lidar_pub = self.create_publisher(
        #     LaserScan,
        #     "/ouster/scan",
        #     1
        # )

        # qos_profile = QoSProfile(depth=10)
        # qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        #
        # # Test
        # self.lidar_sub = self.create_subscription(
        #     LaserScan,
        #     "/ouster/scan",
        #     self.lidar_sub_callback,
        #     qos_profile
        # )


        # Initialize env
        self.env = BaseEnvironment(render_mode=None, n_perception_features=0) # NOTE: Do not initialize lidar yet

        # Action
        self.action = np.array([0.0, 0.0])

        simulation_frequency = 0.1
        state_pub_frequency = 0.01
        lidar_pub_frequency = 0.01
        self.create_timer(simulation_frequency, self.render_callback)

        if self.simulate_vessel:
            self.vessel_state_pub = self.create_publisher(
                BoatState,
                "/microampere/state_est/pos_vel_kalman",
                1
            )
            self.create_timer(state_pub_frequency, self.publish_state)
        else:
            # Set up real vessel
            self.vessel_state_sub = self.create_subscription(
                BoatState,
                "/microampere/state_est/pos_vel_kalman",
                self.vessel_state_callback,
                1
            )
            self.env.vessel = RosVessel(np.array([0,0,0,0,0,0]), 1, 1)

        if self.simulate_lidar:
            self.lidar_pub = self.create_publisher(
                LaserScan,
                "/ouster/scan",
                1
            )
            self.env.lidar_sensor = LidarSimulator(lidar_max_range, n_lidar_rays)
            self.create_timer(lidar_pub_frequency, self.publish_lidar)

        else:
            # Set up real lidar
            self.env.lidar_sensor = RosLidar(lidar_max_range, n_lidar_rays)


        # Set up rendering after choosing the Lidar- and vessel modules
        self.env.render_mode = "human"
        self.env.viewer = Visualizer(1000, 1000, headless=False)
        self.env.init_visualization()

        self.env.reset()
        self.env.render()
        self.setup_obs()

        self.logger.info(f"Node Initialized. Simulate vessel: {self.simulate_vessel}")


    def __del__(self):
        self.env.close()

    def setup_obs(self):

        obst = CircularEntity(np.array([10.0,10.0 ]), 1)
        self.env.add_obstacle(obst)


    # def lidar_sub_callback(self, msg: LaserScan):
    #
    #     if not isinstance(self.env.lidar_sensor, RosLidar):
    #         return
    #
    #     self.env.lidar_sensor.min_pooling_scan(msg)
    #
    #     lidar_scan = self.env.lidar_sensor.sense()
    #     lidar_angles = self.env.lidar_sensor.angles
    #
    #     # Update the visuals
    #     start_point = shapely.Point(self.env.vessel.position)
    #
    #     # lidar_x = lidar_scan * np.cos(angles)
    #     # lidar_y = lidar_scan * np.sin(angles)
    #
    #     for i, angle in enumerate(lidar_angles):
    #         # end_point = shapely.Point(lidar_x[i], lidar_y[i])
    #
    #         true_angle = angle + self.env.vessel.heading
    #         end_point = shapely.Point(
    #             start_point.x + lidar_scan[i] * np.cos(true_angle),
    #             start_point.y + lidar_scan[i] * np.sin(true_angle)
    #         )
    #         self.env.lidar_sensor.update_ray_line(i, start_point, end_point)


    def agent_info_sub_callback(self, msg: RlLogMessage):
        """Update visuals according to the log message from the agent"""

        # NOTE: does only support lidar visuals per now, therefore return if simulating lidar
        if self.simulate_lidar:
            return

        observation = msg.observation
        lidar_scan = np.array(observation[5::]) * self.env.lidar_sensor.max_range
        # print(len(lidar_scan))
        angles = self.env.lidar_sensor.angles

        # vessel_position = self.env.vessel.position
        start_point = shapely.Point(self.env.vessel.position)

        # lidar_x = lidar_scan * np.cos(angles)
        # lidar_y = lidar_scan * np.sin(angles)

        for i, angle in enumerate(angles):
            # end_point = shapely.Point(lidar_x[i], lidar_y[i])

            true_angle = angle + self.env.vessel.heading
            end_point = shapely.Point(
                start_point.x + lidar_scan[i] * np.cos(true_angle),
                start_point.y + lidar_scan[i] * np.sin(true_angle)
            )
            self.env.lidar_sensor.update_ray_line(i, start_point, end_point)

        print("observation processed")


    def obstacle_sub_callback(self, msg: String):
        """Recives a configuration of virtual obstacles and adds it to the environment."""

        self.env.obstacles.clear()

        obstacle_string = msg.data
        obstacle_encoded_list = obstacle_string.split("||")

        # Unpack the obstacle objects
        for encoded_obst in obstacle_encoded_list:
            pickled_obst = base64.b64decode(encoded_obst.encode("ascii"))
            obstacle = pickle.loads(pickled_obst)

            self.env.add_obstacle(obstacle)

            self.logger.info(f"Adding obstacle: {obstacle}")


    def waypoint_callback(self, msg: Waypoint):
        """Update the waypoint visuals"""

        # waypoint = msg.data

        # self.env.reset()
        self.env.goal.position = np.array([msg.xn, msg.yn])
        self.env.goal.angle = msg.psi_n
 

    def render_callback(self):
        """Render a frame. """
        
        # Reusing the env.step function since it handels vessel and lidar sim
        observation, reward, done, truncated, info = self.env.step(self.action)
        # self.action[0], self.action[1] = 0.0, 0.0 # Reset action when we have used it
        self.last_observation = observation

        # if done or truncated:
        #     self.env.reset()
        #     self.publish_state()

        self.env.render()
        self.get_logger().info(f"Vessel state is: {self.env.vessel._state}")

    def publish_state(self):
        """Publish vessel state. To be used if simulating the environment."""

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

    def publish_lidar(self):
        """Mimic the real lidar scan"""

        lidar_scan = self.env.lidar_sensor.last_scan

        # Updscale the scan to match the real lidar
        res = int( self.n_real_lidar_rays/self.env.lidar_sensor.num_rays )
        upscaled_scan = np.repeat(lidar_scan[:, np.newaxis], res, axis=1).flatten()
        mimic_real_scan = upscaled_scan[::-1] # real scan is reversed 

        angle_range = np.array([-np.pi, np.pi])
        angle_inc = ( angle_range[1] - angle_range[0] )/self.n_real_lidar_rays

        lidar_scan_msg = LaserScan(
            angle_min=angle_range[0],
            angle_max=angle_range[1],
            angle_increment=angle_inc,
            range_min=0.0,
            range_max=self.env.lidar_sensor.max_range,
            ranges=mimic_real_scan.tolist()
        )
        self.lidar_pub.publish(lidar_scan_msg)

        

    def vessel_state_callback(self, msg: BoatState):
        # Update vessel state from msg
        self.env.vessel.set_state(msg)

    def thruster_input_callback(self, msg: ThrusterInputs):

        self.get_logger().info(f"got thruster msg: {msg.stb_prop_in, msg.port_prop_in}")

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


def main(args=None):
    rclpy.init(args=args)
    simulator_node = SimulationNode()
    rclpy.spin(simulator_node)

    simulator_node.destroy_node()
    rclpy.shutdown()

