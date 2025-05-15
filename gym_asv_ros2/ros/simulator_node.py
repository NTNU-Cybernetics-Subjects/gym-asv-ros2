
import shapely
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from microamp_interfaces.msg import ThrusterInputs, BoatState, Waypoint
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

import numpy as np
from gym_asv_ros2.gym_asv.entities import CircularEntity
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer, BG_PMG_PATH
from gym_asv_ros2.gym_asv.environment import BaseEnvironment, RandomGoalBlindEnv, RandomGoalWithDockObstacle

from rclpy.logging import LoggingSeverity

import pickle, base64

def add_a_line_of_sigth_obstalce(vessel_pos, goal_pos, goal_angle):


    los = np.arctan2(goal_pos[2], goal_pos[1])

    distance_to_goal = np.linalg.norm(goal_pos - vessel_pos)


class SimulationNode(Node):

    def __init__(self):
        super().__init__("gym_asv_sim_node")

        self.logger = self.get_logger()
        self.logger.set_level(LoggingSeverity.DEBUG)

        ## Get Paramters
        self.declare_parameter("simulate_vessel", True)
        self.simulate_vessel = self.get_parameter("simulate_vessel").get_parameter_value().bool_value


        # Publish vessel state if simulating, else subscribe to vessel state

        self.action_sub = self.create_subscription(

            ThrusterInputs,
            "/microampere/control/thrust",
            self.thruster_input_callback,
            1
        )


        self.lidar_pub = self.create_publisher(
            LaserScan,
            "/ouster/scan",
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


        # Initialize env
        # self.env = RandomGoalWithDockObstacle(render_mode="human")
        self.env = BaseEnvironment(render_mode="human", n_perception_features=0) # NOTE: Currently not using lidar in sim
        self.env.reset()
        self.env.render()

        # Action
        self.action = np.array([0.0, 0.0])

        simulation_frequency = 0.1
        observation_pub_frequence = 0.01
        self.create_timer(simulation_frequency, self.render_callback)

        if self.simulate_vessel:
            self.vessel_state_pub = self.create_publisher(
                BoatState,
                "/microampere/state_est/pos_vel_kalman",
                1
            )
            self.create_timer(observation_pub_frequence, self.publish_state) # FIXME: Do no set up if we are not simulation
        else:
            self.vessel_state_sub = self.create_subscription(
                BoatState,
                "/microampere/state_est/pos_vel_kalman",
                self.vessel_state_callback,
                1
            )
        self.logger.info(f"Node Initialized. Simulate vessel: {self.simulate_vessel}")


    def __del__(self):
        self.env.close()


    def obstacle_sub_callback(self, msg: String):

        self.env.obstacles.clear()

        obstacle_string = msg.data
        obstacle_encoded_list = obstacle_string.split("||")

        # Unpack the obstacle objects
        for encoded_obst in obstacle_encoded_list:
            pickled_obst = base64.b64decode(encoded_obst.encode("ascii"))
            obstacle = pickle.loads(pickled_obst)

            self.env.add_obstacle(obstacle)

            self.logger.info(f"Adding obstacle: {obstacle}")

            
        # pickeled_list = [  for enrypted_obst in obstacle_encoded_list]
        # obstacles = [pickle.loads(pickeled) for pickeled in pickeled_list]
        #
        # self.env.obstacles = obstacles
        # Init visualization on the obstacles

        # for obst in self.env.obstacles:
        #     obstacle.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
        #     obst.init

        # obstacles = [shapely.wkt.loads(obst) for obst in obstacle_list]
        # self.env.obstacles = obstacles


    def waypoint_callback(self, msg: Waypoint):

        # waypoint = msg.data

        # self.env.reset()
        self.env.goal.position = np.array([msg.xn, msg.yn])
        self.env.goal.angle = msg.psi_n
 

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

        # lidar_messurments = self.env.last_lidar_readings
        # lidar_scan_msg = LaserScan()
        # lidar_scan_msg.angle_min = 0
        # lidar_scan_msg.angle_max = 2*np.pi

        # Publish lidar
        # lidar_messurments = self.last_observation.flatten()[6:]
        # lidar_msg = Float32MultiArray(
        #     data=lidar_messurments.tolist()
        # )
        # self.lidar_pub.publish(lidar_msg)


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

        # self.action[0] = msg.stb_prop_in
        # self.action[1] = msg.port_prop_in


def main(args=None):
    rclpy.init(args=args)
    simulator_node = SimulationNode()
    rclpy.spin(simulator_node)

    simulator_node.destroy_node()
    rclpy.shutdown()

