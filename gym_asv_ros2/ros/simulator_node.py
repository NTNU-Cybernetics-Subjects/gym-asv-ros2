from gym_asv_ros2.gym_asv.obstacles import CircularObstacle
from gym_asv_ros2.gym_asv.vessel import Vessel
from gym_asv_ros2.gym_asv.visualization import Visualizer
import rclpy
from rclpy.node import Node
# from std_msgs.msg import String
from blueboat_interfaces.msg import BlueboatActuatorInputs
import numpy as np

class SimulationNode(Node):

    def __init__(self):
        super().__init__("gym_asv_sim_node")
        self.subscriber = self.create_subscription(
            BlueboatActuatorInputs,
            "/thruster_input",
            self.thruster_input_callback,
            10
        )

        self.viewer = Visualizer(1000, 1000)
        init_vessel_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.vessel = Vessel(init_vessel_state, 1,1)

        self.obstacles = []
        self.action = np.array([0.0, 0.0])

        self.init_world()

        self.create_timer(0.01, self.timer_callback)

    def init_world(self):

        self.viewer.add_backround(BG_PMG_PATH)
        self.viewer.add_agent(self.vessel.boundary)
        circular_obst = CircularObstacle(np.array([10,10]), 1)
        circular_obst.init_pyglet_shape(self.viewer.pixels_per_unit, self.viewer.batch)
        self.obstacles.append(circular_obst)

    def thruster_input_callback(self, msg: BlueboatActuatorInputs):
        self.get_logger().info(f"got thruster msg: {msg.stb_prop_force, msg.port_prop_force}")
        self.action[0] = msg.stb_prop_force
        self.action[1] = msg.port_prop_force


    def timer_callback(self):
        self.get_logger().info("timer_callback called")
        self.vessel.step(self.action, 0.1)
        self.viewer.update_camerea_position(self.vessel.position)
        self.viewer.update_agent(self.vessel.heading, self.vessel.position)
        self.viewer.update_background()
        for obst in self.obstacles:
            obst.update_pyglet_position(self.viewer.camera_position)
        self.viewer.update_screen()


def main(args=None):
    rclpy.init(args=args)
    simulator_node = SimulationNode()
    rclpy.spin(simulator_node)

    simulator_node.destroy_node()
    rclpy.shutdown()
   




