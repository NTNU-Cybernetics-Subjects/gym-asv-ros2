# from gym_asv_ros2.obstacles import CircularObstacle
# from gym_asv_ros2.vessel import Vessel
# from gym_asv_ros2.visualization import Visualizer, BG_PMG_PATH
import rclpy
from rclpy.node import Node
from microamp_interfaces.msg import BlueboatActuatorInputs
# from blueboat_interfaces import blueboat_actuator_inputs
import numpy as np
# from gym_asv_ros2.simulator import Game

class ManualControlNode(Node):
    
    def __init__(self) -> None:
        super().__init__("gym_asv_manual_controll_node")

        self.publisher = self.create_publisher(
            BlueboatActuatorInputs,
            "/thruster_input",
            10
        )

        self.action = [0.0,0.0]

        self.get_logger().info("manual_control_node going into listening mode.")
        self.start()


    def start(self):
        while True:
            continue_loop = self.input_loop()
            self.publish_action()
            if not continue_loop:
                pass # does not work to break beacause we are inside rclpy.spin()
                # rclpy.shutdown()

    def input_loop(self):
        print(self.action, flush=True)
        i = input(": ")
        if "u" in i:
            self.action[0] = 0.5

        if "i" in i:
            self.action[1] = 0.5

        if "j" in i:
            self.action[0] = -0.5

        if "k" in i:
            self.action[1] = -0.5

        if len(i) <= 0:
            self.action = [0.0, 0.0]

        if "q" in i:
            return False

        return True

    def publish_action(self):
        # action = self.controller.action
        msg = BlueboatActuatorInputs()
        # msg.header.stamp = self.get_clock()
        msg.stb_prop_force = self.action[0]
        msg.port_prop_force = self.action[1]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    manual_control_node = ManualControlNode()
    rclpy.spin(manual_control_node)

    manual_control_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()




