#!/usr/bin/env python3
import importlib
import rclpy
from rclpy.node import Node
import roslibpy

# ------------------------------------------------------------------
# User configuration: list the topics you want to bridge and their
# full ROS message types (pkg/msg/MsgName).
# ------------------------------------------------------------------
TOPICS_TO_BRIDGE = [
    # ('/chatter', 'std_msgs/msg/String'),
    # ('/pose', 'geometry_msgs/msg/PoseStamped'),
    # ('/microampere/sensors/nav_pvt', 'microamp_interfaces/msg/GNSSNavPvt'),
    # ('/microampere/battery', "sensor_msgs/msg/BatteryState"),
    ("/gym_asv_ros2/internal/log_data", "microamp_interfaces/msg/RlLogMessage"),
    ("/gym_asv_ros2/internal/virtual_obstacles", "std_msgs/msg/String"),
    ("/gym_asv_ros2/internal/waypoint", "microamp_interfaces/msg/Waypoint"),
    ("/microampere/control/thrust", "microamp_interfaces/msg/ThrusterInputs"),
    ("/microampere/state_est/pos_vel_kalman", "microamp_interfaces/msg/BoatState")
    # Add your topics here
]

class SelectedTopicsBridge(Node):
    def __init__(self, remote_host='localhost', remote_port=9090):
        super().__init__('selected_topics_bridge')
        self.get_logger().info(f"Initializing bridge to {remote_host}:{remote_port}")

        # Connect to rosbridge server
        self.rosbridge = roslibpy.Ros(host=remote_host, port=remote_port)
        self.rosbridge.run()
        self.get_logger().info("Connected to rosbridge server")

        # Set up publishers and subscriptions for each topic
        for topic_name, type_str in TOPICS_TO_BRIDGE:
            self._setup_bridge(topic_name, type_str)

    def _setup_bridge(self, topic_name, type_str):
        # Parse package and message class
        if '/msg/' in type_str:
            pkg, msg = type_str.split('/msg/')
        else:
            pkg, msg = type_str.split('/', 1)

        try:
            ros_module = importlib.import_module(f"{pkg}.msg")
            msg_class = getattr(ros_module, msg)
        except (ImportError, AttributeError) as e:
            self.get_logger().error(f"Failed to import {type_str}: {e}")
            return

        # Local ROS2 publisher
        publisher = self.create_publisher(msg_class, topic_name, 10)
        self.get_logger().info(f"Created publisher for {topic_name} [{type_str}]")

        # Remote subscription via rosbridge
        remote_topic = roslibpy.Topic(self.rosbridge, topic_name, type_str)
        remote_topic.subscribe(
            lambda msg, p=publisher, c=msg_class: self._forward(msg, p, c)
        )
        self.get_logger().info(f"Subscribed to remote {topic_name}")

    def _forward(self, msg_dict, publisher, cls):
        # Convert incoming dict to ROS2 message, handling nested messages
        ros_msg = cls()
        self._fill_message(ros_msg, msg_dict)
        publisher.publish(ros_msg)

    def _fill_message(self, ros_msg, msg_dict):
        for field, value in msg_dict.items():
            try:
                if isinstance(value, dict):
                    # Nested message
                    sub_msg = getattr(ros_msg, field)
                    self._fill_message(sub_msg, value)
                elif value is None:
                    # Skip unset fields
                    continue
                else:
                    # Primitive field; let ROS message classes validate
                    setattr(ros_msg, field, value)
            except Exception as e:
                # Log and skip invalid fields
                self.get_logger().warn(f"Skipping field '{field}' due to error: {e}")


def main():
    rclpy.init()
    node = SelectedTopicsBridge(
        remote_host='192.168.2.14',  # update if using SSH tunnel
        remote_port=9090
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.rosbridge.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
