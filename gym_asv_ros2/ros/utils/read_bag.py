#!/usr/bin/env python3
# extract_string_rosbag2.py
import csv
import json
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
# from std_msgs.msg import String
import rosbag2_py
from rosidl_runtime_py.convert import message_to_ordereddict


def read_messages(input_bag: str, topic_filter: list[str], format: str="mcap"):
    """Read a ros2 bag, and output each msg.
        @input_bag: the path of the bag
        @topic_filter: the topics to output msg on
        @format: the format of the rosbag"""

    # Open reader
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id=format),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        ),
    )

    topics = reader.get_all_topics_and_types()
    def typename(name):
        for t in topics:
            if t.name == name:
                return t.type
        raise ValueError(f"Topic {name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in topic_filter:
            continue
    
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="Path to MCAP bag file")
    parser.add_argument("topic", help="Topic to extract (e.g. /chatter)")
    parser.add_argument("-o", "--out", help="Output file", required=False, default="output.csv")
    parser.add_argument("--format", choices=["csv", "json"], required=False, default="json")
    args = parser.parse_args()

    output_filename = args.out
    filname_format = (lambda f: f[-1] if len(f) > 1 else None)(output_filename.split("."))
    output_format = filname_format if filname_format else args.format

    if output_format == "json":
    
        # JSON output
        out = []
        for topic, msg, ts in read_messages(args.bag, [ args.topic ]):

            msg_dict = message_to_ordereddict(msg)
            record = {
                "topic": topic,
                "timestamp": ts,
                "data": msg_dict
            }
            out.append(record)

        with open(output_filename, "w") as jf:
            json.dump(out, jf, indent=2)

    # elif output_format == "csv":
    #     # CSV output
    #     with open(args.out, "w", newline="") as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(["timestamp_ns", "data"])
    #         for _, msg, ts in read_messages(args.bag, args.topic):
    #             writer.writerow([ts, msg.data])

