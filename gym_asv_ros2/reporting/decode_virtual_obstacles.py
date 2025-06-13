import matplotlib.pyplot as plt
from gym_asv_ros2.gym_asv.entities import *

# from mcap_ros2.reader import read_ros2_messages
import shapely

import base64, pickle
import json

def decode_obstacles(obstacle_code: str) -> list[BaseEntity]:

    obstacle_encoded_list = obstacle_code.split("||")
    # print(obstacle_encoded_list)

    obstacles = []
    # Unpack the obstacle objects
    for encoded_obst in obstacle_encoded_list:
        pickled_obst = base64.b64decode(encoded_obst.encode("ascii"))
        obstacle = pickle.loads(pickled_obst)

        obstacles.append(obstacle)

    return obstacles

def obstacles_to_points(obstacles: list[BaseEntity]) -> list[tuple]:

    all_vertecies = []
    for obj in obstacles:
        # print(obj)
        if isinstance(obj.boundary, shapely.LineString):
            obj_vertecies = list(obj.boundary.coords)
            # print(list( obj.boundary.coords ))
        else:
            obj_vertecies = list(obj.boundary.exterior.coords)
            # print(list(obj.boundary.exterior.coords))
        all_vertecies.append(obj_vertecies)

    return all_vertecies


def read_raw_msg(filename: str):

    with open(filename, "r") as f:
        raw_msg = json.load(f)
        
    return raw_msg

if __name__ == "__main__":

    i = 4
    filename = f"obstacle_msgs/obst{i}.json"
    raw_msgs = read_raw_msg(filename)

    out_list = []
    for msg in raw_msgs:
        obstacles = decode_obstacles(msg["data"])
        timestamp = msg["timestamp_ns"]


        all_vertecies = []
        for obj in obstacles:
            print(obj)
            if isinstance(obj.boundary, shapely.LineString):
                obj_vertecies = list(obj.boundary.coords)
                # print(list( obj.boundary.coords ))
            else:
                obj_vertecies = list(obj.boundary.exterior.coords)
                # print(list(obj.boundary.exterior.coords))
            all_vertecies.append(obj_vertecies)


        processed = {
            "timestamp_ns": timestamp,
            "vertecies": all_vertecies
        }
        out_list.append(processed)

    out_msg = {
        "data": out_list
    }
    # with open(f"processed_obstacle_msgs/obst{i}.json", "w") as f:
    #     json.dump(out_msg, f)

