
# Gym asv ros2: A ROS2-based Gymnasium Environment for ASV Simulation, Training and Real-World Deplyment

## Requirements

- Python v3.10.12
- ROS2 Humble 


## Install

```bash
git clone https://github.com/NTNU-Cybernetics-Subjects/gym-asv-ros2.git
cd gym_asv_ros2
pip install -r requirments.txt
```

## Usage without ROS

```bash
python -m gym_asv_ros2.run $MODE
```

where `$MODE` is train/enjoy

- `--logid` will set the name of the logfolder to use.
- `--agent` will set the agent to use (To be used in combination with enjoy)

## Usage with ROS

The ros package is only tested in ROS2 humble. Before launching make sure the
package is built with `colcon build`, and that the environment is sourced
`source install/setup.bash`.

To launch the ROS simulator:

```bash
ros2 run gym_asv_ros2 simulator_node
```

Launch the agent node:

```bash
ros2 run gym_asv_ros2 agent_node --ros-args -p agent:=<agent_path>
```

where `<agent_path>` is the path to where a trained agent is stored.


# Use within the microampere system

Start floxglove and set up floxglove bridge with:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

more information in `Autodocking` repo.

set up static link between ouster-frame and base frame:

```bash
ros2 run tf2_ros static_transform_publisher \
  0 0 0 0 0 0 \
  os_lidar base_link

```

Foxglove will now be connected and the `gym-asv-ros2` module can be used as
described.

## Gym-asv-ros2 playback mode

Start the `simulator-node` in vizualize mode by running:

```
ros2 run gym_asv_ros2 simulator_node --ros-args -p simulate_vessel:=false -p simulate_lidar:=false
```

run a rosbag with:

```
ros2 bag run <$BAG>
```

where `<$BAG>` is the rosbag containing the playback information.




