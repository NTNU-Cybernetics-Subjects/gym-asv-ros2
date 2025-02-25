
# Gym asv ros2

## Install

```
git clone https://github.com/NTNU-Cybernetics-Subjects/gym-asv-ros2.git
cd gym_asv_ros2
pip install -r requirments.txt
```

## Usage without ROS

```
python -m gym_asv_ros2.run $MODE
```
where `$MODE` is train/enjoy

- `--logid` will set the name of the logfolder to use.
- `--agent` will set the agent to use (To be used in combination with enjoy)
