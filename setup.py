from rich.traceback import install
from setuptools import setup, find_packages

package_name = "gym_asv_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "gymnasium==1.0.0",
        "pyglet==2.1.2",
        "pynput==1.7.7",
        "shapely==2.0.7",
        "stable-baselines3==3.4.1",
    ],
    # install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hurodor",
    maintainer_email="hurodor@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "manual_control_node = gym_asv_ros2.ros_nodes.manual_control_node:main",
                "simulator_node = gym_asv_ros2.ros_nodes.simulator_node:main"
        ],
    },
)
