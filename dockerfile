FROM osrf/ros:humble-desktop-full

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# ARG USER_ID  = 1000
# ARG GROUP_ID = 1000
# ARG USERNAME = user

RUN apt-get update && apt-get install -y\
    git\
    python3-pip\
    cmake\
    clang\
    bear\
    zsh\
    python3-colcon-common-extensions\
    libcgal-dev\
    libfftw3-dev

# Install requirements
WORKDIR /root

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt &&\
    source /opt/ros/humble/setup.bash && colcon build



