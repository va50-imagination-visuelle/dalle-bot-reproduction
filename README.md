# DALL-E-Bot reproduction

Reproduction of the pipeline described in *DALL-E-Bot: Introducing Web-Scale
Diffusion Models to Robotics*, with a ROS Noetic/TIAGo simulation environment.

## Project layout

```text
configs/       Reproducible pipeline and experiment configurations
src/           ROS-independent Python implementation of the pipeline
ros_ws/src/    ROS packages connected to the TIAGo simulation
scripts/       Command-line entry points and utility scripts
tests/         Unit, integration and ROS tests
data/          Local inputs and small versioned samples
artifacts/     Generated run outputs and evaluation results
docs/          Scientific and technical documentation
noetic-tiago/  Local Docker simulation bundle (not versioned here)
```

The existing `noetic-tiago` Docker bundle remains the reference simulation
environment. Project-owned ROS packages will live in `ros_ws/src` and will be
mounted or copied into its Catkin workspace during development.
