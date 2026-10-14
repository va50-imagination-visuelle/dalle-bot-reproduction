#!/bin/bash

sudo cp ./src/va50/gripper.urdf.xacro /opt/pal/gallium/share/pal_gripper_description/urdf/
sudo cp ./src/va50/sick_tim571_laser.gazebo.xacro /opt/pal/gallium/share/pmb2_description/urdf/sensors

# Make the project's local Gazebo world and its model:// meshes resolvable.
# Without these paths, Gazebo silently falls back to the similarly named
# world shipped in tiago_gazebo, which does not contain our table setup.
export GAZEBO_RESOURCE_PATH="${PWD}/src/va50:${GAZEBO_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="${PWD}/src/va50/models:${GAZEBO_MODEL_PATH:-}"

# ============================================================
# Start TIAGo simulation
# ============================================================

roslaunch src/tiago_gazebo.launch &
LAUNCH_PID=$!

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping TIAGo simulation..."

    # Send SIGINT (equivalent to Ctrl+C) to roslaunch
    if kill -0 "$LAUNCH_PID" 2>/dev/null; then
        kill -INT "$LAUNCH_PID"
        wait "$LAUNCH_PID" 2>/dev/null
    fi

    echo "TIAGo simulation stopped."
}

# Execute cleanup when script receives Ctrl+C or exits
trap cleanup INT TERM EXIT


echo ""
echo "Waiting for controllers..."

until rostopic list 2>/dev/null | grep -q "^/arm_controller/command$" &&
      rostopic list 2>/dev/null | grep -q "^/torso_controller/command$"; do
    sleep 1
done

echo "Controllers ready!"


# ============================================================
# Move torso
# ============================================================

rostopic pub --once /torso_controller/command trajectory_msgs/JointTrajectory "
joint_names:
- torso_lift_joint
points:
- positions: [0.3084904121192529]
  velocities: []
  accelerations: []
  effort: []
  time_from_start:
    secs: 5
    nsecs: 0"


# ============================================================
# Move arm
# ============================================================

rostopic pub --once /arm_controller/command trajectory_msgs/JointTrajectory "
joint_names:
- arm_1_joint
- arm_2_joint
- arm_3_joint
- arm_4_joint
- arm_5_joint
- arm_6_joint
- arm_7_joint
points:
- positions: [2.0224466632696156, 0.1169263698887244, 1.1384473946763354, 1.847026113795458, 0.07066850738726504, 1.1356681198626797, -1.34258118718481]
  velocities: []
  accelerations: []
  effort: []
  time_from_start:
    secs: 5
    nsecs: 0"


echo ""
echo "Simulation running."
echo "Press Ctrl+C to stop the simulation."

# Keep script alive
wait "$LAUNCH_PID"
