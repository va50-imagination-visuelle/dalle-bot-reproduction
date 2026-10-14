#!/usr/bin/env python
"""Plan (but never execute) a vertical pre-grasp above the fork handle."""

import math
import os
import sys
import time

import moveit_commander
import rospy
import rospkg
import tf
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import (GetPositionFK, GetPositionFKRequest,
                             GetPositionIK, GetPositionIKRequest)
from tf.transformations import inverse_matrix, quaternion_from_matrix, quaternion_matrix


PLANNING_GROUP = "arm_torso"
PLANNING_FRAME = "base_footprint"
EEF_LINK = "arm_tool_link"
GRASP_FRAME = "gripper_grasping_frame"

# Fork collision box dimensions from models/fork/model.sdf (metres).
FORK_SIZE = (0.0197, 0.190, 0.0117)
# Bounding boxes measured from table_dining/meshes/table2.dae. They approximate
# the tabletop and four legs while keeping the scene lightweight for OMPL.
TABLE_TOP_SIZE = (0.942, 0.471, 0.020)
TABLE_TOP_CENTER_Z = 0.329
TABLE_LEG_SIZE = (0.050, 0.050, 0.652)
TABLE_LEG_CENTER_Z = -0.007
TABLE_LEG_CENTERS_XY = ((-0.435, -0.205), (-0.435, 0.215),
                        (0.455, -0.205), (0.455, 0.215))

# In the fork mesh, the four tines are on local +Y; grasp the handle at -Y.
HANDLE_OFFSET_LOCAL = (0.0, -0.060, 0.0)
# At 5 cm, MoveIt's finger collision boxes touched the tabletop. 9 cm cleared
# the table in state-validity checks while retaining a reachable IK solution.
PREGRASP_CLEARANCE = 0.090

POSITION_TOLERANCE = 0.015  # m
ORIENTATION_TOLERANCE = 0.050  # rad
MOVEIT_ORIENTATION_COMPONENT_TOLERANCE = 0.020  # rad per constrained axis
PLANNING_TIME = 10.0
VELOCITY_SCALING = 0.15
ACCELERATION_SCALING = 0.15

TABLE_OBJECT_ID = "tiago_pregrasp_table_top"
FORK_OBJECT_ID = "tiago_pregrasp_fork"
TABLE_LEG_OBJECT_IDS = tuple("tiago_pregrasp_table_leg_{}".format(i)
                             for i in range(1, 5))
LEGACY_OBJECT_IDS = ("table_top_pregrasp", "fork_collision_pregrasp")


def pose_error(actual, target):
    dx = actual.position.x - target.position.x
    dy = actual.position.y - target.position.y
    dz = actual.position.z - target.position.z
    position = math.sqrt(dx * dx + dy * dy + dz * dz)
    actual_q = (actual.orientation.x, actual.orientation.y,
                actual.orientation.z, actual.orientation.w)
    target_q = (target.orientation.x, target.orientation.y,
                target.orientation.z, target.orientation.w)
    dot = min(1.0, abs(sum(a * b for a, b in zip(actual_q, target_q))))
    return position, 2.0 * math.acos(dot)


def validate_ik(target, avoid_collisions):
    rospy.wait_for_service("/compute_ik", timeout=10.0)
    request = GetPositionIKRequest()
    request.ik_request.group_name = PLANNING_GROUP
    request.ik_request.ik_link_name = EEF_LINK
    request.ik_request.pose_stamped = target
    request.ik_request.avoid_collisions = avoid_collisions
    request.ik_request.timeout = rospy.Duration(3.0)
    response = rospy.ServiceProxy("/compute_ik", GetPositionIK)(request)
    return response.error_code.val


def verify_fork_stayed_put(service, original_pose):
    current_pose = get_gazebo_model(service, "fork")
    position, orientation = pose_error(current_pose, original_pose)
    if position > 0.005 or orientation > 0.020:
        raise RuntimeError(
            "Fork moved while planning (%.3f m, %.3f rad); discard this plan "
            "and re-run with its current pose." % (position, orientation))


def get_gazebo_model(service, name):
    result = service(name, PLANNING_FRAME)
    if not result.success:
        raise RuntimeError("Gazebo could not return pose for model '{}': {}".format(
            name, result.status_message))
    return result.pose


def wait_for_scene_objects(scene, expected, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not rospy.is_shutdown():
        known = set(scene.get_known_object_names())
        if set(expected).issubset(known):
            return
        rospy.sleep(0.1)
    raise RuntimeError("MoveIt did not register collision objects: {}".format(
        ", ".join(expected)))


def add_collision_objects(scene, table_pose, fork_pose):
    # Remove only this script's own IDs so rerunning it refreshes live poses.
    for object_id in ((TABLE_OBJECT_ID, FORK_OBJECT_ID) +
                      TABLE_LEG_OBJECT_IDS + LEGACY_OBJECT_IDS):
        scene.remove_world_object(object_id)

    table = PoseStamped()
    table.header.frame_id = PLANNING_FRAME
    table.pose.position.x = table_pose.position.x
    table.pose.position.y = table_pose.position.y
    table.pose.position.z = table_pose.position.z + TABLE_TOP_CENTER_Z
    table.pose.orientation = table_pose.orientation
    scene.add_box(TABLE_OBJECT_ID, table, size=TABLE_TOP_SIZE)

    table_q = (table_pose.orientation.x, table_pose.orientation.y,
               table_pose.orientation.z, table_pose.orientation.w)
    table_rotation = quaternion_matrix(table_q)[:3, :3]
    for object_id, (local_x, local_y) in zip(TABLE_LEG_OBJECT_IDS,
                                             TABLE_LEG_CENTERS_XY):
        offset = table_rotation.dot((local_x, local_y, TABLE_LEG_CENTER_Z))
        leg = PoseStamped()
        leg.header.frame_id = PLANNING_FRAME
        leg.pose.position.x = table_pose.position.x + offset[0]
        leg.pose.position.y = table_pose.position.y + offset[1]
        leg.pose.position.z = table_pose.position.z + offset[2]
        leg.pose.orientation = table_pose.orientation
        scene.add_box(object_id, leg, size=TABLE_LEG_SIZE)

    fork = PoseStamped()
    fork.header.frame_id = PLANNING_FRAME
    fork.pose = fork_pose
    scene.add_box(FORK_OBJECT_ID, fork, size=FORK_SIZE)
    wait_for_scene_objects(
        scene, (TABLE_OBJECT_ID, FORK_OBJECT_ID) + TABLE_LEG_OBJECT_IDS)


def make_pregrasp_target(fork_pose, tool_to_grasp):
    fork_q = (fork_pose.orientation.x, fork_pose.orientation.y,
              fork_pose.orientation.z, fork_pose.orientation.w)
    fork_rotation = quaternion_matrix(fork_q)[:3, :3]

    # Locate the grasp point along the fork's local handle axis.
    handle_offset = fork_rotation.dot(HANDLE_OFFSET_LOCAL)
    target_position = (
        fork_pose.position.x + handle_offset[0],
        fork_pose.position.y + handle_offset[1],
        fork_pose.position.z + FORK_SIZE[2] / 2.0 + PREGRASP_CLEARANCE,
    )

    # grasp_frame +X follows the fingers toward the object (downward for a
    # top-down approach); +Y is the jaw-closing direction across the fork.
    closing_axis = fork_rotation.dot((1.0, 0.0, 0.0))
    approach_axis = (0.0, 0.0, -1.0)
    closing_axis = (closing_axis[0], closing_axis[1], 0.0)
    third_axis = (
        approach_axis[1] * closing_axis[2] - approach_axis[2] * closing_axis[1],
        approach_axis[2] * closing_axis[0] - approach_axis[0] * closing_axis[2],
        approach_axis[0] * closing_axis[1] - approach_axis[1] * closing_axis[0],
    )
    grasp_rotation = [
        [approach_axis[0], closing_axis[0], third_axis[0]],
        [approach_axis[1], closing_axis[1], third_axis[1]],
        [approach_axis[2], closing_axis[2], third_axis[2]],
    ]

    base_to_grasp = quaternion_matrix(quaternion_from_matrix(
        _homogeneous(grasp_rotation, target_position)))
    base_to_grasp[:3, 3] = target_position
    base_to_tool = base_to_grasp.dot(inverse_matrix(tool_to_grasp))

    target = PoseStamped()
    target.header.frame_id = PLANNING_FRAME
    target.header.stamp = rospy.Time.now()
    target.pose.position.x = base_to_tool[0, 3]
    target.pose.position.y = base_to_tool[1, 3]
    target.pose.position.z = base_to_tool[2, 3]
    q = quaternion_from_matrix(base_to_tool)
    (target.pose.orientation.x, target.pose.orientation.y,
     target.pose.orientation.z, target.pose.orientation.w) = q
    return target, target_position


def _homogeneous(rotation, translation):
    matrix = [[rotation[row][column] for column in range(3)] + [translation[row]]
              for row in range(3)]
    matrix.append([0.0, 0.0, 0.0, 1.0])
    return matrix


def verify_robot_description_resources():
    # MoveIt can start even when package:// collision meshes cannot be resolved.
    # In that case collision checking silently loses important robot geometry.
    resources = {
        "tiago_description": ("meshes/arm/arm_3_collision.dae",),
        "pal_gripper_description": (
            "meshes/gripper_link_collision.stl",
            "meshes/gripper_finger_link.stl",
        ),
    }
    package_index = rospkg.RosPack()
    missing = []
    for package_name, files in resources.items():
        try:
            package_path = package_index.get_path(package_name)
        except rospkg.ResourceNotFound:
            missing.append(package_name)
            continue
        missing.extend(os.path.join(package_name, relative)
                       for relative in files
                       if not os.path.isfile(os.path.join(package_path, relative)))
    if missing:
        raise RuntimeError(
            "Robot collision description is incomplete; missing ROS resources: {}. "
            "Source /opt/pal/gallium/setup.bash before planning.".format(
                ", ".join(missing)))


def verify_plan_endpoint(robot, trajectory, target):
    if not trajectory.joint_trajectory.points:
        raise RuntimeError("MoveIt returned an empty joint trajectory")

    final_state = robot.get_current_state()
    positions_by_name = dict(zip(final_state.joint_state.name,
                                 final_state.joint_state.position))
    final_point = trajectory.joint_trajectory.points[-1]
    positions_by_name.update(zip(trajectory.joint_trajectory.joint_names,
                                 final_point.positions))
    final_state.joint_state.position = [
        positions_by_name[name] for name in final_state.joint_state.name
    ]

    request = GetPositionFKRequest()
    request.header.frame_id = PLANNING_FRAME
    request.fk_link_names = [EEF_LINK]
    request.robot_state = final_state
    rospy.wait_for_service("/compute_fk", timeout=10.0)
    response = rospy.ServiceProxy("/compute_fk", GetPositionFK)(request)
    if response.error_code.val != 1 or not response.pose_stamped:
        raise RuntimeError("MoveIt FK could not verify the final planned pose "
                           "(error {})".format(response.error_code.val))

    position, orientation = pose_error(response.pose_stamped[0].pose, target.pose)
    rospy.loginfo("Planned endpoint error: position %.4f m, orientation %.4f rad",
                  position, orientation)
    if position > POSITION_TOLERANCE or orientation > ORIENTATION_TOLERANCE:
        raise RuntimeError("Planned endpoint exceeds the configured tolerances")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("plan_fork_pregrasp")

    try:
        verify_robot_description_resources()
        rospy.wait_for_service("/gazebo/get_model_state", timeout=15.0)
        get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        fork_pose = get_gazebo_model(get_model_state, "fork")
        table_pose = get_gazebo_model(get_model_state, "table_dining")

        listener = tf.TransformListener()
        listener.waitForTransform(EEF_LINK, GRASP_FRAME, rospy.Time(0),
                                  rospy.Duration(5.0))
        translation, rotation = listener.lookupTransform(
            EEF_LINK, GRASP_FRAME, rospy.Time(0))
        tool_to_grasp = quaternion_matrix(rotation)
        tool_to_grasp[:3, 3] = translation

        scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        add_collision_objects(scene, table_pose, fork_pose)

        group = moveit_commander.MoveGroupCommander(PLANNING_GROUP)
        robot = moveit_commander.RobotCommander()
        group.set_pose_reference_frame(PLANNING_FRAME)
        group.set_planning_time(PLANNING_TIME)
        group.set_num_planning_attempts(10)
        group.set_goal_position_tolerance(POSITION_TOLERANCE)
        # MoveIt constrains orientation per component; the final acceptance
        # below uses a quaternion-angle error, so use tighter axis tolerances.
        group.set_goal_orientation_tolerance(
            MOVEIT_ORIENTATION_COMPONENT_TOLERANCE)
        group.set_max_velocity_scaling_factor(VELOCITY_SCALING)
        group.set_max_acceleration_scaling_factor(ACCELERATION_SCALING)

        target, grasp_position = make_pregrasp_target(fork_pose, tool_to_grasp)
        rospy.loginfo("Planning group: %s; planning frame: %s; EEF: %s",
                      PLANNING_GROUP, group.get_planning_frame(), EEF_LINK)
        rospy.loginfo("Fork handle pre-grasp point in %s: (%.4f, %.4f, %.4f) m",
                      PLANNING_FRAME, *grasp_position)
        rospy.loginfo("Clearance above fork collision box: %.3f m",
                      PREGRASP_CLEARANCE)
        rospy.logwarn("Table and fork are represented by conservative box proxies. "
                      "Review the path in RViz before execution.")

        ik_without_collision = validate_ik(target, avoid_collisions=False)
        if ik_without_collision != 1:
            raise RuntimeError(
                "Pre-grasp is not kinematically reachable by {} (IK error {}). "
                "No trajectory will be planned or executed.".format(
                    PLANNING_GROUP, ik_without_collision))
        ik_with_collision = validate_ik(target, avoid_collisions=True)
        if ik_with_collision != 1:
            raise RuntimeError(
                "The target IK solution collides with the robot or planning scene "
                "(IK error {}).".format(ik_with_collision))

        group.set_pose_target(target, EEF_LINK)
        result = group.plan()
        if isinstance(result, tuple):
            planned, trajectory = result[0], result[1]
            planning_time = result[2] if len(result) > 2 else 0.0
            error_code = result[3] if len(result) > 3 else None
        else:
            trajectory = result
            planned = bool(trajectory.joint_trajectory.points)
            planning_time = 0.0
            error_code = None
        group.clear_pose_targets()

        if not planned or not trajectory.joint_trajectory.points:
            code = error_code.val if error_code is not None else "unknown"
            raise RuntimeError("No path found (MoveIt error {}, {:.3f} s)".format(
                code, planning_time))

        verify_fork_stayed_put(get_model_state, fork_pose)
        verify_plan_endpoint(robot, trajectory, target)

        display_publisher = rospy.Publisher(
            "/move_group/display_planned_path", DisplayTrajectory,
            queue_size=1, latch=True)
        rospy.sleep(0.5)
        display = DisplayTrajectory()
        display.model_id = "tiago"
        display.trajectory_start = robot.get_current_state()
        display.trajectory.append(trajectory)
        display_publisher.publish(display)

        duration = trajectory.joint_trajectory.points[-1].time_from_start.to_sec()
        rospy.loginfo("Plan valid: %d trajectory points, %.2f s; planning took %.3f s.",
                      len(trajectory.joint_trajectory.points), duration, planning_time)
        rospy.loginfo("Plan published on /move_group/display_planned_path. "
                      "This script does NOT execute the trajectory.")
        rospy.sleep(1.0)
    except Exception as error:
        rospy.logerr("Pre-grasp planning stopped: %s", error)
        raise SystemExit(1)
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
