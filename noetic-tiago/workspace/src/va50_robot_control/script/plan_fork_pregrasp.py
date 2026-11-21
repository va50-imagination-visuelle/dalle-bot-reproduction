#!/usr/bin/env python
"""Plan a vertical fork approach or run a simulated pick and place."""

import argparse
import copy
import math
import os
import sys
import time

import moveit_commander
import rospy
import rospkg
import tf
import tf2_geometry_msgs  # Registers geometry_msgs conversions with tf2.
import tf2_ros
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import (GetPositionFK, GetPositionFKRequest,
                             GetPositionIK, GetPositionIKRequest)
from sensor_msgs.msg import JointState
from tf.transformations import inverse_matrix, quaternion_from_matrix, quaternion_matrix
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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
# Approach high enough for the long finger boxes to clear the table during
# lateral motion, then descend vertically until their tips surround the handle.
PREGRASP_CLEARANCE = 0.130
GRASP_CLEARANCE = 0.090
GRASP_DESCENT = PREGRASP_CLEARANCE - GRASP_CLEARANCE
LIFT_DISTANCE = 0.080
PLACE_RELEASE_CLEARANCE = 0.010
PLACE_POSITION_TOLERANCE = 0.020  # m
PLACE_ORIENTATION_TOLERANCE = 0.15  # rad

GRIPPER_JOINTS = ("gripper_left_finger_joint",
                  "gripper_right_finger_joint")
GRIPPER_OPEN_POSITION = 0.045
GRIPPER_CLOSED_POSITION = 0.0
GRIPPER_COMMAND_TIME = 2.0
GRIPPER_TOUCH_LINKS = ("arm_tool_link", "gripper_link",
                       "gripper_left_finger_link",
                       "gripper_right_finger_link")

POSITION_TOLERANCE = 0.015  # m
MOVEIT_POSITION_TOLERANCE = 0.005  # m
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan a fork pre-grasp or execute a simulated pick and place.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the validated trajectory in addition to publishing it",
    )
    parser.add_argument(
        "--grasp",
        action="store_true",
        help="execute pre-grasp, descend, close the fingers and test-lift the fork",
    )
    parser.add_argument(
        "--pick-place",
        action="store_true",
        help="grasp, lift and place the fork at an offset on the same table",
    )
    parser.add_argument(
        "--place-offset", nargs=2, type=float, metavar=("DX", "DY"),
        help="fork displacement in base_footprint in metres (default: 0 0.08)",
    )
    parser.add_argument(
        "--pose-source",
        choices=("gazebo", "perception"),
        default="gazebo",
        help="source of the fork pose (default: gazebo)",
    )
    parser.add_argument(
        "--fork-pose-topic",
        default="/perception/fork/pose",
        help="geometry_msgs/PoseStamped topic used with --pose-source perception",
    )
    parser.add_argument(
        "--perception-timeout",
        type=float,
        default=10.0,
        help="maximum wait for a perceived pose and its TF transform, in seconds",
    )
    parser.add_argument(
        "--max-pose-age",
        type=float,
        default=1.0,
        help="maximum accepted age of a non-zero perception timestamp, in seconds",
    )
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if args.place_offset is not None and not args.pick_place:
        parser.error("--place-offset requires --pick-place")
    if args.pick_place and args.pose_source != "gazebo":
        parser.error("--pick-place requires --pose-source gazebo")
    return args


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
            "Fork moved (%.3f m, %.3f rad); discard this plan "
            "and re-run with its current pose." % (position, orientation))


def get_gazebo_model(service, name):
    result = service(name, PLANNING_FRAME)
    if not result.success:
        raise RuntimeError("Gazebo could not return pose for model '{}': {}".format(
            name, result.status_message))
    return result.pose


def normalized_pose(pose_stamped):
    position = pose_stamped.pose.position
    quaternion = pose_stamped.pose.orientation
    values = (position.x, position.y, position.z,
              quaternion.x, quaternion.y, quaternion.z, quaternion.w)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Perception pose contains a non-finite value")

    norm = math.sqrt(quaternion.x * quaternion.x +
                     quaternion.y * quaternion.y +
                     quaternion.z * quaternion.z +
                     quaternion.w * quaternion.w)
    if norm < 1e-6:
        raise RuntimeError("Perception pose contains a null quaternion")
    quaternion.x /= norm
    quaternion.y /= norm
    quaternion.z /= norm
    quaternion.w /= norm
    return pose_stamped


def get_perceived_fork_pose(tf_buffer, topic, timeout, max_age):
    rospy.loginfo("Waiting up to %.1f s for fork pose on %s", timeout, topic)
    perceived = rospy.wait_for_message(
        topic, PoseStamped, timeout=timeout)
    if not perceived.header.frame_id:
        raise RuntimeError("Perception pose has no frame_id")
    perceived = normalized_pose(perceived)

    if perceived.header.stamp != rospy.Time(0):
        age = (rospy.Time.now() - perceived.header.stamp).to_sec()
        if age < -0.1:
            raise RuntimeError(
                "Perception pose timestamp is {:.3f} s in the future".format(-age))
        if age > max_age:
            raise RuntimeError(
                "Perception pose is stale ({:.3f} s > {:.3f} s)".format(
                    age, max_age))

    try:
        transformed = tf_buffer.transform(
            perceived, PLANNING_FRAME, rospy.Duration(timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as error:
        raise RuntimeError(
            "Cannot transform perceived fork pose from {} to {}: {}".format(
                perceived.header.frame_id, PLANNING_FRAME, error))
    transformed = normalized_pose(transformed)
    rospy.loginfo(
        "Perceived fork pose transformed %s -> %s: "
        "position=(%.4f, %.4f, %.4f)",
        perceived.header.frame_id, PLANNING_FRAME,
        transformed.pose.position.x, transformed.pose.position.y,
        transformed.pose.position.z)
    return transformed.pose


def wait_for_scene_objects(scene, expected, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not rospy.is_shutdown():
        known = set(scene.get_known_object_names())
        if set(expected).issubset(known):
            return
        rospy.sleep(0.1)
    raise RuntimeError("MoveIt did not register collision objects: {}".format(
        ", ".join(expected)))


def wait_for_scene_object_removed(scene, object_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not rospy.is_shutdown():
        if object_id not in scene.get_known_object_names():
            return
        rospy.sleep(0.1)
    raise RuntimeError("MoveIt did not remove collision object: {}".format(
        object_id))


def wait_for_attached_object(scene, object_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not rospy.is_shutdown():
        if object_id in scene.get_attached_objects([object_id]):
            return
        rospy.sleep(0.1)
    raise RuntimeError("MoveIt did not attach collision object: {}".format(
        object_id))


def command_gripper(publisher, position):
    deadline = time.time() + 5.0
    while publisher.get_num_connections() == 0 and time.time() < deadline:
        rospy.sleep(0.1)
    if publisher.get_num_connections() == 0:
        raise RuntimeError("No subscriber on /gripper_controller/command")

    command = JointTrajectory()
    command.header.stamp = rospy.Time.now()
    command.joint_names = list(GRIPPER_JOINTS)
    point = JointTrajectoryPoint()
    point.positions = [position, position]
    point.time_from_start = rospy.Duration(GRIPPER_COMMAND_TIME)
    command.points = [point]
    publisher.publish(command)
    rospy.sleep(GRIPPER_COMMAND_TIME + 0.5)

    state = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
    positions = []
    efforts = []
    for joint in GRIPPER_JOINTS:
        index = state.name.index(joint)
        positions.append(state.position[index])
        efforts.append(state.effort[index])
    rospy.loginfo(
        "Gripper command %.3f m; measured positions=(%.4f, %.4f) m; "
        "efforts=(%.3f, %.3f)", position, positions[0], positions[1],
        efforts[0], efforts[1])
    return positions, efforts


def execute_cartesian(group, target_pose, label):
    trajectory, fraction = group.compute_cartesian_path(
        [target_pose], 0.005, 0.0, avoid_collisions=True)
    rospy.loginfo("%s Cartesian path fraction: %.3f", label, fraction)
    if fraction < 0.999 or not trajectory.joint_trajectory.points:
        raise RuntimeError(
            "{} Cartesian path is incomplete ({:.1f}%)".format(
                label, 100.0 * fraction))
    if not group.execute(trajectory, wait=True):
        group.stop()
        raise RuntimeError("MoveIt failed to execute {} path".format(label))
    group.stop()
    return trajectory


def make_place_target(fork_pose, table_pose, offset):
    dx, dy = offset
    if not all(math.isfinite(value) for value in offset):
        raise RuntimeError("Place offset must be finite")
    if math.hypot(dx, dy) <= 2 * PLACE_POSITION_TOLERANCE:
        raise RuntimeError("Place offset must exceed 4 cm to verify relocation")

    target = copy.deepcopy(fork_pose)
    target.position.x += dx
    target.position.y += dy
    # Keep the entire fork footprint above the tabletop, with room to settle.
    table_inverse = inverse_matrix(pose_matrix(table_pose))
    target_in_table = table_inverse.dot(pose_matrix(target))
    for x in (-FORK_SIZE[0] / 2, FORK_SIZE[0] / 2):
        for y in (-FORK_SIZE[1] / 2, FORK_SIZE[1] / 2):
            corner = target_in_table.dot((x, y, 0, 1))
            if any(abs(corner[i]) + 0.02 > TABLE_TOP_SIZE[i] / 2
                   for i in (0, 1)):
                raise RuntimeError("Place target is too close to the table edge")
    return target


def pose_matrix(pose):
    matrix = quaternion_matrix((pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w))
    matrix[:3, 3] = (pose.position.x, pose.position.y, pose.position.z)
    return matrix


def place_fork(group, scene, gripper, get_model_state, target, lifted_fork):
    lifted_tool = group.get_current_pose(EEF_LINK).pose
    dx = target.position.x - lifted_fork.position.x
    dy = target.position.y - lifted_fork.position.y
    rospy.loginfo("Fork target in %s: (%.4f, %.4f, %.4f) m",
                  PLANNING_FRAME, target.position.x, target.position.y,
                  target.position.z)

    # Translate the measured tool pose by the desired fork displacement.
    # Preserve the grasp orientation and the clearance achieved by the lift.
    above_target = copy.deepcopy(lifted_tool)
    above_target.position.x += dx
    above_target.position.y += dy
    execute_cartesian(group, above_target, "Place transfer")

    carried_fork = get_gazebo_model(get_model_state, "fork")
    expected_fork = copy.deepcopy(lifted_fork)
    expected_fork.position.x = target.position.x
    expected_fork.position.y = target.position.y
    position, orientation = pose_error(carried_fork, expected_fork)
    if position > PLACE_POSITION_TOLERANCE or orientation > PLACE_ORIENTATION_TOLERANCE:
        raise RuntimeError("Fork did not follow the gripper during transfer "
                           "({:.4f} m, {:.4f} rad)".format(position, orientation))

    release = group.get_current_pose(EEF_LINK).pose
    release.position.z += (target.position.z + PLACE_RELEASE_CLEARANCE
                           - carried_fork.position.z)
    execute_cartesian(group, release, "Place descent")
    positions, _ = command_gripper(gripper, GRIPPER_OPEN_POSITION)
    if any(abs(value - GRIPPER_OPEN_POSITION) > 0.003 for value in positions):
        raise RuntimeError("Gripper did not open at the place target")

    scene.remove_attached_object(EEF_LINK, name=FORK_OBJECT_ID)
    # Detaching returns the proxy to the world at the old carried pose.
    scene.remove_world_object(FORK_OBJECT_ID)
    wait_for_scene_object_removed(scene, FORK_OBJECT_ID)
    rospy.sleep(0.5)
    placed = PoseStamped()
    placed.header.frame_id = PLANNING_FRAME
    placed.pose = get_gazebo_model(get_model_state, "fork")
    scene.add_box(FORK_OBJECT_ID, placed, size=FORK_SIZE)
    wait_for_scene_objects(scene, (FORK_OBJECT_ID,))
    execute_cartesian(group, above_target, "Place retreat")

    settled = get_gazebo_model(get_model_state, "fork")
    rospy.sleep(2.0)
    verify_fork_stayed_put(get_model_state, settled)
    final = get_gazebo_model(get_model_state, "fork")
    position, orientation = pose_error(final, target)
    rospy.loginfo("Fork place error: position %.4f m, orientation %.4f rad",
                  position, orientation)
    if position > PLACE_POSITION_TOLERANCE or orientation > PLACE_ORIENTATION_TOLERANCE:
        raise RuntimeError("Fork settled outside the place tolerances")
    placed.pose = final
    scene.add_box(FORK_OBJECT_ID, placed, size=FORK_SIZE)
    rospy.loginfo("Fork pick and place succeeded.")


def add_collision_objects(scene, table_pose, fork_pose):
    # Remove only this script's own IDs so rerunning it refreshes live poses.
    scene.remove_attached_object(EEF_LINK, name=FORK_OBJECT_ID)
    rospy.sleep(0.2)
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
    args = parse_args()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("plan_fork_pregrasp")

    try:
        verify_robot_description_resources()
        rospy.wait_for_service("/gazebo/get_model_state", timeout=15.0)
        get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        fork_gazebo_pose = get_gazebo_model(get_model_state, "fork")
        table_pose = get_gazebo_model(get_model_state, "table_dining")
        place_target = None
        if args.pick_place:
            place_target = make_place_target(
                fork_gazebo_pose, table_pose, args.place_offset or (0.0, 0.08))
        grasp = args.grasp or args.pick_place

        if args.pose_source == "perception":
            tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
            tf2_ros.TransformListener(tf_buffer)
            fork_pose = get_perceived_fork_pose(
                tf_buffer, args.fork_pose_topic, args.perception_timeout,
                args.max_pose_age)
        else:
            fork_pose = fork_gazebo_pose
            rospy.loginfo("Using Gazebo ground truth for the fork pose")

        listener = tf.TransformListener()
        listener.waitForTransform(EEF_LINK, GRASP_FRAME, rospy.Time(0),
                                  rospy.Duration(5.0))
        translation, rotation = listener.lookupTransform(
            EEF_LINK, GRASP_FRAME, rospy.Time(0))
        tool_to_grasp = quaternion_matrix(rotation)
        tool_to_grasp[:3, 3] = translation

        scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        add_collision_objects(scene, table_pose, fork_pose)

        gripper_publisher = rospy.Publisher(
            "/gripper_controller/command", JointTrajectory, queue_size=1)
        if grasp:
            command_gripper(gripper_publisher, GRIPPER_OPEN_POSITION)

        group = moveit_commander.MoveGroupCommander(PLANNING_GROUP)
        robot = moveit_commander.RobotCommander()
        group.set_pose_reference_frame(PLANNING_FRAME)
        group.set_planning_time(PLANNING_TIME)
        group.set_num_planning_attempts(10)
        group.set_goal_position_tolerance(MOVEIT_POSITION_TOLERANCE)
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

        verify_fork_stayed_put(get_model_state, fork_gazebo_pose)
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
        rospy.loginfo("Plan published on /move_group/display_planned_path.")

        if args.execute or grasp:
            rospy.logwarn("Executing the validated pre-grasp trajectory.")
            executed = group.execute(trajectory, wait=True)
            group.stop()
            if not executed:
                raise RuntimeError(
                    "MoveIt did not execute the pre-grasp trajectory successfully")

            rospy.sleep(0.5)
            actual = group.get_current_pose(EEF_LINK).pose
            position, orientation = pose_error(actual, target.pose)
            rospy.loginfo(
                "Executed endpoint error: position %.4f m, orientation %.4f rad",
                position, orientation)
            if (position > POSITION_TOLERANCE or
                    orientation > ORIENTATION_TOLERANCE):
                raise RuntimeError(
                    "Executed endpoint exceeds the configured tolerances")
            verify_fork_stayed_put(get_model_state, fork_pose)
            rospy.loginfo("Pre-grasp trajectory executed successfully.")

            if grasp:
                # Contact with the fork is intentional during the descent, so
                # remove only its proxy while retaining the table obstacles.
                scene.remove_world_object(FORK_OBJECT_ID)
                wait_for_scene_object_removed(scene, FORK_OBJECT_ID)

                if GRASP_DESCENT > 0.0:
                    grasp_target = copy.deepcopy(target.pose)
                    grasp_target.position.z -= GRASP_DESCENT
                    execute_cartesian(group, grasp_target, "Grasp descent")
                else:
                    rospy.loginfo(
                        "Finger tips are already at handle height; "
                        "skipping descent.")

                command_gripper(
                    gripper_publisher, GRIPPER_CLOSED_POSITION)

                # Reinsert the live fork pose and attach it to the tool in the
                # MoveIt scene. Gazebo contact physics remains responsible for
                # whether the real simulated fork follows during the lift.
                fork_before_lift = get_gazebo_model(get_model_state, "fork")
                fork_collision = PoseStamped()
                fork_collision.header.frame_id = PLANNING_FRAME
                fork_collision.pose = fork_before_lift
                scene.add_box(FORK_OBJECT_ID, fork_collision, size=FORK_SIZE)
                wait_for_scene_objects(scene, (FORK_OBJECT_ID,))
                scene.attach_box(
                    EEF_LINK, FORK_OBJECT_ID,
                    touch_links=list(GRIPPER_TOUCH_LINKS))
                wait_for_attached_object(scene, FORK_OBJECT_ID)

                lift_target = copy.deepcopy(
                    group.get_current_pose(EEF_LINK).pose)
                lift_target.position.z += LIFT_DISTANCE
                execute_cartesian(group, lift_target, "Grasp lift")
                rospy.sleep(0.5)

                fork_after_lift = get_gazebo_model(get_model_state, "fork")
                fork_lift = (fork_after_lift.position.z -
                             fork_before_lift.position.z)
                rospy.loginfo("Fork vertical displacement after lift: %.4f m",
                              fork_lift)
                if fork_lift < LIFT_DISTANCE * 0.5:
                    raise RuntimeError(
                        "Physical grasp failed: fork lifted only {:.4f} m".format(
                            fork_lift))
                rospy.loginfo("Fork grasp and test lift succeeded.")
                if place_target is not None:
                    place_fork(group, scene, gripper_publisher, get_model_state,
                               place_target, fork_after_lift)
        else:
            rospy.loginfo(
                "Planning-only mode: pass --execute to move the robot.")
            rospy.sleep(1.0)
    except Exception as error:
        rospy.logerr("Pre-grasp planning stopped: %s", error)
        raise SystemExit(1)
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
