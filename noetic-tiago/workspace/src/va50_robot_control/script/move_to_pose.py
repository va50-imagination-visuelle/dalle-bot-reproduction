#!/usr/bin/env python

import math
import sys

import moveit_commander
import rospy
from geometry_msgs.msg import PoseStamped


# Pose fixe de test, relevée dans la simulation de ce projet. Les coordonnées
# sont en mètres et le quaternion suit l'ordre x, y, z, w.
TARGET_POSITION = (0.5611, -0.0640, 0.85775)
TARGET_ORIENTATION = (-0.492542, 0.494445, 0.478248, 0.533110)
POSITION_TOLERANCE = 0.015  # m
ORIENTATION_TOLERANCE = 0.05  # rad


def pose_errors(actual, target):
    position_error = math.sqrt(
        (actual.position.x - target.position.x) ** 2
        + (actual.position.y - target.position.y) ** 2
        + (actual.position.z - target.position.z) ** 2
    )

    actual_q = (
        actual.orientation.x,
        actual.orientation.y,
        actual.orientation.z,
        actual.orientation.w,
    )
    target_q = (
        target.orientation.x,
        target.orientation.y,
        target.orientation.z,
        target.orientation.w,
    )
    dot = min(1.0, abs(sum(a * b for a, b in zip(actual_q, target_q))))
    orientation_error = 2.0 * math.acos(dot)
    return position_error, orientation_error


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_to_pose")

    group = moveit_commander.MoveGroupCommander("arm")
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(10)
    group.set_goal_position_tolerance(POSITION_TOLERANCE)
    group.set_goal_orientation_tolerance(ORIENTATION_TOLERANCE)

    target = PoseStamped()
    target.header.frame_id = "base_footprint"
    target.header.stamp = rospy.Time.now()
    target.pose.position.x, target.pose.position.y, target.pose.position.z = TARGET_POSITION
    (
        target.pose.orientation.x,
        target.pose.orientation.y,
        target.pose.orientation.z,
        target.pose.orientation.w,
    ) = TARGET_ORIENTATION

    rospy.loginfo("Repère de planification : %s", group.get_planning_frame())
    rospy.loginfo("Effecteur : %s", group.get_end_effector_link())
    rospy.loginfo("Cible (m) : %s", TARGET_POSITION)
    group.set_pose_target(target, "arm_tool_link")

    planned, trajectory, planning_time, error_code = group.plan()
    if not planned or not trajectory.joint_trajectory.points:
        group.clear_pose_targets()
        group.stop()
        moveit_commander.roscpp_shutdown()
        rospy.logerr(
            "Échec de planification (MoveIt error code %s, %.3f s).",
            error_code.val,
            planning_time,
        )
        raise SystemExit(1)

    rospy.loginfo("Plan trouvé en %.3f s; exécution...", planning_time)
    executed = group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()

    if not executed:
        moveit_commander.roscpp_shutdown()
        rospy.logerr("Le contrôleur n'a pas terminé la trajectoire avec succès.")
        raise SystemExit(2)

    rospy.sleep(0.5)
    actual = group.get_current_pose("arm_tool_link").pose
    position_error, orientation_error = pose_errors(actual, target.pose)
    rospy.loginfo(
        "Erreurs finales : position %.4f m, orientation %.4f rad",
        position_error,
        orientation_error,
    )
    moveit_commander.roscpp_shutdown()

    if (
        position_error > POSITION_TOLERANCE
        or orientation_error > ORIENTATION_TOLERANCE
    ):
        rospy.logerr("La pose finale dépasse la tolérance définie.")
        raise SystemExit(3)

    rospy.loginfo("Cible atteinte dans les tolérances définies.")


if __name__ == "__main__":
    main()
