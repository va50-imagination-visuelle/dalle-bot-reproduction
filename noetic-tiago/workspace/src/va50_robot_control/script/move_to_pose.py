#!/usr/bin/env python

import sys
import rospy
import moveit_commander


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_to_pose")

    group = moveit_commander.MoveGroupCommander("arm")
    group.set_planning_time(10.0)

    gripper = moveit_commander.MoveGroupCommander("gripper")
    rospy.loginfo("Repère : %s", group.get_planning_frame())
    rospy.loginfo("Effecteur : %s", group.get_end_effector_link())
    # Fermer avec le gripper
    gripper.go([0.01, 0.01], wait=True)
    gripper.stop()

    # Petite cible de test : 2 cm au-dessus de la pose actuelle.
    target = group.get_current_pose().pose
    #target.position.z += 0.02
    target.position.x -=0.02
    target.position.y +=0.1

    group.set_pose_target(target)
    success = group.go(wait=True)

    group.stop()
    group.clear_pose_targets()
    if success:
    	gripper.go([0.044, 0.044], wait=True)
    	gripper.stop()

    moveit_commander.roscpp_shutdown()
    moveit_commander.roscpp_shutdown()

    if not success:
        rospy.logerr("MoveIt n'a pas pu exécuter le mouvement.")
        raise SystemExit(1)

    rospy.loginfo("Mouvement terminé.")


if __name__ == "__main__":
    main()
