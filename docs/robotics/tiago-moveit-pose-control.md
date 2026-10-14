# Commander le bras et le gripper du TIAGo avec MoveIt (ROS Noetic)

Cette note documente la commande de test de l’issue 35. Elle concerne le TIAGo simulé dans Gazebo avec `move_group` lancé par `tiago_gazebo.launch`.

## Quel groupe déplacer ?

- `arm` est le groupe MoveIt à utiliser pour déplacer le bras jusqu’à une pose de pince. Son effecteur observé dans cette configuration est `arm_tool_link`.
- `gripper` commande séparément l’ouverture des doigts (articulations `gripper_left_finger_joint` et `gripper_right_finger_joint`). Il ne sert pas à déplacer la pince dans l’espace.
- `arm_torso` permet également de planifier en incluant le mouvement du torse, si celui-ci est nécessaire. Le test courant utilise `arm` et garde donc le torse fixe.

Les valeurs observées à l’exécution sont le repère de planification `base_footprint` et l’effecteur `arm_tool_link`. Pour éviter toute ambiguïté, un objectif fixe doit être envoyé avec un `PoseStamped` dont le `header.frame_id` indique le repère de la cible.

MoveIt (`move_group`) effectue la planification, vérifie les collisions et demande l’exécution de la trajectoire aux contrôleurs déjà lancés. Il n’est pas nécessaire de réimplémenter l’IK. Dans cette configuration, le solveur configuré pour le groupe du bras est KDL et le plugin de planification est OMPL.

## Exemple : envoyer une pose cartésienne au bout de la pince

```python
import sys
import rospy
import moveit_commander

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("move_arm_to_pose")

arm = moveit_commander.MoveGroupCommander("arm")
arm.set_planning_time(10.0)

current = arm.get_current_pose()  # PoseStamped, observée en base_footprint
target = current
target.pose.position.z += 0.02  # cible 2 cm au-dessus de la pose actuelle

rospy.loginfo("Repère de planification : %s", arm.get_planning_frame())
rospy.loginfo("Effecteur : %s", arm.get_end_effector_link())
arm.set_pose_target(target, "arm_tool_link")
success = arm.go(wait=True)
arm.stop()
arm.clear_pose_targets()
moveit_commander.roscpp_shutdown()

if not success:
    rospy.logerr("MoveIt n'a pas pu exécuter le mouvement.")
    raise SystemExit(1)
```

La position est exprimée en mètres et l’orientation en quaternion. Pour une cible absolue saisie à la main, construire un `geometry_msgs/PoseStamped`, définir `header.frame_id = "base_footprint"`, renseigner XYZ et le quaternion, puis le passer à `set_pose_target`. Une cible peut être inatteignable ou en collision ; commencer près de la pose actuelle et vérifier le résultat dans Gazebo et RViz.

## Commander les doigts séparément

Dans le modèle local, chaque articulation de doigt a une plage nominale de `0` à `0.045` m. Les tests MoveIt dans la simulation ont rejeté les limites exactes ; utiliser une petite marge intérieure :

```python
gripper = moveit_commander.MoveGroupCommander("gripper")

gripper.go([0.001, 0.001], wait=True)  # presque fermé
gripper.stop()

gripper.go([0.044, 0.044], wait=True)  # presque ouvert
gripper.stop()
```

L’ordre des deux valeurs suit `gripper.get_active_joints()`, vérifiable à l’exécution. Ces commandes contrôlent uniquement les doigts, pas la position XYZ de la pince.

## Procédure de test dans Docker

1. Depuis l’hôte, lancer le bureau VNC puis la simulation :

   ```bash
   cd noetic-tiago
   ./.local-novnc/server.sh
   ./.local-novnc/simulation.sh
   ```

   (`./server.sh` à la racine démarre le conteneur SSH, pas le bureau noVNC.)
2. Ouvrir un deuxième terminal dans le bureau noVNC. Charger les environnements et compiler si nécessaire :

   ```bash
   source /opt/pal/gallium/setup.bash
   source /opt/ros/noetic/setup.bash --extend
   source /home/pal/ros_ws/devel/setup.bash --extend
   ```

3. Vérifier que `move_group`, Gazebo et les contrôleurs tournent :

   ```bash
   rosnode list | grep -E 'move_group|gazebo'
   rostopic list | grep -E 'joint_states|arm_controller|gripper_controller'
   rostopic info /arm_controller/command
   ```

4. Lancer le test depuis un package construit dans le workspace, par exemple `rosrun va50_robot_control move_to_pose.py`.
5. Vérifier que le script retourne `success=True` (ou ne signale pas d’échec), que `/joint_states` évolue et que le bras suit le mouvement dans Gazebo. RViz permet de visualiser l’état, le repère cible et la trajectoire MoveIt.

Le test live du 14 octobre 2026 a réussi : pose initiale de l’effecteur `(0.5611, -0.0640, 0.8378)` m en `base_footprint`, cible `z + 0.02` m, exécution MoveIt `True`, pose finale `(0.5601, -0.0639, 0.8560)` m. Les commandes MoveIt du gripper vers `[0.001, 0.001]` puis `[0.044, 0.044]` ont aussi toutes deux retourné `True`.

Interfaces vérifiées dans le conteneur du projet :

- API de haut niveau : `moveit_commander.MoveGroupCommander("arm")`, méthodes `get_current_pose`, `set_pose_target`, `go(wait=True)`, `stop` et `clear_pose_targets`.
- Action MoveIt : `/move_group` (topics action `/move_group/goal`, `/move_group/status`, `/move_group/result`, etc.; goal de type `moveit_msgs/MoveGroupActionGoal`).
- Actions de contrôleur : `/arm_controller/follow_joint_trajectory` et `/gripper_controller/follow_joint_trajectory` (goal de type `control_msgs/FollowJointTrajectoryActionGoal`). `move_group` utilise ces actions pour exécuter la trajectoire planifiée.
- Commandes bas niveau également présentes : `/arm_controller/command` et `/gripper_controller/command`, type `trajectory_msgs/JointTrajectory`. Pour ce test, préférer l’API MoveIt, qui planifie et vérifie les collisions, plutôt que publier soi-même une trajectoire brute.
- Visualisation de trajectoire : `/move_group/display_planned_path`, type `moveit_msgs/DisplayTrajectory`.
- Services utiles observés : `/plan_kinematic_path`, `/compute_ik`, `/compute_fk`, `/compute_cartesian_path`, `/check_state_validity`, `/get_planning_scene`, `/apply_planning_scene`.

Pour refaire l’inventaire sur une autre version du conteneur : `rosnode info /move_group`, `rostopic list`, `rostopic type <topic>` et `rosservice list`.

## Interfaces et sources

- MoveIt Python API : [`MoveGroupCommander` dans la documentation Noetic](https://docs.ros.org/en/noetic/api/moveit_commander/html/index.html) et [tutoriel Python Move Group](https://moveit.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html).
- Configuration MoveIt de TIAGo maintenue par PAL Robotics : [pal-robotics/tiago_moveit_config](https://github.com/pal-robotics/tiago_moveit_config).
- Guide PAL sur la planification de mouvements TIAGo : [How to plan motions with MoveIt](https://docs.pal-robotics.com/25.01/manipulation/motion-planning-moveit.html).

