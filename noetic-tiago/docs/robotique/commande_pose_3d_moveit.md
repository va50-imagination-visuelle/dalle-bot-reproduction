# Commander la pose 3D de la pince TIAGo avec MoveIt

Cette procédure décrit l'interface retenue dans la simulation VA50 pour
planifier et exécuter le déplacement de la pince vers une position et une
orientation connues. Elle ne traite ni de vision, ni de conversion 2D vers 3D,
ni de réimplémentation de la cinématique inverse.

## Interface retenue

L'interface applicative est l'API Python `moveit_commander`. Une cible est un
`geometry_msgs/PoseStamped` transmis à un `MoveGroupCommander` :

```python
group = moveit_commander.MoveGroupCommander("arm")
target = PoseStamped()
target.header.frame_id = "base_footprint"
# Renseigner target.pose.position et target.pose.orientation.
group.set_pose_target(target, "arm_tool_link")
planned, trajectory, planning_time, error_code = group.plan()
group.execute(trajectory, wait=True)
```

Le script de référence du projet est
[`workspace/src/va50_robot_control/script/move_to_pose.py`](../../workspace/src/va50_robot_control/script/move_to_pose.py).
Il sépare volontairement `plan()` et `execute()` afin de détecter une
planification vide avant tout mouvement.

## Repères, groupes et effecteur

| Élément | Valeur | Rôle |
|---|---|---|
| Repère de planification | `base_footprint` | Repère dans lequel exprimer la pose cible |
| Groupe MoveIt principal | `arm` | Les sept articulations du bras |
| Groupe alternatif | `arm_torso` | Bras et articulation verticale du torse |
| Lien effecteur | `arm_tool_link` | Lien dont MoveIt commande la pose cartésienne |
| Groupe des doigts | `gripper` | Ouverture et fermeture; ne commande pas une pose cartésienne |

Dans ce projet, « envoyer une pose au gripper » signifie déplacer le bras pour
amener `arm_tool_link` à la pose demandée. L'ouverture et la fermeture des deux
doigts constituent une commande séparée.

Ces valeurs peuvent être contrôlées dans une simulation active :

```bash
python3 - <<'PY'
import moveit_commander
moveit_commander.roscpp_initialize([])
robot = moveit_commander.RobotCommander()
print(robot.get_group_names())
for name in ("arm", "arm_torso"):
    group = moveit_commander.MoveGroupCommander(name)
    print(name, group.get_planning_frame(), group.get_end_effector_link())
PY
```

Le résultat attendu contient `arm`, `arm_torso`, `gripper`, le repère
`base_footprint` et l'effecteur `arm_tool_link`.

## Chaîne de commande

MoveIt calcule la cinématique inverse et la trajectoire, puis transmet les
positions articulaires aux contrôleurs ROS :

| Mouvement | Contrôleur | Interface principale |
|---|---|---|
| Bras | `arm_controller` | `/arm_controller/follow_joint_trajectory` |
| Torse | `torso_controller` | `/torso_controller/follow_joint_trajectory` |
| Doigts | `gripper_controller` | `/gripper_controller/follow_joint_trajectory` |

Les trois contrôleurs sont des `position_controllers/JointTrajectoryController`.
Les interfaces utiles au diagnostic sont :

- `/compute_ik` : service de cinématique inverse ;
- `/compute_fk` : service de cinématique directe ;
- `/plan_kinematic_path` : service de planification MoveIt ;
- `/move_group/status` : état de l'action MoveIt ;
- `/joint_states` : état articulaire mesuré ;
- `/arm_controller/command` : commande articulaire directe, utilisée par le
  script d'initialisation mais non nécessaire avec `moveit_commander`.

Pour vérifier les contrôleurs :

```bash
rosservice call /controller_manager/list_controllers
rostopic list | grep follow_joint_trajectory
```

## Procédure de test dans Gazebo

Dans un premier terminal du conteneur :

```bash
cd /home/pal/ros_ws
source /opt/pal/gallium/setup.bash
catkin_make -DCATKIN_ENABLE_TESTING=OFF -DDISABLE_PAL_FLAGS=ON
source devel/setup.bash --extend
./start_tiago_simulation.sh
```

Attendre le message `Controllers ready!`. Dans un second terminal du même
conteneur :

```bash
cd /home/pal/ros_ws
source /opt/pal/gallium/setup.bash
source devel/setup.bash --extend
python3 src/va50_robot_control/script/move_to_pose.py
```

Le test doit :

1. afficher `base_footprint` comme repère et `arm_tool_link` comme effecteur ;
2. trouver une trajectoire non vide ;
3. exécuter la trajectoire avec `arm_controller` ;
4. mesurer la pose finale ;
5. terminer avec `Cible atteinte dans les tolérances définies.`

Les seuils d'acceptation sont de 15 mm en position et 0,05 rad en orientation.
Les tolérances demandées au planificateur sont volontairement plus strictes
(5 mm et 0,02 rad par axe) afin de conserver une marge pour l'exécution. RViz
permet de contrôler visuellement la trajectoire et Gazebo le mouvement effectif
du robot.

### Résultat de validation

Le test a été rejoué dans Gazebo le 30 octobre 2026 depuis la pose initiale du
projet avec la version installée par Catkin :

```text
Plan trouvé en 0.032 s; exécution...
Erreurs finales : position 0.0053 m, orientation 0.0145 rad
Cible atteinte dans les tolérances définies.
```

La trajectoire a donc été planifiée, exécutée par `arm_controller` et validée
par une nouvelle lecture de la pose de `arm_tool_link`.

## Échecs attendus et diagnostic

Le script retourne un code non nul dans les cas suivants :

- `1` : MoveIt n'a trouvé aucune trajectoire ; vérifier la portée, les
  collisions, le quaternion et la disponibilité de `/move_group` ;
- `2` : le contrôleur n'a pas exécuté la trajectoire ; vérifier
  `arm_controller` et son action `follow_joint_trajectory` ;
- `3` : la pose mesurée dépasse les tolérances ; examiner `/joint_states`, TF
  et la pose obtenue dans RViz.

Toujours normaliser le quaternion cible et vérifier que la cible ne traverse ni
le robot ni la table. Pour une cible utilisant aussi le torse, remplacer le
groupe `arm` par `arm_torso`.

## Références disponibles dans l'environnement

- Le package PAL installé `tiago_moveit_config` contient les configurations
  SRDF, cinématiques, OMPL et contrôleurs utilisées par `/move_group` sous
  `/opt/pal/gallium/share/tiago_moveit_config`.
- `tiago_moveit_config/launch/demo_tiago.launch` et
  `moveit_planning_execution.launch` montrent comment PAL charge MoveIt.
- [`move_to_pose.py`](../../workspace/src/va50_robot_control/script/move_to_pose.py)
  est l'exemple minimal d'exécution du projet.
- [`plan_fork_pregrasp.py`](../../workspace/src/va50_robot_control/script/plan_fork_pregrasp.py)
  est un exemple plus avancé de scène de collision, validation IK/FK et
  publication dans RViz; il ne déclenche volontairement aucun mouvement.
- Le [manuel TIAGo de PAL Robotics](https://docs.pal-robotics.com/tiago-single/handbook.html)
  complète les informations propres au robot.
