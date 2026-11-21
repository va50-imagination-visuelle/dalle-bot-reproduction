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

## Pré-préhension et saisie physique de la fourche

Le script `plan_fork_pregrasp.py` propose quatre modes :

```bash
# Planifier et publier dans RViz, sans mouvement.
rosrun va50_robot_control plan_fork_pregrasp.py

# Exécuter seulement la trajectoire de pré-préhension.
rosrun va50_robot_control plan_fork_pregrasp.py --execute

# Ouvrir la pince, approcher, descendre, fermer puis tester le levage.
rosrun va50_robot_control plan_fork_pregrasp.py --grasp

# Saisir, transporter et déposer la fourche sur la même table.
rosrun va50_robot_control plan_fork_pregrasp.py --pick-place
```

Le mode `--grasp` conserve une approche haute à 13 cm afin que les longs doigts
ne balaient pas la table pendant le mouvement latéral. Il descend ensuite
verticalement de 4 cm, commande les deux doigts vers leur position fermée,
attache la géométrie de la fourche à la scène MoveIt puis remonte de 8 cm. La
réussite est décidée à partir du déplacement réel de la fourche retourné par
Gazebo, et non uniquement à partir de la trajectoire théorique.

Lors du test du 30 octobre 2026, les doigts se sont immobilisés à 7,9 mm et
11,8 mm avec environ 16 N d'effort chacun. La fourche s'est élevée de 41,3 mm
et sa hauteur est restée stable cinq secondes après le mouvement. Cela valide
une saisie par contact physique dans Gazebo, sans plugin d'attachement.

## Premier Pick & Place dans Gazebo

Compiler et lancer la simulation comme dans la procédure ci-dessus. Attendre
le message `Simulation running.` et la fin du mouvement d'initialisation du
bras avant d'exécuter les commandes suivantes dans un second terminal du
conteneur :

```bash
cd /home/pal/ros_ws
source /opt/pal/gallium/setup.bash
source devel/setup.bash --extend
rosrun va50_robot_control plan_fork_pregrasp.py --pose-source gazebo
rosrun va50_robot_control plan_fork_pregrasp.py --pose-source gazebo --pick-place
```

La première commande publie la trajectoire d'approche dans RViz. La deuxième
mesure la pose initiale de la fourche dans Gazebo. La cible de dépôt conserve
sa hauteur et son orientation initiales et la décale de **8 cm sur +Y** dans
`base_footprint`. Pour choisir un autre décalage en mètres, ajouter
`--place-offset DX DY`. Le programme vérifie que le décalage dépasse 4 cm et
que les coins de la boîte de la fourche restent à au moins 2 cm des bords.
Utiliser une zone libre de la table. Le scénario de référence emploie la scène
initiale VA50 et Gazebo fournit les poses connues sans entrée caméra.

Après la saisie et le levage, MoveIt déplace la pince fermée au-dessus de la
cible en conservant l'orientation de prise. Gazebo vérifie que la fourche a
suivi le déplacement. La pince descend, relâche la fourche à environ 1 cm de
sa hauteur de repos, puis se retire. MoveIt détache la boîte de collision de
l'effecteur et la réinsère à la position mesurée. La fourche se déplace par
contact dans Gazebo.

Le succès exige une pose stable après deux secondes et une erreur finale de
**20 mm maximum** en position et **0,15 rad maximum** en orientation. Le
programme renvoie un code non nul si une étape échoue et indique l'étape
concernée. Relancer la scène initiale avant de répéter le scénario de
référence, car la cible se calcule depuis la pose courante de la fourche.

Validation depuis une scène redémarrée, avec le décalage par défaut :

| Mesure | Résultat |
|---|---:|
| Levage physique de la fourche | 52,1 mm |
| Trajectoires cartésiennes de transfert, descente et retrait | 100 % chacune |
| Erreur finale de position | 16,1 mm |
| Erreur finale d'orientation | 0,0018 rad |
| Résultat du script | `Fork pick and place succeeded.` (code 0) |

## Interface entre perception et préhension

Le planificateur accepte deux sources pour la pose de la fourche :

```bash
# Vérité terrain Gazebo, utilisée pour les tests et régressions.
rosrun va50_robot_control plan_fork_pregrasp.py --pose-source gazebo

# Pose estimée par le pipeline de perception.
rosrun va50_robot_control plan_fork_pregrasp.py --pose-source perception
```

Le contrat d'entrée du mode perception est :

| Champ | Contrat |
|---|---|
| Topic par défaut | `/perception/fork/pose` |
| Type ROS | `geometry_msgs/PoseStamped` |
| Position | centre 3D de la fourche |
| `frame_id` | repère connu de TF, par exemple le repère optique de la caméra |
| Horodatage | date de l'image RGB-D utilisée pour l'estimation |
| Orientation | axes de l'objet définis ci-dessous |

La convention d'orientation est déterminante : `+Y` doit aller du manche vers
les dents, `+X` traverse la largeur de la fourche et `+Z` suit la normale de la
table. Le planificateur place le point de saisie à `-6 cm` sur l'axe local Y.
Une orientation retournée de 180 degrés enverrait donc la pince vers les dents.
La perception doit lever l'ambiguïté de l'axe PCA en identifiant l'extrémité du
manche.

Le planificateur refuse une pose sans `frame_id`, trop ancienne, datée dans le
futur, contenant une valeur non finie ou un quaternion nul. Il normalise le
quaternion puis utilise TF pour convertir la pose vers `base_footprint`. Les
options de réglage sont :

```text
--fork-pose-topic /perception/fork/pose
--perception-timeout 10.0
--max-pose-age 1.0
```

Une fois la pose reçue, les modes de sécurité restent identiques : aucune
commande sans option, pré-préhension avec `--execute`, et saisie complète avec
`--grasp`. Pendant les essais en simulation, Gazebo reste utilisé uniquement
pour la table, la physique et la vérification indépendante du déplacement de la
fourche; il n'est plus la source de la cible en mode perception.

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
  est un exemple plus avancé de scène de collision, validation IK/FK,
  publication dans RViz et saisie physique optionnelle. Il reste en mode
  planification seule tant que `--execute` ou `--grasp` n'est pas fourni.
- Le [manuel TIAGo de PAL Robotics](https://docs.pal-robotics.com/tiago-single/handbook.html)
  complète les informations propres au robot.
