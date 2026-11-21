# DALL-E-Bot reproduction

Reproduction du pipeline décrit dans l'article *DALL-E-Bot: Introducing
Web-Scale Diffusion Models to Robotics*, avec ROS Noetic, le robot TIAGo,
Gazebo et RViz.

## Organisation du projet

L'environnement de développement principal se trouve dans `noetic-tiago/` :

```text
noetic-tiago/
├── build.sh                  Construction de l'image Docker
├── server.sh                 Démarrage du conteneur
├── client.sh                 Ouverture d'un terminal dans le conteneur
├── docker/                   Dockerfile, configuration et image PAL locale
├── workspace/
│   ├── src/                  Sources ROS et ressources de simulation
│   ├── build/                Fichiers Catkin générés, ignorés par Git
│   └── devel/                Environnement Catkin généré, ignoré par Git
├── ros/                      Logs et état local ROS, ignorés par Git
└── share/                    Échange temporaire de fichiers, ignoré par Git
```

Le dossier `noetic-tiago/workspace` est monté dans le conteneur sous
`/home/pal/ros_ws`. Les modifications réalisées dans `workspace/src` sont donc
persistantes et doivent être versionnées. Les dossiers générés par Catkin ne
doivent pas être ajoutés à Git.

## Prérequis

- Linux avec Docker installé et le daemon Docker démarré ;
- accès de l'utilisateur courant à Docker ;
- une session graphique X11 pour afficher Gazebo et RViz ;
- l'archive de développement TIAGo fournie par PAL Robotics.

Placer l'archive à l'emplacement exact suivant :

```text
noetic-tiago/docker/image/TIAGo-development-computer-ubuntu-20.04.tar.gz
```

Cette archive fait plusieurs gigaoctets et est volontairement ignorée par
Git. Voir également `noetic-tiago/docker/image/README.md`.

## 1. Construire l'image Docker

Les commandes doivent être exécutées depuis le dossier `noetic-tiago` :

```bash
cd noetic-tiago
./build.sh
```

L'image produite est :

```text
epan/tiago:noetic
```

Pour reconstruire entièrement l'image sans utiliser le cache Docker :

```bash
./build.sh -r
```

Vérification :

```bash
docker image ls epan/tiago:noetic
```

## 2. Démarrer le conteneur

```bash
./server.sh
```

Le conteneur s'appelle `epan-tiago-noetic-ssh`. Son serveur SSH écoute sur le
port `2222`, car le port standard `22` peut déjà être utilisé sur l'hôte.

Vérification :

```bash
docker ps --filter name=epan-tiago-noetic-ssh
```

Le script monte automatiquement les dossiers suivants :

| Hôte | Conteneur | Rôle |
|---|---|---|
| `noetic-tiago/workspace` | `/home/pal/ros_ws` | Sources et compilation ROS |
| `noetic-tiago/ros` | `/home/pal/.ros` | Logs et cache ROS |
| `noetic-tiago/share` | `/home/pal/share` | Échange de fichiers |

Le conteneur est lancé avec `--rm` : il est supprimé automatiquement lorsqu'il
s'arrête, mais les dossiers montés ci-dessus restent présents sur l'hôte.

## 3. Entrer dans le conteneur

```bash
./client.sh
```

Cette commande utilise `docker exec` et ouvre un terminal Bash avec
l'utilisateur `pal`. Elle ne nécessite pas de connexion SSH.

## 4. Compiler le workspace ROS

Dans le conteneur :

```bash
cd /home/pal/ros_ws
catkin_make -DCATKIN_ENABLE_TESTING=OFF -DDISABLE_PAL_FLAGS=ON
source devel/setup.bash
```

Une recompilation est nécessaire après l'ajout d'un package, d'un message,
d'un service ou d'une cible CMake. Les modifications de scripts Python ne
nécessitent pas toujours une recompilation.

## 5. Lancer TIAGo dans Gazebo et RViz

Toujours dans le conteneur :

```bash
cd /home/pal/ros_ws
./start_tiago_simulation.sh
```

Le script :

1. lance TIAGo et le monde VA50 dans Gazebo ;
2. démarre les contrôleurs ROS ;
3. ouvre RViz avec la configuration du projet ;
4. positionne le torse et le bras du robot devant la table.

Utiliser `Ctrl+C` dans le terminal de lancement pour arrêter proprement la
simulation.

## Affichage graphique

Gazebo et RViz utilisent la variable `DISPLAY` et le socket X11 de l'hôte. Sur
une machine distante, une session de bureau distant est recommandée. Un
transfert X11 avec `ssh -Y` peut également être utilisé, mais il peut être lent
pour les applications 3D et nécessite une configuration X11 fonctionnelle.

## Arrêter le conteneur

```bash
docker stop epan-tiago-noetic-ssh
```

## Développement

Le code ROS, les fichiers Launch, URDF/Xacro, mondes Gazebo, modèles et
configurations RViz doivent être ajoutés sous :

```text
noetic-tiago/workspace/src/
```

Ne pas placer de code source dans `workspace/build` ou `workspace/devel` : ces
dossiers sont reconstruits automatiquement et sont ignorés par Git.
