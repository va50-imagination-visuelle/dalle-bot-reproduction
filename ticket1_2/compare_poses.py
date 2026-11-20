# [M2][Vision] Ticket 2 - Associer les objets et comparer leurs poses 2D (dx, dy, dtheta)
#
# Appele automatiquement par generate_from_semantics.py, ou seul :
#   py compare_poses.py images/table3.jpeg outputs/table3_zimage_seed0_target.png
#
# Convention (celle de find_angle) : x vers la droite, y vers le BAS (pixels),
# theta = 0 deg a l'horizontale, positif dans le sens horaire a l'ecran.

import json
import os
import sys

import cv2
import numpy as np

from extract_segmentation_position_orientation import (
    find_angle, find_center, model_prediction, normalize_angle)

CLASSES = ["plate", "fork", "knife", "spoon", "glass", "cup", "bowl", "napkin"]


def get_poses(image_path, seg_path):
    """Objets d'une image avec le module YOLOE : liste de (classe, x, y, theta).
    Enregistre aussi l'image de segmentation (boites + masques) dans seg_path."""
    result = model_prediction("models/yoloe-26l-seg.pt", image_path, CLASSES)[0]
    cv2.imwrite(seg_path, result.plot())
    poses = []
    for mask, box, bbox in zip(result.masks.xy, result.boxes, result.boxes.xyxy):
        label = result.names[int(box.cls[0])]
        x, y = find_center(bbox)
        theta = 0.0 if label == "plate" else float(find_angle(mask))
        poses.append((label, x, y, theta))
    return poses


def match(real, target):
    """Associe les objets de meme classe (paires les plus proches d'abord) et calcule les deltas."""
    moves = []
    for label in sorted({p[0] for p in real + target}):
        r = [p for p in real if p[0] == label]
        t = [p for p in target if p[0] == label]
        pairs = []
        for _, i, j in sorted((np.hypot(a[1] - b[1], a[2] - b[2]), i, j)
                              for i, a in enumerate(r) for j, b in enumerate(t)):
            if i not in {p[0] for p in pairs} and j not in {p[1] for p in pairs}:
                pairs.append((i, j))
        for i, j in pairs:
            a, b = r[i], t[j]
            moves.append({"object": label, "real": a[1:], "target": b[1:],
                          "dx": b[1] - a[1], "dy": b[2] - a[2],
                          # un couvert n'a pas de sens : on garde la plus petite rotation
                          "dtheta": normalize_angle(b[3] - a[3])})
        for k in set(range(len(r))) - {i for i, _ in pairs}:
            print(f"{label} present dans l'image reelle mais pas dans la cible : {r[k][1:]}")
        for k in set(range(len(t))) - {j for _, j in pairs}:
            print(f"{label} present dans la cible mais pas dans l'image reelle : {t[k][1:]}")
    return moves


def draw(real_path, target_path, moves, out_path):
    """Image de controle : centre (point vert) et axe (trait rouge) de chaque objet, reel | cible."""
    real = cv2.imread(real_path)
    target = cv2.resize(cv2.imread(target_path), (real.shape[1], real.shape[0]))
    s = max(real.shape[:2]) / 1000
    for m in moves:
        for img, (x, y, th) in ((real, m["real"]), (target, m["target"])):
            dx, dy = 120 * s * np.cos(np.radians(th)), 120 * s * np.sin(np.radians(th))
            cv2.line(img, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), (0, 0, 255), int(3 * s))
            cv2.circle(img, (int(x), int(y)), int(8 * s), (0, 255, 0), -1)
            cv2.putText(img, f"{m['object']} ({x:.0f}, {y:.0f}, {th:.0f} deg)",
                        (int(x + 15 * s), int(y - 15 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * s, (255, 0, 0), int(2 * s))
    cv2.imwrite(out_path, np.hstack([real, target]))


def compare(real_path, target_path):
    base = os.path.splitext(target_path)[0]
    moves = match(get_poses(real_path, base + "_seg_reel.jpg"),
                  get_poses(target_path, base + "_seg_cible.jpg"))

    print(f"\n{'objet':8s}{'reel (x, y, theta)':>26s}{'cible (x, y, theta)':>26s}{'dx':>8s}{'dy':>8s}{'dtheta':>9s}")
    for m in moves:
        pose = lambda p: f"({p[0]:.0f}, {p[1]:.0f}, {p[2]:.1f})"
        print(f"{m['object']:8s}{pose(m['real']):>26s}{pose(m['target']):>26s}"
              f"{m['dx']:+8.0f}{m['dy']:+8.0f}{m['dtheta']:+8.1f}°")

    with open(base + "_moves.json", "w", encoding="utf-8") as f:
        json.dump({"real_image": real_path, "target_image": target_path, "moves": moves}, f, indent=2)
    draw(real_path, target_path, moves, base + "_poses.jpg")
    print(f"\nDeplacements : {base}_moves.json")
    print(f"Controle     : {base}_poses.jpg")
    print(f"Segmentation : {base}_seg_reel.jpg et {base}_seg_cible.jpg")
    return moves


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
