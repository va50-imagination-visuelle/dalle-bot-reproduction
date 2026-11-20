# [M2][Vision] Ticket 1 - Generer l'image cible a partir de la semantique de l'image reelle
#
#   1. YOLOE detecte les objets de la photo reelle      (yoloe_semantics.py)
#   2. le LLM Qwen3-4B redige le prompt                  (prompt_llm.py)
#   3. Z-Image-Turbo genere l'image, sans la photo       (zimage_generator.py)
#   4. l'image est remise aux dimensions de la photo (critere 1)
#   5. YOLOE verifie les objets de l'image generee (critere 3)
#   6. ticket 2 : poses 2D des deux images, matching et deplacements (compare_poses.py)
#
# Lancement :  py generate_from_semantics.py images/table3.jpeg
#              py generate_from_semantics.py images/table3.jpeg --tries 3   (3 images, garde la meilleure)

import argparse
import json
import os
import random

import cv2
import numpy as np
from PIL import Image

try:  # certificats de Windows (evite l'erreur SSL CERTIFICATE_VERIFY_FAILED de Hugging Face)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import compare_poses
import prompt_llm
import yoloe_semantics
import zimage_generator

parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", default="images/table3.jpeg")
parser.add_argument("--classes", nargs="+", default=yoloe_semantics.DEFAULT_CLASSES)
parser.add_argument("--seed", type=int, default=None, help="graine du 1er essai (sinon aleatoire)")
parser.add_argument("--tries", type=int, default=1,
                    help="nombre d'images generees ; avec plus d'une, on garde la meilleure")
parser.add_argument("--table", default="a light wooden tabletop", help="surface de la table")
parser.add_argument("--out_dir", default="outputs")
args = parser.parse_args()

# 1. semantique de la photo reelle
yolo = yoloe_semantics.load_model(detection_element=args.classes)
real = yoloe_semantics.load_image(args.image)
W, H = real.size
real_objects = yoloe_semantics.detect_objects(yolo, real)
if not real_objects:
    raise SystemExit(f"Aucun objet detecte dans {args.image} parmi {args.classes}")
print(f"Image reelle : {args.image} ({W} x {H})")
print(f"Objets       : {dict(real_objects)}")

# 2. le LLM redige le prompt
tokenizer, llm = prompt_llm.load_llm()
prompt = prompt_llm.write_prompt(tokenizer, llm, real_objects, args.table)
print(f"Prompt       :\n{prompt}\n")

# 3-5. generation, mise aux dimensions, verification
pipe = zimage_generator.load_pipeline(tokenizer, llm)
gen_w, gen_h = zimage_generator.generation_size(W, H)
first_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

# on genere toutes les images puis on garde la meilleure :
# d'abord les bons objets (critere 3), puis le moins de bord de table visible
best = None
for i in range(args.tries):
    seed = first_seed + i
    target = zimage_generator.generate(pipe, prompt, gen_w, gen_h, seed).resize((W, H), Image.LANCZOS)
    found = yoloe_semantics.detect_objects(yolo, target)
    errors = sum((real_objects - found).values()) + sum((found - real_objects).values())
    edge = zimage_generator.table_edge_score(target)
    print(f"Image {i + 1}/{args.tries} (graine {seed}) : {dict(found)} "
          f"-> {'OK' if errors == 0 else 'DIFFERENT'}, bord de table {edge:.0f}")
    if best is None or (errors, edge) < (best["errors"], best["edge"]):
        best = {"seed": seed, "target": target, "found": found, "errors": errors, "edge": edge}
print(f"Meilleure    : graine {best['seed']}")

# sauvegarde : image cible, comparaison reel | cible, trace json
name = os.path.splitext(os.path.basename(args.image))[0]
tag = f"{name}_zimage_seed{best['seed']}"
os.makedirs(args.out_dir, exist_ok=True)
target_path = os.path.join(args.out_dir, f"{tag}_target.png")
best["target"].save(target_path)

side = cv2.cvtColor(np.hstack([np.array(real), np.array(best["target"])]), cv2.COLOR_RGB2BGR)
scale = max(W, H) / 1000
for txt, x in (("REEL", 20), ("CIBLE", W + 20)):
    pos = (x, int(50 * scale))
    cv2.putText(side, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5 * scale, (0, 0, 0), int(8 * scale))
    cv2.putText(side, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5 * scale, (255, 255, 255), int(3 * scale))
compare_path = os.path.join(args.out_dir, f"{tag}_compare.jpg")
cv2.imwrite(compare_path, side)

with open(os.path.join(args.out_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
    json.dump({"real_image": args.image, "size": [W, H], "seed": best["seed"],
               "real_objects": dict(real_objects), "target_objects": dict(best["found"]),
               "objects_match": best["errors"] == 0, "prompt": prompt,
               "target_image": target_path}, f, indent=2, ensure_ascii=False)

print(f"\nResultat     : {'memes objets que la photo' if best['errors'] == 0 else 'objets differents'}")
print(f"Image cible  : {target_path}  ({W} x {H})")
print(f"Comparaison  : {compare_path}")

# ticket 2 : deplacement de chaque objet entre la photo et l'image cible
compare_poses.compare(args.image, target_path)
