import argparse
import json
import os
import re
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForImageTextToText, Sam3Processor, Sam3Model

# -------------------------------------------------------------------------------------
# 1. PARAMETRES
# -------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--vlm", default="Qwen/Qwen3-VL-2B-Instruct")
parser.add_argument("--min_score", type=float, default=0.50) # confiance minimale de SAM 3
parser.add_argument("--instruction", default="") # consigne ajoutee a l'edit_prompt
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------------------
# 2. PHOTO
# -------------------------------------------------------------------------------------
image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
W, H = image.size

name = os.path.splitext(os.path.basename(args.image))[0]
out_dir = os.path.join("out_pipeline", name)
os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
print(f"Image : {args.image}  ({W} x {H})  |  appareil : {device}")

# -------------------------------------------------------------------------------------
# 3. QWEN3-VL : LISTE DES OBJETS
# -------------------------------------------------------------------------------------
PROMPT = """You see a scene from above. List every physical object lying on the main surface (table, desk, shelf...).
Return ONLY valid JSON, no other text, with this exact format:
{"context": "<type of scene, e.g. dinner table, desk>", "objects": [{"name": "<short English noun, singular, e.g. fork>", "count": <integer>}]}
Rules: one entry per kind of object, with count = how many there are. Do not list the surface itself.
"""

vlm = AutoModelForImageTextToText.from_pretrained(args.vlm, dtype=torch.bfloat16, device_map="auto").eval()
vproc = AutoProcessor.from_pretrained(args.vlm)

small = image.copy()
small.thumbnail((1024, 1024)) # reduction de la copie pour le VLM - eviter de saturer la memoire

def ask_vlm():
    messages = [{"role": "user", "content": [{"type": "image", "image": small}, {"type": "text", "text": PROMPT}]}]
    inputs = vproc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(vlm.device)
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=512, do_sample=False)
    return vproc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

scene_vlm, raw = None, ""
for attempt in range(3): # JSON est invalide -->  3 essais 
    raw = ask_vlm()
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        scene_vlm = json.loads(m.group(0))
        assert isinstance(scene_vlm.get("objects"), list) and scene_vlm["objects"]
        break
    except Exception:
        scene_vlm = None
        print(f"  reponse non exploitable (essai {attempt + 1}), on redemande")

if scene_vlm is None:
    raise SystemExit(f"Qwen3-VL n'a pas renvoye de JSON valide :\n{raw}")

# liste nettoyee (noms en minuscules, sans doublons, avec le nombre attendu)
names, expected = [], {}
for o in scene_vlm["objects"]:
    n = str(o.get("name", "")).strip().lower()
    if n and n not in names:
        names.append(n)
        expected[n] = int(o.get("count", 1) or 1)
print("Qwen3-VL :", ", ".join(f"{n} x{expected[n]}" for n in names))

del vlm
if device == "cuda":
    torch.cuda.empty_cache()

context = scene_vlm.get("context") or "table"

# -------------------------------------------------------------------------------------
# 4. EDIT_PROMPT POUR LE GENERATEUR D'IMAGE
# -------------------------------------------------------------------------------------
def plural(n):
    """Pluriel anglais par regles simples (knife --> knives, glass --> glasses, fork --> forks)."""
    if n.endswith("fe"):
        return n[:-2] + "ves"
    if n.endswith(("s", "x", "ch", "sh", "z")):
        return n + "es"
    return n + "s"

def build_edit_prompt(expected, context, instruction=""):
    words = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five"}
    parts = [f"{words.get(c, c)} {n if c == 1 else plural(n)}" for n, c in expected.items()]
    items = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return (f"Rearrange these objects into a neat {context} setting: {items}. "
            f"Keep exactly the same objects, the same camera view and the same {context}. "
            f"Do not add or remove anything. {instruction}").strip()

edit_prompt = build_edit_prompt(expected, context, args.instruction)

# -------------------------------------------------------------------------------------
# 5. SAM 3 : UN MASQUE PAR OBJET
# -------------------------------------------------------------------------------------
sam = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
sproc = Sam3Processor.from_pretrained("facebook/sam3")

def sam3(text):
    """Cherche TOUS les objets correspondant a 'text' : [(score, masque, boite), ...]"""
    inputs = sproc(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam(**inputs)
    res = sproc.post_process_instance_segmentation(outputs, threshold=args.min_score, mask_threshold=0.5,
                                                   target_sizes=inputs.get("original_sizes").tolist())[0]
    return [(float(res["scores"][k]),
             res["masks"][k].cpu().numpy().astype(bool),
             [float(v) for v in res["boxes"][k].tolist()])
            for k in range(len(res["scores"]))]

def largest_component(m):
    """Garde la plus grande zone du masque - supp les petits morceaux detaches."""
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return m if n_lab <= 2 else lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))

def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)

named = []
for n in names:
    found = sam3(n)
    print(f"  SAM 3 '{n}' : {len(found)} trouve(s) (attendu : {expected[n]})")
    named += [(s, n, largest_component(m), b) for s, m, b in found]

# du meilleur score au moins bon : on enleve le bruit (< 200 px) et les doublons (IoU > 0.7)
kept = []
for s, n, m, b in sorted(named, key=lambda c: c[0], reverse=True):
    if m.sum() < 200:
        continue
    if any(iou(m, k[2]) > 0.7 for k in kept):
        continue
    kept.append((s, n, m, b))

# -------------------------------------------------------------------------------------
# 6. SAUVEGARDE DES MASQUES + IMAGES DE CONTROLE
# -------------------------------------------------------------------------------------
bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
overlay, semantic = bgr.copy(), np.zeros_like(bgr)
scale = max(W, H) / 1500

# une couleur par NOM --> image semantique
labels = sorted({n for _, n, _, _ in kept})
palette = {n: tuple(int(c) for c in cv2.applyColorMap(
    np.uint8([[int(255 * k / max(len(labels), 1))]]), cv2.COLORMAP_TURBO)[0, 0])
    for k, n in enumerate(labels)}

objects = []
for s, n, m, b in kept:
    oid = f"o{len(objects) + 1}"
    path = os.path.join(out_dir, "masks", f"{oid}_{n.replace(' ', '_')}.png")
    cv2.imwrite(path, m.astype(np.uint8) * 255)
    objects.append({"id": oid, "name": n, "score": round(s, 3), "bbox_px": [round(v) for v in b], "mask": path})

    col = palette[n]
    overlay[m] = (0.45 * overlay[m] + 0.55 * np.array(col)).astype(np.uint8)
    semantic[m] = col
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, cnts, -1, col, max(2, int(2 * scale)))

    label = f"{oid} {n} ({s:.2f})"
    ys, xs = np.nonzero(m)
    pos = (int(xs.mean()), int(ys.mean()))
    cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 0), int(5 * scale))
    cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), int(2 * scale))

cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)
cv2.imwrite(os.path.join(out_dir, "semantic.png"), semantic)

# controle : le VLM et SAM 3 trouvent-ils le meme nombre de chaque objet ?
counts = {n: sum(o["name"] == n for o in objects) for n in names}
mismatch = {n: {"vlm": expected[n], "sam3": counts[n]} for n in names if counts[n] != expected[n]}
if mismatch:
    print(f"  Attention, comptages differents entre Qwen3-VL et SAM 3 : {mismatch}")

# -------------------------------------------------------------------------------------
# 7. scene.json
# -------------------------------------------------------------------------------------
with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as f:
    json.dump({"image": args.image, "size": [W, H],
               "context": context,
               "edit_prompt": edit_prompt,
               "vlm_objects": expected,
               "count_mismatch": mismatch,
               "objects": objects},
              f, indent=2, ensure_ascii=False)

print(f"\n{len(objects)} objets gardes :")
for o in objects:
    print(f'  {o["id"]:>3}  {o["name"]:<9} score {o["score"]:.2f}')
print(f"Prompt edition : {edit_prompt}")
print(f"Resultats dans {out_dir}\\")