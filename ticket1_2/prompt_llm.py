# Redaction du prompt par une IA (LLM Qwen3-4B, en local).
# On lui donne la liste EXACTE des objets, la disposition EXACTE voulue et un exemple de prompt
# qui fonctionne. Ce LLM est aussi l'encodeur de texte de Z-Image-Turbo (un seul chargement).

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"

# disposition voulue de chaque objet ; l'ordre compte : un objet n'est decrit
# qu'apres celui qui lui sert de repere (la cuillere "a droite du couteau" apres le couteau)
PLACEMENT = {
    "plate": "in the center of the table",
    "fork": "immediately to the left of the plate, vertical",
    "knife": "immediately to the right of the plate, vertical",
    "spoon": "to the right of the knife, vertical",
    "glass": "above the plate, at the top right",
    "cup": "at the top right of the plate",
    "bowl": "on top of the plate",
    "napkin": "to the left of the fork",
}

# exemple de prompt qui fonctionne (fourni par le prof) : donne au LLM le style attendu
EXAMPLE_PROMPT = """A perfectly top-down, 90-degree bird's-eye view of a table, with the camera positioned directly above it, facing straight down.

Exactly three objects must be visible on the tabletop:

One plate, positioned centrally.

One fork, placed immediately to the right of the plate.

One knife, placed immediately to the left of the plate.

All three objects must be clearly visible, fully inside the frame, and spatially separated. The fork and knife should be arranged parallel to each other, with their handles oriented in the same direction.

Strict object constraint: The plate, fork, and knife are the ONLY objects in the entire image. No additional items, utensils, cutlery, food, decorations, hands, people, or background objects. No text or labels.

Clean, uncluttered composition, with the table as the only background."""


def load_llm():
    """Qwen3-4B en 4 bits (tient dans 8 Go de VRAM). Renvoie (tokenizer, model)."""
    tokenizer = AutoTokenizer.from_pretrained(ZIMAGE_REPO, subfolder="tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        ZIMAGE_REPO, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    return tokenizer, model


def build_request(objects, table="a light wooden tabletop"):
    """Consigne donnee au LLM : liste exacte des objets et disposition exacte voulue."""
    order = list(PLACEMENT)
    objects = dict(sorted(objects.items(),
                          key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))
    has_plate = "plate" in objects

    listing = "\n".join(f"- {n} {name}" for name, n in objects.items())
    layout = []
    for name, n in objects.items():
        where = PLACEMENT.get(name) if has_plate else None
        where = where or "flat on the table, clearly separated from the other objects"
        if n > 1:
            where += f" (the {n} {name}s side by side, parallel)"
        layout.append(f"- {name}: {where}")

    return (
        "Write a prompt for a text-to-image model (Z-Image-Turbo).\n"
        "The image must show a perfectly top-down, 90-degree bird's-eye view of a table.\n"
        f"Table surface: {table}. The tabletop fills the ENTIRE image from edge to edge: "
        "no table edge, floor, table legs, chairs or room may be visible.\n\n"
        f"Objects (exact list, exact quantities, nothing else allowed):\n{listing}\n\n"
        "Exact layout:\n" + "\n".join(layout) + "\n\n"
        "Requirements: state the exact total number of objects; give one separate line per "
        "object with its exact position; say that all objects are clearly visible, fully "
        "inside the frame and spatially separated; add a strict constraint that these are "
        "the ONLY objects in the image (no additional items, utensils, cutlery, food, "
        "decorations, hands, people or text); describe the table surface and state that the "
        "tabletop fills the entire frame edge to edge with no table edge, floor or furniture "
        "visible; end with a clean, uncluttered composition with the table as the only "
        "background.\n\n"
        "Use exactly the same style and structure as this example prompt, which works well:\n"
        f'"""\n{EXAMPLE_PROMPT}\n"""\n\n'
        "Answer with the prompt only, in English, no title, no quotes, no explanation."
    )


def write_prompt(tokenizer, model, objects, table="a light wooden tabletop"):
    """Le LLM redige le prompt de generation a partir des objets detectes."""
    messages = [{"role": "user", "content": build_request(objects, table)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)  # reponse directe
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=600, do_sample=False)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # les premiers mots pesent le plus pour le modele d'image : on commence par imposer
    # un gros plan sur le plateau (sinon il dessine la table entiere, avec son bord et le sol)
    framing = (f"Close-up overhead photo of {table} that fills the entire frame from edge to "
               "edge; only the tabletop surface is visible, no table edge, floor or furniture.")
    return framing + "\n\n" + answer.strip().strip('"').strip()
