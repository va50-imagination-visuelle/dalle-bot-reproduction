# Generation d'image a partir du texte seul avec Z-Image-Turbo (Tongyi-MAI / Alibaba), en local.
# Version allegee pour un GPU de 8 Go : transformer quantifie GGUF Q4_K_M + encodeur 4 bits
# + offload (chaque sous-modele ne passe sur le GPU que quand il sert, ~4,5 Go de VRAM max).

import numpy as np
import torch
from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
from huggingface_hub import hf_hub_download

ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
GGUF_REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF_FILE = "z-image-turbo-Q4_K_M.gguf"


def load_pipeline(tokenizer, text_encoder):
    """Pipeline Z-Image ; reutilise le LLM deja charge comme encodeur de texte."""
    transformer = ZImageTransformer2DModel.from_single_file(
        hf_hub_download(GGUF_REPO, GGUF_FILE),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=ZIMAGE_REPO, subfolder="transformer", dtype=torch.bfloat16)
    pipe = ZImagePipeline.from_pretrained(ZIMAGE_REPO, transformer=transformer,
                                          text_encoder=text_encoder, tokenizer=tokenizer,
                                          dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generation_size(width, height):
    """~1 Mpx avec le meme rapport largeur/hauteur que la photo (multiples de 64)."""
    k = (1024 * 1024 / (width * height)) ** 0.5
    return max(64, int(width * k) // 64 * 64), max(64, int(height * k) // 64 * 64)


def table_edge_score(image):
    """Plus grand saut de luminosite entre deux lignes horizontales de l'image.

    Un bord de table (plateau puis sol, pied ou chaise) donne un saut brutal (> ~50) ;
    un plateau qui remplit toute l'image reste faible (< ~20). Plus petit = mieux.
    """
    gray = np.asarray(image.convert("L").resize((200, 300)), dtype=np.float32)
    rows = gray.mean(axis=1)[::3]
    return float(np.abs(np.diff(rows)).max())


def generate(pipe, prompt, width, height, seed):
    return pipe(
        prompt=prompt, width=width, height=height, num_inference_steps=9,
        guidance_scale=0.0,  # 0 pour les modeles "Turbo" (guidage distille)
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]
