# Semantique d'une image avec YOLOE : quels objets sont presents, et combien.

from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLOE

# vocabulaire donne a YOLOE (detection a vocabulaire ouvert : il ne cherche que ces classes)
DEFAULT_CLASSES = ["plate", "fork", "knife", "spoon", "glass", "cup", "bowl", "napkin"]


def load_image(image_path):
    """Photo en RGB avec l'orientation EXIF appliquee (photos de telephone)."""
    return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")


def load_model(model_choice="models/yoloe-26l-seg.pt", detection_element=DEFAULT_CLASSES):
    model = YOLOE(model_choice)
    model.set_classes(detection_element)
    return model


def detect_objects(model, image, conf=0.25):
    """Compte les objets detectes dans une image PIL : Counter({"plate": 1, "fork": 1, ...})."""
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = model.predict(image_bgr, conf=conf, verbose=False)[0]
    return Counter(result.names[int(c)] for c in result.boxes.cls)
