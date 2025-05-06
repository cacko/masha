from pathlib import Path
from torch import mode
from transformers import AutoImageProcessor, AutoModelForImageClassification
from diffusers.utils import load_image
import os
from stringcase import titlecase

HGROOT = Path(os.environ.get("HUGGINGROOT"))


def get_dog_breed(image: Path):
    image = load_image(image.as_posix())
    model_path = HGROOT / "dog-breeds-multiclass-image-classification-with-vit"
    image_processor = AutoImageProcessor.from_pretrained(model_path.as_posix())
    model = AutoModelForImageClassification.from_pretrained(model_path.as_posix())

    inputs = image_processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    # model predicts one of the 120 Stanford dog breeds classes
    predicted_class_idx = logits.argmax(-1).item()
    return titlecase(model.config.id2label[predicted_class_idx])
