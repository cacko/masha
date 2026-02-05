import logging
from pathlib import Path
from corestring import round2
import torch
from diffusers.utils import load_image
from masha.image.upscale import Upscale
from masha.pipelines import BasePipeline, TORCH_DEVICE
from masha.pipelines.image_clasify.models import ClassifyResult
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from PIL import Image


def convertor_upscale(image: Image.Image):
    try:
        image = image.convert("RGB")
        w, h = image.width, image.height
        min_dim = min(w, h)
        assert 1200 > min_dim
        scale = min(round2(1200 / min_dim), 4)
        return Upscale.upscale_img(img=image, scale=scale).convert("RGB")
    except AssertionError:
        pass

    return image


class ImageClassify(BasePipeline):

    _tokenizer = None
    _model = None
    _pipeline = None

    @property
    def tokenizer(self):
        self._tokenizer = AutoImageProcessor.from_pretrained(
            self.tokenizerPath.as_posix(),
            use_fast=True,
        )
        self._tokenizer.model_max_length = 1024
        return self._tokenizer

    @property
    def model(self):
        self._model = AutoModelForImageClassification.from_pretrained(
            self.modelPath.as_posix(),
        )
        return self._model

    @property
    def pipeline(self):
        if not self._pipeline:
            self._pipeline = pipeline(
                "image-classification",
                image_processor=self.tokenizer,
                model=self.model,
                device=TORCH_DEVICE,
            )
        return self._pipeline

    def getAnswer(self, image: Path, threshold=0.2, **kwargs) -> list[ClassifyResult]:

        processor = self.tokenizer
        model = self.model

        inputs = processor(
            images=load_image(image.as_posix(), convert_method=convertor_upscale),
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        logging.debug(predicted_class_idx)
        logging.debug(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
        try:
            res = self.pipeline(image.as_posix())
            logging.debug(res)
            assert res
            tags = [ClassifyResult(cls=self.__class__.__name__, **x) for x in res]
            return sorted(
                filter(lambda x: x.score > threshold, tags),
                key=lambda y: y.score,
                reverse=True,
            )
        except Exception as e:
            logging.debug(e)
            return [
                ClassifyResult(
                    value=model.config.id2label[predicted_class_idx],
                    score=0.5,
                    cls=self.__class__.__name__,
                )
            ]
