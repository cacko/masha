from pathlib import Path
from tabnanny import verbose
from tf_keras.models import Model, Sequential
from tf_keras.layers import (
    Convolution2D,
    Flatten,
    Activation,
    MaxPooling2D,
    ZeroPadding2D,
    Dropout,
)
from tf_keras.preprocessing import image
from typing import Any
import os
import cv2
import numpy as np
import tensorflow as tf

from masha.name.classifiers import Ethnicity

HGROOT = Path(os.environ.get("HUGGINGROOT"))
import os


def base_model() -> Sequential:
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model


def resize_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    # Put the base image in the middle of the padded image
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # make it 4-dimensional how ML models expect
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img


class DeepfaceMeta(type):
    __instances: dict[str, "BaseClient"] = {}

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        k = cls.__name__
        if k not in cls.__instances:
            cls.__instances[k] = type.__call__(cls, *args, **kwds)
        return cls.__instances[k]

    def classify(cls, image_path: Path):
        return cls().get_result(image_path)


class BaseClient(object, metaclass=DeepfaceMeta):
    model: Model
    model_name: str
    model_path: str
    classes: int

    def __init__(self) -> None:
        tf.config.set_visible_devices(tf.config.get_visible_devices("CPU"))
        self.model = self.load_model()


    def get_result(self, image_path: Path):
        raise NotImplemented

    def load_image(self, image_path: Path):
        img = cv2.imread(image_path.as_posix())
        img = resize_image(img=img, target_size=(224, 224))
        return img

    def load_model(self) -> Model:
        model_path = HGROOT / self.model_path
        model = base_model()
        base_model_output = Sequential()
        base_model_output = Convolution2D(self.classes, (1, 1), name="predictions")(
            model.layers[-4].output
        )
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        client_model = Model(inputs=model.input, outputs=base_model_output)
        client_model.load_weights(model_path.as_posix())
        return client_model


class RaceClient(BaseClient):

    labels = ["asian", "indian", "black", "white", "middle_eastern", "latino_hispanic"]
    model_name = "Race"
    model_path = "race_model_single_batch.h5"
    classes = 6

    def get_result(self, image_path: Path) -> str:
        img = self.load_image(image_path=image_path)
        race_predictions = self.model.predict(img, verbose=0)
        return Ethnicity(self.labels[np.argmax(race_predictions)])


class AgeClient(BaseClient):

    classes = 101
    model_name = "Age"
    model_path = "age_model_weights.h5"

    def find_apparent_age(self, age_predictions: np.ndarray) -> np.float64:
        output_indexes = np.array(list(range(0, 101)))
        apparent_age = np.sum(age_predictions * output_indexes)
        return round(apparent_age)

    def get_result(self, image_path: Path):
        img = self.load_image(image_path=image_path)
        age_predictions = self.model.predict(img, verbose=0)
        return self.find_apparent_age(age_predictions)
