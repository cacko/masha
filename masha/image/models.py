import shutil
from pydantic import BaseModel, Field
from typing import Any, Optional
from enum import StrEnum
from pathlib import Path
from PIL import Image
from PIL.ExifTags import Base as TagNames
from masha.core import Choices
from masha.core.bytes import round8
import torch
from corestring import truncate, clean_newlines
from stringcase import alphanumcase
from datetime import datetime
from functools import reduce
from masha.name.classifiers import Ethnicity
import shlex
from masha.image.huggingface.lora import clean_lora
from fuzzelinho import extract


class IncompatibleTemplate(Exception):
    pass


AnimalName = StrEnum(
    "AnimalName",
    [
        "Aardvark",
        "Albatross",
        "Alligator",
        "Alpaca",
        "Ant",
        "Anteater",
        "Antelope",
        "Ape",
        "Armadillo",
        "Donkey",
        "Baboon",
        "Badger",
        "Barracuda",
        "Bat",
        "Bear",
        "Beaver",
        "Bee",
        "Bison",
        "Boar",
        "Buffalo",
        "Butterfly",
        "Camel",
        "Capybara",
        "Caribou",
        "Cassowary",
        "Cat",
        "Caterpillar",
        "Cattle",
        "Chamois",
        "Cheetah",
        "Chicken",
        "Chimpanzee",
        "Chinchilla",
        "Chough",
        "Clam",
        "Cobra",
        "Cockroach",
        "Cod",
        "Cormorant",
        "Coyote",
        "Crab",
        "Crane",
        "Crocodile",
        "Crow",
        "Curlew",
        "Deer",
        "Dinosaur",
        "Dog",
        "Dogfish",
        "Dolphin",
        "Dotterel",
        "Dove",
        "Dragonfly",
        "Duck",
        "Dugong",
        "Dunlin",
        "Eagle",
        "Echidna",
        "Eel",
        "Eland",
        "Elephant",
        "Elk",
        "Emu",
        "Falcon",
        "Ferret",
        "Finch",
        "Fish",
        "Flamingo",
        "Fly",
        "Fox",
        "Frog",
        "Gaur",
        "Gazelle",
        "Gerbil",
        "Giraffe",
        "Gnat",
        "Gnu",
        "Goat",
        "Goldfinch",
        "Goldfish",
        "Goose",
        "Gorilla",
        "Goshawk",
        "Grasshopper",
        "Grouse",
        "Guanaco",
        "Gull",
        "Hamster",
        "Hare",
        "Hawk",
        "Hedgehog",
        "Heron",
        "Herring",
        "Hippopotamus",
        "Hornet",
        "Horse",
        "Human",
        "Hummingbird",
        "Hyena",
        "Ibex",
        "Ibis",
        "Jackal",
        "Jaguar",
        "Jay",
        "Jellyfish",
        "Kangaroo",
        "Kingfisher",
        "Koala",
        "Kookabura",
        "Kouprey",
        "Kudu",
        "Lapwing",
        "Lark",
        "Lemur",
        "Leopard",
        "Lion",
        "Llama",
        "Lobster",
        "Locust",
        "Loris",
        "Louse",
        "Lyrebird",
        "Magpie",
        "Mallard",
        "Manatee",
        "Mandrill",
        "Mantis",
        "Marten",
        "Meerkat",
        "Mink",
        "Mole",
        "Mongoose",
        "Monkey",
        "Moose",
        "Mosquito",
        "Mouse",
        "Mule",
        "Narwhal",
        "Newt",
        "Nightingale",
        "Octopus",
        "Okapi",
        "Opossum",
        "Oryx",
        "Ostrich",
        "Otter",
        "Owl",
        "Oyster",
        "Panther",
        "Parrot",
        "Partridge",
        "Peafowl",
        "Pelican",
        "Penguin",
        "Pheasant",
        "Pig",
        "Pigeon",
        "Pony",
        "Porcupine",
        "Porpoise",
        "Quail",
        "Quelea",
        "Quetzal",
        "Rabbit",
        "Raccoon",
        "Rail",
        "Ram",
        "Rat",
        "Raven",
        "Red deer",
        "Red panda",
        "Reindeer",
        "Rhinoceros",
        "Rook",
        "Salamander",
        "Salmon",
        "Sand Dollar",
        "Sandpiper",
        "Sardine",
        "Scorpion",
        "Seahorse",
        "Seal",
        "Shark",
        "Sheep",
        "Shrew",
        "Skunk",
        "Snail",
        "Snake",
        "Sparrow",
        "Spider",
        "Spoonbill",
        "Squid",
        "Squirrel",
        "Starling",
        "Stingray",
        "Stinkbug",
        "Stork",
        "Swallow",
        "Swan",
        "Tapir",
        "Tarsier",
        "Termite",
        "Tiger",
        "Toad",
        "Trout",
        "Turkey",
        "Turtle",
        "Viper",
        "Vulture",
        "Wallaby",
        "Walrus",
        "Wasp",
        "Weasel",
        "Whale",
        "Wildcat",
        "Wolf",
        "Wolverine",
        "Wombat",
        "Woodcock",
        "Woodpecker",
        "Worm",
        "Wren",
        "Yak",
        "Zebra",
    ],
)


class Resolutions(Choices, StrEnum):
    HD = "16:9"
    UHD = "21:9"
    SD = "4:3"
    SDI = "3:4"
    HDI = "9:16"
    UHDI = "21:9"
    SQ = "1:1"


class Sex(Choices, StrEnum):
    M = "male"
    F = "female"

    @property
    def label(self) -> str:
        match self:
            case Sex.M:
                return "man"
            case Sex.F:
                return "woman"
        return "woman"


class PipelineType(StrEnum):
    KANDINSKY = "KandinskyV22Pipeline"
    KANDINSKY3 = "Kandinsky3Pipeline"
    SD = "StableDiffusionPipeline"
    SDXL = "StableDiffusionXLPipeline"
    COREML = "CoreMLStableDiffusionPipeline"
    PIXART = "PixArtAlphaPipeline"
    LCM = "LatentConsistencyModelPipeline"
    SDXL_TURBO = ""
    SD3 = "StableDiffusion3Pipeline"
    FLUX = "FluxPipeline"


class SDModelType(StrEnum):
    KANDINSKY = "kandinsky"
    KANDINSKY3 = "kandinsky3"
    SD = "sd"
    SDXL = "sdxl"
    SD3 = "sd3"
    COREML = "coreml"
    PIXART = "pixart"
    LCM = "lcm"
    OPENDALLE = "opendalle"
    FLUX = "flux"


class PipelineParams(BaseModel):
    prompt: str = Field(default="")
    height: int = Field(default=1024)
    width: int = Field(default=1024)
    guidance_scale: float = Field(default=7)
    num_inference_steps: int = Field(default=25)
    num_images_per_prompt: int = Field(default=1)
    strength: float = Field(default=0.5)
    image_guidance_scale: float = Field(default=1)
    seed: Optional[int] = None
    upscale: Optional[int] = None
    auto_prompt: Optional[str] = None
    model: str = Field(default="default")
    aspect_ratio: Optional[str] = None
    negative_prompt: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    editing_prompt: Optional[list[str]] = None
    cross_attention_kwargs: Optional[dict[str, float]] = None
    clip_skip: Optional[int] = None
    scale: Optional[float] = None
    person: Optional[bool] = False
    no_age: Optional[bool] = None
    no_race: Optional[bool] = None
    template: Optional[str] = None
    style: Optional[str] = None

    def __init__(self, **data):
        args = {
            k: clean_newlines(v) if isinstance(v, str) else v
            for k, v in data.items()
            if v is not None
        }
        super().__init__(**args)

    @property
    def output_prompt(self) -> str:
        res = [self.prompt]
        if self.prefix:
            res = [self.prefix, *res]
        if self.suffix:
            res.append(self.suffix)
        return ", ".join(res)

    @property
    def aspect(self) -> Optional[float]:
        try:
            assert self.aspect_ratio
            res = Resolutions[self.aspect_ratio.upper()]
            w, h = list(map(int, res.value.split(":")))
            return w / h
        except (ValueError, AssertionError):
            return None

    @property
    def output_width(self) -> int:
        try:
            assert self.aspect
            return round8(
                max(
                    max(
                        self.width * self.aspect,
                        self.height / self.aspect,
                    ),
                    self.width,
                )
            )
        except AssertionError:
            return round8(self.width)

    @property
    def output_height(self) -> int:
        try:
            assert self.aspect
            return round8((self.output_width / self.aspect))
        except AssertionError:
            return round8(self.height)


    def apply_person(self, sex: Sex = None, age: int = None, race: Ethnicity = None):
        parts = []
        prompt = self.prompt
        try:
            assert age
            assert not self.no_age
            parts.append(f"{age} year old")
        except AssertionError:
            pass
        try:
            assert race
            assert not self.no_race
            parts.append(f"{race.value}")
        except AssertionError:
            pass
        try:
            assert sex
            parts.append(sex.label)
            match sex:
                case Sex.M:
                    prompt = (
                        prompt.replace("woman", "")
                        .replace("female", "")
                        .replace("her", "his")
                    )
                case _:
                    pass
        except AssertionError:
            pass
        pattern_suffix = " ".join(parts)
        if pattern_suffix not in prompt:
            self.prompt = f"{prompt}, {pattern_suffix}"


class VariationPipelineParams(BaseModel):
    guidance_scale: float = Field(default=3)
    num_inference_steps: int = Field(default=50)
    num_images_per_prompt: int = Field(default=1)


class OutputParams(BaseModel, arbitrary_types_allowed=True):
    prompt: str = Field(default="")
    height: Optional[int] = None
    width: Optional[int] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    generator: Optional[torch.Generator] = None
    model: Optional[str] = None
    aspect_ratio: Optional[Resolutions] = None
    editing_prompt: Optional[list[str]] = None
    num_images_per_prompt: Optional[int] = None
    cross_attention_kwargs: Optional[dict[str, float]] = None
    target_size: Optional[tuple[int, int]] = None
    prompt_embeds: Optional[Any] = None
    pooled_prompt_embeds: Optional[Any] = None
    negative_prompt_embeds: Optional[Any] = None
    negative_pooled_prompt_embeds: Optional[Any] = None
    strength: Optional[float] = None
    clip_skip: Optional[int] = None
    scale: Optional[float] = None
    template: Optional[str] = None
    style: Optional[str] = None

    def __init__(self, **data):
        args = {
            k: clean_newlines(v) if isinstance(v, str) else v
            for k, v in data.items()
            if v is not None
        }
        super().__init__(**args)

    def to_output(self) -> dict:
        binaries = [
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "generator",
        ]
        return {
            k: v
            for k, v in self.model_dump().items()
            if all([v is not None, k not in binaries])
        }

    def __to_pipe(self, exclude: list[str]):
        res = self.model_dump(
            exclude_none=True, exclude=[*exclude, "template", "style"]
        )
        try:
            res["prompt"] = clean_lora(res["prompt"])
        except Exception:
            pass
        return res

    def to_pipe(self) -> dict:
        return self.__to_pipe(
            [
                "seed",
                "model",
                "aspect",
                "prompt_embeds",
                "negative_prompt_embeds",
                "negative_pooled_prompt_embeds",
                "pooled_prompt_embeds",
                "scale",
            ]
        )

    def to_img2img_pipe(self) -> dict:
        return self.__to_pipe(
            [
                "seed",
                "model",
                "aspect",
                "width",
                "height",
                "prompt",
                "negative_prompt",
                "scale",
            ]
        )

    def to_pipe_xl(self) -> dict:
        return self.__to_pipe(
            [
                "seed",
                "model",
                "aspect",
                "prompt_embeds",
                "negative_prompt_embeds",
                "negative_pooled_prompt_embeds",
                "pooled_prompt_embeds",
                "scale",
                "strength",
            ],
        )
        
    def to_pipe_flux(self) -> dict:
        return self.__to_pipe(
            [
                "seed",
                "model",
                "aspect",
                "prompt_embeds",
                "negative_prompt_embeds",
                "negative_pooled_prompt_embeds",
                "pooled_prompt_embeds",
                "scale",
                "strength",
            ],
        )

    def to_face_pipe(self) -> dict:
        return self.__to_pipe(
            [
                "model",
                "aspect",
                "prompt_embeds",
                "negative_prompt_embeds",
                "negative_pooled_prompt_embeds",
                "pooled_prompt_embeds",
                "generator",
                "strength",
            ],
        )

    def to_pipe_img2img_xl(self) -> dict:
        return self.__to_pipe(
            [
                "seed",
                "model",
                "aspect",
                "width",
                "height",
                "prompt_embeds",
                "negative_prompt_embeds",
                "negative_pooled_prompt_embeds",
                "pooled_prompt_embeds",
                "scale",
            ],
        )

    def to_kandy_pipe(self) -> dict:
        return self.__to_pipe(["seed", "model", "aspect", "scale"])

    def to_kandyimg2img_pipe(self) -> dict:
        return self.__to_pipe(["seed", "model", "aspect", "width", "height"])

    def to_pixart_pipe(self) -> dict:
        return self.__to_pipe(
            [
                "seed",
                "model",
                "aspect",
                "negative_pooled_prompt_embeds",
                "pooled_prompt_embeds",
                "cross_attention_kwargs",
                "scale",
            ]
        )


class ImageResult(BaseModel):
    image: list[Path] = []
    params: Optional[OutputParams] = None
    seed: Optional[int] = None

    @property
    def text(self) -> str:
        try:
            assert self.params
            data = {
                "-g": self.params.guidance_scale,
                "-i": self.params.num_inference_steps,
                "-n": self.params.negative_prompt,
                "-r": self.params.aspect_ratio,
                "-s": self.seed,
                "-m": self.params.model,
                "-st": self.params.strength,
                "-cs": self.params.clip_skip,
                "-sc": self.params.scale,
                "-h": self.params.height,
                "-w": self.params.width,
            }
            options = [f"{k} {v}" for k, v in data.items() if v]
            return f"{self.params.prompt} {' '.join(options)}"
        except AssertionError:
            return ""

    def prompt_to_filename(self) -> str:
        assert self.params
        return "_".join(
            map(
                alphanumcase,
                filter(
                    None,
                    [
                        self.params.style,
                        self.params.template,
                        self.params.model or "default",
                        truncate(self.params.prompt, 20),
                        f"g{self.params.guidance_scale}",
                        f"i{self.params.num_inference_steps}",
                        f"w{self.params.width}h{self.params.height}",
                        f"s{self.seed}",
                        f"{datetime.now().isoformat()}",
                    ],
                ),
            )
        )

    def write_exif(self, extra_exif: dict[str, str] = {}):
        for image in self.image:
            img = Image.open(image.as_posix())
            ex = img.getexif()
            ex[TagNames.ImageDescription] = self.text
            for k, v in extra_exif.items():
                ex[getattr(TagNames, k)] = v
            img.save(image.as_posix(), exif=ex)

    def save_to(self, output_dir: Path) -> list[Path]:
        dst_result = []
        for idx, img in enumerate(self.image):
            dst = output_dir / f"{self.prompt_to_filename()}{idx:02d}{img.suffix}"
            shutil.copy(img.as_posix(), dst.as_posix())
            dst_result.append(dst)
        return dst_result


class Image2ImageResult(ImageResult):
    @property
    def text(self) -> str:
        try:
            assert self.params
            data = {
                "-g": self.params.guidance_scale,
                "-i": self.params.num_inference_steps,
                "-n": self.params.negative_prompt,
                "-r": self.params.aspect_ratio,
                "-st": self.params.strength,
                "-m": self.params.model,
                "-s": self.seed,
                "-cs": self.params.clip_skip,
                "-sc": self.params.scale,
                "-h": self.params.height,
                "-w": self.params.width,
            }

            def reducer(res, k):
                if data[k] is not None:
                    res.append(k)
                    res.append(f"{data[k]}")
                return res

            options = reduce(reducer, data.keys(), [])
            return f"{self.params.prompt} {shlex.join(options)}"
        except AssertionError:
            return ""


class Face2ImageResult(ImageResult):
    @property
    def text(self) -> str:
        try:
            assert self.params
            data = {
                "-g": self.params.guidance_scale,
                "-i": self.params.num_inference_steps,
                "-n": self.params.negative_prompt,
                "-m": self.params.model,
                "-s": self.seed,
                "-cs": self.params.clip_skip,
                "-sc": self.params.scale,
                "-h": self.params.height,
                "-w": self.params.width,
            }

            def reducer(res, k):
                if data[k] is not None:
                    res.append(k)
                    res.append(f"{data[k]}")
                return res

            options = reduce(reducer, data.keys(), [])
            return f"{self.params.prompt} {shlex.join(options)}"
        except AssertionError:
            return ""


class Category(StrEnum):
    MINIMAL = "minimal"
    ABSTRACT = "abstract"
    LANDSCAPE = "landscape"
    CARTOON = "cartoon"
    FANTASY = "fantasy"
    NATURE = "nature"
    HORROR = "horror"
    WHATEVER = "whatever"

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]
    
    @classmethod
    def guess(cls, name: str):
        return extract(name, cls.values())
    


class UploadRequest(BaseModel):
    image_url: str
    category: str
    upscale: int = Field(default=4)


class GeoLocation(BaseModel):
    country: str
    country_iso: str
    city: str
    name: str
    subdivions: Optional[list[str]] = None
    addressLine: Optional[str] = None
    postCode: Optional[str] = None
    location: Optional[list[float]] = None
    extra: Optional[dict[str, str]] = None


class Embedding(BaseModel):
    path: Path
    token: str
