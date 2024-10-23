from enum import StrEnum
from pathlib import Path
from typing import Any, Optional, ClassVar
from pydantic import BaseModel, Field
from corefile import filepath
import yaml
from masha.config import app_config
import re

from masha.image.models import IncompatibleTemplate, SDModelType, Sex
from masha.name.classifiers import Ethnicity

OBJECT_PATTERN = re.compile(r"(/([^/]+)/)", re.DOTALL)

GENERATED_PATH = Path(app_config.cache.generated_root)


p = "/Volumes/Devo/Code/masha/masha/image/config.yaml"


class ApplyToParamsMixin(object):

    def apply(self, input_params: dict):
        return input_params


class LoraConfig(BaseModel):
    root: str
    root_xl: str
    root_3: str
    root_flux: str


class TIConfig(BaseModel):
    root: str
    root_2: str
    root_xl: str
    root_3: str


class ClassifyConfig(BaseModel):
    model: str
    tokenizer: str
    attraction_model: str
    age_model: str
    ethnicity_model: str
    gender_model: str


class AgenConfig(BaseModel):
    age_model: str
    age_proto: str
    root_path: Path


class GenderConfig(BaseModel):
    gender_model: str
    gender_proto: str
    root_path: Path


class VariationConfig(BaseModel):
    model: str


class UpscaleConfig(BaseModel):
    root: str


class ParamsConfig(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    negative_prompt: Optional[str] = None
    guidance_scale: Optional[str | float] = None
    num_inference_steps: Optional[str | int] = None
    clip_skip: Optional[int] = None
    strength: Optional[float] = None
    aspect_ratio: Optional[str] = None
    scale: Optional[float] = None
    seed: Optional[int] = None
    person: Optional[bool] = None
    upscale: Optional[int] = None


class SDConfig(BaseModel, arbitrary_types_allowed=True):
    name: str
    model: str
    sd_type: SDModelType
    option: str
    category: str
    lcm: Optional[str] = None
    coreml: Optional[bool] = Field(default=False)
    scheduler_class: Optional[str] = None
    scheduler_args: Optional[dict[str, any]] = Field(default={})
    custom_params: ParamsConfig = ParamsConfig()
    img2img_model: Optional[str] = None

    def __init__(self, **kwds):
        kwds.setdefault("model", kwds.get("name"))
        super().__init__(**kwds)


class TemplateConfig(BaseModel, ApplyToParamsMixin):
    name: str
    category: str
    prompt: Optional[str] = Field(default="")
    models: Optional[list[str]] = None
    custom_params: ParamsConfig = ParamsConfig()
    scheduler_class: Optional[str] = None
    scheduler_args: Optional[dict[str, Any]] = None
    categories: ClassVar[set[str]] = set()
    categories_enum: StrEnum = None
    for_gender: Optional[Sex] = None
    no_race: Optional[bool] = None
    no_age: Optional[bool] = None

    @staticmethod
    def add_categories(values: list[str]):
        TemplateConfig.categories.update(map(str.strip, values))

    def __init__(self, **kwds):
        TemplateConfig.add_categories(kwds.get("category").split(","))
        TemplateConfig.categories_enum = StrEnum(
            "categories", list(TemplateConfig.categories)
        )
        super().__init__(**kwds)

    @classmethod
    def get_categories(cls) -> list[str]:
        return TemplateConfig.categories

    def in_category(self, category: str) -> bool:
        cats = map(str.strip, self.category.lower().split(","))
        return category.lower() in list(cats)

    def apply(
        self,
        input_params: dict = {},
        sex: Optional[Sex] = None,
        age: Optional[int] = None,
        race: Optional[Ethnicity] = None,
    ):
        params = {k: v for k, v in input_params.items() if v is not None}
        prompt = params.get("prompt", "")
        pattern_suffix = []
        try:
            apply_person = next(
                filter(
                    lambda x: x is not None,
                    [params.get("person", None), self.custom_params.person, False],
                )
            )
            assert apply_person
            try:
                assert not self.no_age
                assert not params.get("no_age")
                assert age
                pattern_suffix.append(f"{age} year old")
            except AssertionError:
                pass
            try:
                assert not self.no_race
                assert not params.get("no_race")
                assert race
                pattern_suffix.append(race.value)
            except AssertionError:
                pass
            try:
                assert sex
                if self.for_gender and self.for_gender != sex:
                    raise IncompatibleTemplate(f"Not compatible with {sex}")
                pattern_suffix.append(sex.label)
            except AssertionError:
                pass

            pattern_suffix = " ".join(pattern_suffix)
            assert pattern_suffix not in prompt
        except AssertionError:
            pattern_suffix = ""
        try:
            assert "//" not in self.prompt
            self.prompt += ",//"
        except AssertionError:
            pass
        try:
            m = OBJECT_PATTERN.match(prompt)
            assert m
            tpl = self.prompt.replace("//", f"{pattern_suffix} {m.group(2)}")
            prompt = prompt.replace(m.group(1), tpl)
        except AssertionError:
            prompt = self.prompt.replace("//", f"{pattern_suffix} {prompt}")
        match sex:
            case Sex.M:
                prompt = prompt.replace("woman", "").replace("her", "his")
        custom_params = (
            self.custom_params.model_dump(exclude_none=True, exclude=["prompt"])
            if self.custom_params
            else {}
        )
        return {
            **custom_params,
            **params,
            **dict(prompt=prompt.strip(), person=False, template=self.name),
        }


class StyleConfig(BaseModel, ApplyToParamsMixin):
    name: str
    prompt: str
    model: str
    guidance_scale: float = Field(default=20)
    num_inference_steps: int = Field(default=20)
    width: int = Field(default=640)
    height: int = Field(default=640)
    strength: float = Field(default=0.45)
    upscale: int = Field(default=4)
    negative_prompt: Optional[str] = None
    clip_skip: Optional[int] = None
    style_options: ClassVar[set[str]] = set()
    styles_enum: StrEnum = None

    def __init__(self, **kwds):
        super().__init__(**kwds)
        StyleConfig.add_styles([kwds.get("name")])
        StyleConfig.styles_enum = StrEnum("styles", StyleConfig.get_styles())

    @staticmethod
    def add_styles(values: list[str]):
        StyleConfig.style_options.update(map(str.strip, values))

    @classmethod
    def get_styles(cls) -> list[str]:
        return list(StyleConfig.style_options)

    def apply(self, input_params: dict):
        params = input_params.copy()
        params["prompt"] = f"{params.get('prompt', '')} {self.prompt}".strip()
        return {
            **self.model_dump(exclude=["model", "name"], exclude_none=True),
            **{k: v for k, v in params.items() if v is not None},
            **dict(style=self.name),
        }


class Image2TextConfig(BaseModel):
    model: str
    tokenizer: str


class QRCodeConfig(BaseModel):
    name: str
    control_net: str
    model: str = Field(default="tvxl")
    controlnet_conditioning_scale: Optional[float] = Field(default=1.2)
    guidance_scale: Optional[float] = Field(default=5)
    prompt: Optional[str] = Field(default="")
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20)
    seed: Optional[int] = None
    strength: Optional[float] = Field(default=0.8)
    scheduler_class: Optional[str] = None


class PromptConfig(BaseModel):
    name: str
    model: str
    tokenizer: str


class EmbeddingConfig(BaseModel):
    name: str
    token: list[str]


class Face2ImgConfig(BaseModel):
    default_model: str
    face_store: Path


class Img2CaptionConfig(BaseModel):
    model: str


class DeepfaceConfig(BaseModel):
    age: str
    race: str
    
class StreetViewConfig(BaseModel):
    local_url: str
    url: str


class ImageConfig(BaseModel):
    classify: ClassifyConfig
    variation: VariationConfig
    txt2img: list[SDConfig]
    templates: list[TemplateConfig]
    qrcode: list[QRCodeConfig]
    prompts: list[PromptConfig]
    upscale: UpscaleConfig
    lora: LoraConfig
    textual_inversion: TIConfig
    styles: list[StyleConfig]
    embeddings: list[EmbeddingConfig]
    face2img: Face2ImgConfig
    age: AgenConfig
    gender: GenderConfig
    img2caption: Img2CaptionConfig
    deepface: DeepfaceConfig
    image2text: Image2TextConfig = None
    streetview: StreetViewConfig

    def get_sd_config(self, name: str) -> Optional[SDConfig]:
        try:
            assert self.txt2img
            return next(filter(lambda t: t.name == name, self.txt2img), None)
        except AssertionError:
            pass
        return None

    def get_template(self, name: str) -> Optional[TemplateConfig]:
        try:
            assert self.templates
            return next(filter(lambda t: t.name == name, self.templates), None)
        except AssertionError:
            pass
        return None

    def get_template_category(self, name: str) -> Optional[list[TemplateConfig]]:
        try:
            assert self.templates
            return list(filter(lambda c: c.in_category(name), self.templates))
        except AssertionError:
            pass
        return None

    def options_for_category(cls, category: str) -> list[str]:
        return list(
            map(
                lambda c: c.option,
                filter(lambda cfg: category in cfg.category, cls.configs.values()),
            )
        )

    def get_qrcode(self, name: str) -> Optional[QRCodeConfig]:
        try:
            assert self.qrcode
            return next(filter(lambda t: t.name == name, self.qrcode), None)
        except AssertionError:
            pass
        return None

    def get_style(self, name: str) -> Optional[StyleConfig]:
        try:
            assert self.styles
            return next(filter(lambda st: st.name == name, self.styles), None)
        except AssertionError:
            pass
        return None

    @property
    def style_options(self) -> list[str]:
        return [st.name for st in self.styles]


config_root = Path(__file__).parent / "config"
data = yaml.full_load(Path(p).read_text())
for fp in filepath(config_root, suffixes=[".yaml", ".yml"]):
    node = fp.stem
    dc = yaml.full_load(fp.read_text())
    data[node] = [dict(name=k, **items) for k, items in dc.items()]

image_config = ImageConfig(**data)
