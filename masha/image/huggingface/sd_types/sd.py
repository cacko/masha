from re import template
import cv2
import rich
from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    DDIMScheduler,
)
import logging
from humanfriendly import format_size
from torch.mps import current_allocated_memory
from masha.image.huggingface.utils import (
    load_image,
    get_compel_prompts,
)
from pathlib import Path
from masha.image.huggingface.lora.sd_loaders import LoadersSDMixin
from rich import print
from masha.image.config import image_config
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from insightface.utils import face_align


class StableDiffusionSD(BaseStableDiffusion, LoadersSDMixin):
    pipeline: StableDiffusionPipeline

    @property
    def text_inversion_root(self) -> Path:
        return self.__class__.dataRoot / image_config.textual_inversion.root

    @property
    def lora_path(self) -> Path:
        return self.__class__.loraPath

    @property
    def ip_adapter_model_path(self) -> Path:
        return (
            self.__class__.dataRoot
            / "IP-Adapter-FaceID"
            / "ip-adapter-faceid-plus_sd15.bin"
        )

    @property
    def vae_model_path(self) -> Path:
        return self.__class__.dataRoot / "sd-vae-ft-mse"

    @property
    def ip_adapter_encoder_path(self) -> Path:
        return self.__class__.dataRoot / "CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __get_output_params(self, seed, use_compel=False) -> OutputParams:
        params = self.params
        prompt = params.output_prompt
        negative_prompt = params.negative_prompt
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        prompt_embeds, negative_prompt_embeds = (
            (None, None)
            if not use_compel
            else get_compel_prompts(self.pipeline, prompt, negative_prompt)
        )
        return OutputParams(
            prompt=prompt,
            height=params.output_height,
            width=params.output_width,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps,
            negative_prompt=params.negative_prompt,
            generator=generator,
            seed=seed,
            editing_prompt=params.editing_prompt,
            num_images_per_prompt=params.num_images_per_prompt,
            model=self.__class__.option,
            cross_attention_kwargs=params.cross_attention_kwargs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            strength=params.strength,
            clip_skip=params.clip_skip,
            scale=params.scale,
            template=params.template,
            style=params.style,
        )

    def get_face2img_result(self, seed, faceid_embeds, **kwds):
        self.init_face2img_pipe()
        output_params = self.__get_output_params(seed)
        ip_model = IPAdapterFaceIDPlus(
            sd_pipe=self.pipeline,
            image_encoder_path=self.ip_adapter_encoder_path.as_posix(),
            ip_ckpt=self.ip_adapter_model_path.as_posix(),
            device=self.__class__.device,
        )
        image = cv2.imread(kwds.get("face_path").as_posix())
        face_image = cv2.resize(image, (224, 224))
        images = ip_model.generate(
            faceid_embeds=faceid_embeds,
            face_image=face_image,
            num_samples=output_params.num_images_per_prompt,
            **output_params.to_face_pipe(),
        )
        return (images, output_params)

    def set_face2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        assert model_path.is_file()
        self.pipeline = StableDiffusionPipeline.from_single_file(
            model_path.as_posix(),
            torch_dtype=torch.float16,
            feature_extractor=None,
            safety_checker=None,
            **pipe_args,
        ).to(self.__class__.device)
        assert self.pipeline
        logging.info(f">> PIPELINE: {self.pipeline.__class__.__name__}")
        try:
            assert self.scheduler
            scheduler = self.scheduler.from_config(
                config=self.pipeline.scheduler.config, **self.scheduler_args
            )
            self.pipeline.scheduler = scheduler
        except AssertionError:
            pass
        logging.info(f">> SCHDULER {self.pipeline.scheduler.__class__.__name__}")
        logging.info(self.pipeline.scheduler.config)
        logging.debug(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.loadLoraWeights()
            self.loadTextualInversion()
            logging.debug(f"MEM LORA - {format_size(current_allocated_memory())}")
        except Exception as e:
            logging.exception(e)
            logging.warning("failed")

    def get_txt2img_result(self, seed):
        self.init_txt2img_pipe()
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_pipe()
        print(output_params.to_output())
        result = self.pipeline(**to_pipe)
        return (result, output_params)

    def set_text2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        assert model_path
        assert model_path.is_file()
        self.pipeline = StableDiffusionPipeline.from_single_file(
            model_path.as_posix(),
            use_safetensors=True,
            torch_dtype=torch.float16,
            **pipe_args,
        ).to(self.__class__.device)
        try:
            assert self.scheduler
            scheduler = self.scheduler.from_config(
                config=self.pipeline.scheduler.config, **self.scheduler_args
            )
            self.pipeline.scheduler = scheduler  # type: ignore
        except AssertionError:
            pass
        logging.info(f"SCHEDULER {self.pipeline.scheduler}")
        try:
            self.loadLoraWeights()
            self.loadTextualInversion()
        except Exception as e:
            logging.exception(e)
            logging.info("failed")
        logging.debug(f"Memory allocated - {format_size(current_allocated_memory())}")

    def get_img2img_result(self, seed, image_path):
        self.init_img2img_pipe()
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_img2img_pipe()
        print(output_params.to_output())
        to_pipe["image"] = load_image(image_path, size=(768, 768))
        result = self.pipeline(**to_pipe)
        return (result, output_params)

    def set_img2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        assert model_path
        pipe_args = dict(use_safetensors=True, **pipe_args)
        assert model_path.is_file()
        self.pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
            model_path.as_posix(),
            torch_dtype=torch.float16,
            **pipe_args,
        )
        try:
            assert self.scheduler
            scheduler = self.scheduler.from_config(
                config=self.pipeline.scheduler.config, **self.scheduler_args
            )
            self.pipeline.scheduler = scheduler
        except AssertionError:
            pass
        self.pipeline.to(self.__class__.device)
        logging.debug(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.loadLoraWeights()
            self.loadTextualInversion()
        except Exception as e:
            logging.exception(e)
            logging.info("failed")
        logging.info(f"SCHEDULER {self.pipeline.scheduler.__class__.__name__}")
        logging.debug(f"SCHEDULER {self.pipeline.scheduler}")
        logging.debug(f"MEM LORA - {format_size(current_allocated_memory())}")
