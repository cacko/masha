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
from masha.image.huggingface.lora.sd_loaders import (
    loadLoraWeights,
    loadTextualInversion,
)
from rich import print
from masha.image.config import image_config
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from insightface.utils import face_align


class StableDiffusionSD(BaseStableDiffusion):
    pipeline: StableDiffusionPipeline

    @property
    def text_inversion_root(self) -> Path:
        return self.__class__.dataRoot / image_config.textual_inversion.root

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

    def __get_output_params(self, seed, no_compel=False) -> OutputParams:
        params = self.params
        prompt = params.output_prompt
        negative_prompt = params.negative_prompt
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        prompt_embeds, negative_prompt_embeds = (
            (None, None)
            if no_compel
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
        self.pipeline = self.face2img_pipe
        output_params = self.__get_output_params(seed, no_compel=True)
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
        ip_model = None
        self.pipeline = None
        return (images, output_params)

    def get_face2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        assert model_path.is_file()
        pipe = StableDiffusionPipeline.from_single_file(
            model_path.as_posix(),
            torch_dtype=torch.float16,
            feature_extractor=None,
            safety_checker=None,
            **pipe_args,
        )
        assert pipe
        logging.info(f">> PIPELINE: {pipe.__class__.__name__}")
        try:
            assert self.scheduler
            scheduler = self.scheduler.from_config(
                config=pipe.scheduler.config, **self.scheduler_args
            )
            pipe.scheduler = scheduler
        except AssertionError:
            pass
        logging.info(f">> SCHDULER {pipe.scheduler.__class__.__name__}")
        logging.info(pipe.scheduler.config)
        self.pipeline = pipe.to(self.__class__.device)
        pipe = None
        logging.info(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.pipeline = self.loadLoraWeights()
            self.pipeline = self.loadTextualInversion()
            logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        except Exception as e:
            logging.exception(e)
            logging.warning("failed")
        return self.pipeline

    def get_txt2img_result(self, seed):
        self.pipeline = self.txt2img_pipe
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_pipe()
        print(output_params.to_output())
        result = self.pipeline(**to_pipe)
        return (result, output_params)

    def get_text2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        assert model_path
        assert model_path.is_file()
        sd_pipe = StableDiffusionPipeline.from_single_file(
            model_path.as_posix(),
            use_safetensors=True,
            torch_dtype=torch.float16,
            **pipe_args,
        )
        try:
            assert self.scheduler
            scheduler = self.scheduler.from_config(
                config=sd_pipe.scheduler.config, **self.scheduler_args
            )
            sd_pipe.scheduler = scheduler  # type: ignore
        except AssertionError:
            pass
        logging.info(f"SCHEDULER {sd_pipe.scheduler}")
        self.pipeline = sd_pipe.to(self.__class__.device)  # type: ignore
        sd_pipe = None
        try:
            self.pipeline = self.loadLoraWeights()
            self.pipeline = self.loadTextualInversion()
        except Exception as e:
            logging.exception(e)
            logging.info("failed")
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")
        return self.pipeline

    def get_img2img_result(self, seed, image_path):
        self.pipeline = self.img2img_pipe
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_img2img_pipe()
        print(output_params.to_output())
        to_pipe["image"] = load_image(image_path, size=(768, 768))
        result = self.pipeline(**to_pipe)
        return (result, output_params)

    def get_img2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        logging.info(f"MEM START - {format_size(current_allocated_memory())}")
        assert model_path
        pipe_args = dict(use_safetensors=True, **pipe_args)
        assert model_path.is_file()
        sd_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            model_path.as_posix(),
            torch_dtype=torch.float16,
            **pipe_args,
        )
        try:
            assert self.scheduler
            scheduler = self.scheduler.from_config(
                config=sd_pipe.scheduler.config, **self.scheduler_args
            )
            sd_pipe.scheduler = scheduler
        except AssertionError:
            pass
        self.pipeline = sd_pipe.to(self.__class__.device)
        sd_pipe = None
        logging.info(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.pipeline = self.loadLoraWeights()
            self.pipeline = self.loadTextualInversion()
        except Exception as e:
            logging.exception(e)
            logging.info("failed")
        logging.info(f"SCHEDULER {self.pipeline.scheduler.__class__.__name__}")
        logging.debug(f"SCHEDULER {self.pipeline.scheduler}")
        logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        return self.pipeline

    def loadLoraWeights(self):
        self.pipeline = loadLoraWeights(
            pipeline=self.pipeline,
            prompt=self.params.prompt,
            lora_path=self.__class__.loraPath,
        )
        return self.pipeline

    def loadTextualInversion(
        self,
    ):
        self.pipeline = loadTextualInversion(
            pipeline=self.pipeline,
            prompt=self.params.prompt,
            negative_prompt=self.params.negative_prompt,
            text_inversion_root=self.text_inversion_root,
        )
        return self.pipeline
