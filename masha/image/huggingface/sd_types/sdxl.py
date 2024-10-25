import cv2
import rich
from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.pipelines import (
    StableDiffusionXLPipeline,
)
import logging
from humanfriendly import format_size
from torch.mps import current_allocated_memory
from masha.image.huggingface.utils import (
    load_image,
    get_compel_prompts_xl,
)
from pathlib import Path
from masha.image.config import image_config
import logging
from masha.image.huggingface.lora.sdxl_loaders import LoadersSDXLMixin
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL


class StableDiffusionSDXL(BaseStableDiffusion, LoadersSDXLMixin):
    _params = dict(negative_prompt="blurry")

    @property
    def text_inversion_root(self) -> Path:
        return self.__class__.dataRoot / image_config.textual_inversion.root_xl

    @property
    def lora_path(self) -> Path:
        return self.__class__.loraxlPath

    @property
    def ip_adapter_faceid_model_path(self) -> Path:
        return (
            self.__class__.dataRoot
            / "IP-Adapter-FaceID"
            / "ip-adapter-faceid-plusv2_sdxl.bin"
        )

    @property
    def ip_adapter_faceid_encoder_path(self) -> Path:
        return self.__class__.dataRoot / "CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __get_output_params(self, seed, no_compel=False) -> OutputParams:
        params = self.params
        prompt = params.output_prompt
        negative_prompt = params.negative_prompt
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = (
            (None, None, None, None)
            if no_compel
            else get_compel_prompts_xl(
                self.txt2img_pipe, prompt, negative_prompt=negative_prompt
            )
        )

        return OutputParams(
            prompt=prompt,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps,
            negative_prompt=negative_prompt,
            generator=generator,
            width=params.output_width,
            height=params.output_height,
            seed=seed,
            num_images_per_prompt=params.num_images_per_prompt,
            model=self.__class__.option,
            cross_attention_kwargs=params.cross_attention_kwargs,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=params.clip_skip,
            strength=params.strength,
            scale=params.scale,
        )

    def get_img2img_result(self, seed, image_path: Path):
        self.pipeline: StableDiffusionXLImg2ImgPipeline = self.img2img_pipe
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_pipe_img2img_xl()
        image = load_image(image_path=image_path, size=(1024, 1024))
        result = self.pipeline(
            **to_pipe,
            image=image,
            callback_on_step_end=self.__class__.interrupt_callback,
        )
        self.pipeline = None
        return (result, output_params)

    def get_txt2img_result(
        self,
        seed,
    ):
        self.pipeline = self.txt2img_pipe
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_pipe_xl()
        result = self.pipeline(
            **to_pipe,
            callback_on_step_end=self.__class__.interrupt_callback,
        )
        self.pipeline = None
        return (result, output_params)

    def get_face2img_result(self, seed, faceid_embeds, **kwds):
        self.pipeline = self.face2img_pipe
        output_params = self.__get_output_params(seed, no_compel=True)
        ip_model = IPAdapterFaceIDPlusXL(
            sd_pipe=self.pipeline,
            image_encoder_path=self.ip_adapter_faceid_encoder_path,
            ip_ckpt=self.ip_adapter_faceid_model_path,
            device=self.__class__.device,
            torch_dtype=torch.float16,
        )
        image = cv2.imread(kwds.get("face_path").as_posix())
        face_image = cv2.resize(image, (256, 256))
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
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path.as_posix(),
            add_watermarker=False,
            torch_dtype=torch.float16,
            **pipe_args,
        )
        rich.print(pipe.scheduler.config)
        assert pipe
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
        self.pipeline: StableDiffusionXLPipeline = pipe.to(self.__class__.device)
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

    def get_img2img_pipeline(self, pipe_args):
        model_path = self.__class__.img2imgModelPath
        logging.info(f"MODEL PATH {model_path}")
        logging.info(f"MEM START - {format_size(current_allocated_memory())}")
        pipe_args = dict(use_safetensors=True, **pipe_args)
        assert model_path.is_file()
        sd_base = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path.as_posix(),
            torch_dtype=torch.float16,
            **pipe_args,
        )
        try:
            assert self.scheduler
            scheduler = self.scheduler.from_config(
                config=sd_base.scheduler.config, **self.scheduler_args
            )
            sd_base.scheduler = scheduler  # type: ignore
        except AssertionError:
            pass
        logging.info(f"SCHEDULER {sd_base.scheduler.__class__.__name__}")
        self.pipeline = sd_base.to(self.__class__.device)
        sd_base = None
        logging.info(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.pipeline = self.loadLoraWeights()
            self.pipeline = self.loadTextualInversion()
            logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        except Exception as e:
            logging.exception(e)
            logging.warning("failed")
        return self.pipeline

    def get_text2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        sd_base = None
        logging.info(f"MODEL PATH {model_path}")
        logging.info(f"MEM START - {format_size(current_allocated_memory())}")
        assert model_path.is_file()
        sd_base = StableDiffusionXLPipeline.from_single_file(
            model_path.as_posix(),
            add_watermarker=False,
            torch_dtype=torch.float16,
            **pipe_args,
        )
        sd_base.scheduler = self.scheduler.from_config(
            sd_base.scheduler.config, **self.scheduler_args
        )
        logging.info(f"SCHEDULER {sd_base.scheduler.__class__.__name__}")
        logging.info(sd_base.scheduler.config)
        self.pipeline = sd_base.to(self.__class__.device)
        sd_base = None
        try:
            self.pipeline = self.loadLoraWeights()
            self.pipeline = self.loadTextualInversion()
            logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        except Exception as e:
            logging.exception(e)
            logging.warning("failed")
        return self.pipeline
