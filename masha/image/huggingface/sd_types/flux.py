import cv2
import rich
from masha.core.image import get_width_height
from masha.image.huggingface.lora.mflux_loaders import LoadersMFluxMixin
from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers import FluxPipeline, FluxKontextPipeline, FluxImg2ImgPipeline
import logging
from masha.image.huggingface.utils import (
    get_compel_prompts_xl,
)
from pathlib import Path
from masha.image.config import image_config
from masha.image.huggingface.utils import load_image
import logging


class StableDiffusionFlux(BaseStableDiffusion, LoadersMFluxMixin):
    _params = dict(negative_prompt="blurry")

    @property
    def text_inversion_root(self) -> Path:
        return self.__class__.dataRoot / image_config.textual_inversion.root_xl

    @property
    def lora_path(self) -> Path:
        return self.__class__.lorafluxPath


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

    def get_face2img_result(self, seed, faceid_embeds, **kwds):
        self.init_face2img_pipe()
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
        return (images, output_params)

    def set_face2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        # assert model_path.is_file()
        pipe = FluxPipeline.from_pretrained(
            FLUX_MODELS.LITE,
            add_watermarker=False,
            torch_dtype=torch.bfloat16,
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
        self.pipeline: FluxPipeline = pipe.to(self.__class__.device)
        pipe = None
        # logging.info(f"MEM PIPE - {format_size(current_allocated_memory())}")
        # try:
        #     self.pipeline = self.loadLoraWeights()
        #     self.pipeline = self.loadTextualInversion()
        #     logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        # except Exception as e:
        #     logging.exception(e)
        #     logging.warning("failed")
        return self.pipeline

    def get_img2img_result(self, seed, image_path: Path):
        self.init_img2img_pipe()
        output_params = self.__get_output_params(seed, no_compel=True)
        width, height = get_width_height(
            image_path, max(output_params.height, output_params.width)
        )
        result = self.pipeline(
            image=load_image(image_path, (output_params.width, output_params.height)),
            prompt=output_params.prompt,
            guidance_scale=output_params.guidance_scale,
            width=width,
            height=height,
        )
        return (result, output_params)

    def set_img2img_pipeline(self, pipe_args) -> FluxKontextPipeline|FluxImg2ImgPipeline:
        model_path = self.__class__.img2imgModelPath
        if 'kontext' in str(model_path):
            self.pipeline = FluxKontextPipeline.from_pretrained(f"{model_path}", torch_dtype=torch.bfloat16)
        else:
            self.pipeline = FluxImg2ImgPipeline.from_pretrained(f"{model_path}", torch_dtype=torch.bfloat16)
        self.pipeline.enable_model_cpu_offload()
        try:
            self.loadLoraWeights()
        except AssertionError as e:
            pass
        except Exception as e:
            logging.error(e)
        return self.pipeline

    def get_txt2img_result(
        self,
        seed,
    ):
        self.init_txt2img_pipe()
        output_params = self.__get_output_params(seed, no_compel=True)
        to_pipe = output_params.to_pipe_flux()
        result = self.pipeline(
            **to_pipe,
            callback_on_step_end=self.__class__.interrupt_callback,
        )
        return (result, output_params)

    def set_text2img_pipeline(self, pipe_args) -> FluxPipeline:
        model_path = self.__class__.modelPath
        self.pipeline = FluxPipeline.from_pretrained(f"{model_path}", torch_dtype=torch.bfloat16)
        self.pipeline.enable_model_cpu_offload()
        try:
            self.loadLoraWeights()
        except AssertionError:
            pass
        except Exception as e:
            logging.error(e)
        return self.pipeline
