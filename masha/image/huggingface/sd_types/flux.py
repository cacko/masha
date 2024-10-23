import cv2
import rich
from masha.image.huggingface.lora.flux_loaders import LoadersFluxMixin
from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from mflux import Config, Flux1, ModelConfig
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
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL


class StableDiffusionFlux(BaseStableDiffusion, LoadersFluxMixin):
    _params = dict(negative_prompt="blurry")

    @property
    def text_inversion_root(self) -> Path:
        return self.__class__.dataRoot / image_config.textual_inversion.root_xl

    @property
    def lora_path(self) -> Path:
        return self.__class__.lorafluxPath

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
            pag_scale=params.pag_scale,
        )



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
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
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
        self.pipeline = self.img2img_pipe
        output_params = self.__get_output_params(seed, no_compel=True)
        image = self.pipeline.generate_image(
            seed=output_params.seed,
            prompt=output_params.prompt,
            config=Config(
                num_inference_steps=output_params.num_inference_steps,
                height=output_params.height,
                width=output_params.width,
                guidance=output_params.guidance_scale,
                init_image_path=image_path,
                init_image_strength=output_params.strength
            ),
        )
        self.pipeline = None
        return (FluxPipelineOutput(images=[image.image]), output_params)

    def get_img2img_pipeline(self, pipe_args):
        model_path = self.__class__.img2imgModelPath
        params = dict(
            model_config=ModelConfig.from_alias(model_path.name),
            quantize=8,
        )
        try:
            paths, scales = self.loadLoraWeights()
            params.update(dict(lora_paths=paths, lora_scales=scales))
        except AssertionError as e:
            logging.exception(e)
        except Exception as e:
            logging.error(e)

        flux = Flux1(**params)
        self.pipeline = flux
        return self.pipeline

    def get_txt2img_result(
        self,
        seed,
    ):
        self.pipeline: Flux1 = self.txt2img_pipe
        output_params = self.__get_output_params(seed, no_compel=True)
        to_pipe = output_params.to_pipe_flux()
        image = self.pipeline.generate_image(
            seed=output_params.seed,
            prompt=output_params.prompt,
            config=Config(
                num_inference_steps=output_params.num_inference_steps,
                height=output_params.height,
                width=output_params.width,
                guidance=output_params.guidance_scale,
            ),
        )
        self.pipeline = None
        return (FluxPipelineOutput(images=[image.image]), output_params)

    def get_text2img_pipeline(self, pipe_args) -> Flux1:
        model_path = self.__class__.modelPath
        params = dict(
            model_config=ModelConfig.from_alias(model_path.name),
            quantize=8,
        )
        try:
            paths, scales = self.loadLoraWeights()
            params.update(dict(lora_paths=paths, lora_scales=scales))
        except AssertionError as e:
            logging.exception(e)
        except Exception as e:
            logging.error(e)

        flux = Flux1(**params)
        self.pipeline = flux
        return self.pipeline
