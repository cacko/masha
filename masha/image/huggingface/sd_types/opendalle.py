from importlib_metadata import PathDistribution
from kiwisolver import strength
from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers import (
    UNet2DConditionModel,
    LCMScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.pipelines import (
    DiffusionPipeline,
    AutoPipelineForImage2Image,
)
from compel import Compel, ReturnedEmbeddingsType
import logging
from humanfriendly import format_size
from torch.mps import current_allocated_memory
from masha.image.huggingface.utils import (
    get_lora_models,
    get_ti_models,
    load_image,
    get_compel_prompts_xl,
)
from pathlib import Path
from masha.config import app_config
from masha.image.config import image_config
from rich import print
import logging


class StableDiffusionOpenDalle(BaseStableDiffusion):
    _params = dict(negative_prompt="blurry")

    @property
    def text_inversion_root(self) -> Path:
        return self.__class__.dataRoot / image_config.textual_inversion.root_xl

    def __get_output_params(self, seed) -> OutputParams:
        params = self.params
        prompt = params.output_prompt
        negative_prompt = params.negative_prompt
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = get_compel_prompts_xl(self.pipeline, prompt, negative_prompt)
        return OutputParams(
            prompt=prompt,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps,
            negative_prompt=params.negative_prompt,
            generator=generator,
            width=params.output_width,
            height=params.output_height,
            seed=seed,
            strength=params.strength,
            num_images_per_prompt=params.num_images_per_prompt,
            model=self.__class__.option,
            cross_attention_kwargs=params.cross_attention_kwargs,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=params.clip_skip,
        )

    def get_img2img_result(self, seed, image_path: PathDistribution):
        self.pipeline: DiffusionPipeline = self.img2img_pipe
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_pipe_img2img_xl()
        logging.debug(to_pipe)
        image = load_image(image_path=image_path, size=(1024, 1024))
        result = self.pipeline(
            **to_pipe,
            image=image,
            callback_on_step_end=self.__class__.interrupt_callback,
        )
        return (result, output_params)

    def get_txt2img_result(
        self,
        seed,
    ):
        self.pipeline: AutoPipelineForImage2Image = self.txt2img_pipe
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_pipe_xl()
        logging.debug(to_pipe)
        result = self.pipeline(**to_pipe)
        return (result, output_params)

    def get_img2img_pipeline(self, pipe_args):
        model_path = self.__class__.img2imgModelPath
        logging.info(f"MODEL PATH {model_path}")
        logging.info(f"MEM START - {format_size(current_allocated_memory())}")
        pipe_args = dict(use_safetensors=True, **pipe_args)
        try:
            lcm_path = self.__class__.lcmPath
            assert lcm_path
            logging.info(f"Loading UNET from {lcm_path}")
            unet = UNet2DConditionModel.from_pretrained(
                lcm_path, torch_dtype=torch.float16
            )
            sd_base = AutoPipelineForImage2Image.from_pretrained(
                model_path.as_posix(),
                add_watermarker=False,
                torch_dtype=torch.float16,
                unet=unet,
                **pipe_args,
            )
            sd_base.scheduler = LCMScheduler.from_config(sd_base.scheduler.config)
        except AssertionError:
            if model_path.is_file():
                sd_base = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_path.as_posix(),
                    torch_dtype=torch.float16,
                    **pipe_args,
                )
            else:
                sd_base = AutoPipelineForImage2Image.from_pretrained(
                    model_path.as_posix(),
                    torch_dtype=torch.float16,
                    **pipe_args,
                )
            try:
                assert self.scheduler
                scheduler = self.scheduler.from_config(
                    config=sd_base.scheduler.config,  # type: ignore
                )
                sd_base.scheduler = scheduler  # type: ignore
            except AssertionError:
                pass
        logging.info(f"SCHEDULER {sd_base.scheduler.__class__.__name__}")
        pytorch_pipe = sd_base.to(self.__class__.device)  # type: ignore
        logging.info(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.loadLoraWeights(pytorch_pipe, self.__class__.loraxlPath)
            self.loadTextualInversion(pytorch_pipe)
        except Exception as e:
            logging.exception(e)
            logging.warning("failed")
        logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        return pytorch_pipe

    @torch.no_grad()
    def get_text2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        logging.info(f"MEM START - {format_size(current_allocated_memory())}")
        sd_base = DiffusionPipeline.from_pretrained(
            model_path.as_posix(),
            add_watermarker=False,
            torch_dtype=torch.float16,
            **pipe_args,
        )
        logging.info(f"SCHEDULER {sd_base.scheduler.__class__.__name__}")
        pytorch_pipe: StableDiffusionXLPipeline = sd_base.to(self.__class__.device)  # type: ignore
        logging.info(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.loadLoraWeights(pytorch_pipe, self.__class__.loraxlPath)
            self.loadTextualInversion(pytorch_pipe)
        except Exception as e:
            logging.exception(e)
            logging.warning("failed")
        logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        return pytorch_pipe

    @torch.no_grad()
    def loadLoraWeights(self, sd_pipe: AutoPipelineForImage2Image, lora_path: Path):
        try:
            assert self.params
            loras = list(get_lora_models(self.params.prompt, lora_path))
            assert len(loras)
            adapter_names = []
            adapter_weights = []
            for lora, wgt in loras:
                logging.info(f">> LOADING {lora.as_posix()}")
                adapter_name = lora.stem
                weights_file = lora.as_posix()
                sd_pipe.load_lora_weights(
                    ".", weight_name=weights_file, adapter_name=adapter_name
                )
                adapter_names.append(f"adapter_{adapter_name}")
                adapter_weights.append(wgt)
            sd_pipe.set_adapters(
                adapter_names=adapter_names, adapter_weights=adapter_weights
            )
            logging.info(sd_pipe.get_list_adapters())
            logging.info("...LOADED")
            sd_pipe.fuse_lora(adapter_names=adapter_names)
        except AssertionError:
            pass

    @torch.no_grad()
    def loadTextualInversion(
        self,
        sd_pipe: DiffusionPipeline,
    ):
        try:
            txt = f"{self.params.prompt} {self.params.negative_prompt}"
            models = list(get_ti_models(txt, self.text_inversion_root))
            if not models:
                return
            assert hasattr(sd_pipe, "load_textual_inversion")
            logging.info("LOADING TEXTUAL INVERSIONS")
            sd_pipe.load_textual_inversion(
                [m[0].as_posix() for m in models], local_files_only=True
            )
        except AssertionError:
            pass
