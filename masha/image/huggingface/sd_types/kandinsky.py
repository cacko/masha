from kiwisolver import strength
from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import logging
import pprint
from humanfriendly import format_size
from torch.mps import current_allocated_memory
from masha.image.huggingface.utils import load_image


class StableDiffusionKANDINSKY(BaseStableDiffusion):
    @torch.no_grad()
    def get_txt2img_result(self, seed):
        params = self.params
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        prompt = params.output_prompt
        output_params = OutputParams(
            prompt=prompt,
            height=params.output_height,
            width=params.output_width,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps // params.strength,
            negative_prompt=params.negative_prompt,
            generator=generator,
            seed=seed,
            editing_prompt=params.editing_prompt,
            num_images_per_prompt=params.num_images_per_prompt,
            model=self.__class__.option,
        )
        to_pipe = output_params.to_kandy_pipe()
        logging.info(pprint.pformat(output_params.to_output()))
        self.pipeline = self.txt2img_pipe
        result = self.pipeline(**to_pipe)
        self.pipeline = None
        return (result, output_params)

    @torch.no_grad()
    def get_img2img_result(self, seed, image_path):
        params = self.params
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        prompt = params.output_prompt
        output_params = OutputParams(
            prompt=prompt,
            height=params.output_height,
            width=params.output_width,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps // params.strength,
            negative_prompt=params.negative_prompt,
            generator=generator,
            strength=params.strength,
            editing_prompt=params.editing_prompt,
            num_images_per_prompt=params.num_images_per_prompt,
            model=self.__class__.option,
        )
        to_pipe = output_params.to_kandy_pipe()
        logging.info(pprint.pformat(output_params.to_output()))
        self.pipeline = self.img2img_pipe
        result = self.pipeline(
            **to_pipe,
            image=load_image(image_path, size=(512, 512)),
        )
        self.pipeline = None
        return (result, output_params)

    @torch.no_grad()
    def get_text2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        sd_pipe = AutoPipelineForText2Image.from_pretrained(
            model_path.as_posix(), torch_dtype=torch.float16, **pipe_args
        )
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")
        return sd_pipe.to(self.__class__.device) 

    @torch.no_grad()
    def get_img2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        sd_pipe = AutoPipelineForImage2Image.from_pretrained(
            model_path.as_posix(), torch_dtype=torch.float16, **pipe_args
        )
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")
        return sd_pipe.to(self.__class__.device)
