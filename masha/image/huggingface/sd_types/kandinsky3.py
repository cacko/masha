from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import logging
import pprint
from humanfriendly import format_size
from torch.mps import current_allocated_memory
from masha.image.huggingface.utils import load_image


class StableDiffusionKANDINSKY3(BaseStableDiffusion):
    def get_txt2img_result(self, seed):
        self.init_txt2img_pipe()
        params = self.params
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        prompt = params.output_prompt
        output_params = OutputParams(
            prompt=prompt,
            height=params.output_height,
            width=params.output_width,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps * 2,
            negative_prompt=params.negative_prompt,
            generator=generator,
            seed=seed,
            editing_prompt=params.editing_prompt,
            num_images_per_prompt=params.num_images_per_prompt,
            model=self.__class__.option,
            cross_attention_kwargs=params.cross_attention_kwargs,
        )
        to_pipe = output_params.to_kandy_pipe()
        logging.info(pprint.pformat(output_params.to_output()))
        result = self.pipeline(**to_pipe)
        return (result, output_params)

    def get_text2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_path.as_posix(),
            torch_dtype=torch.float16,
            use_safetensors=True,
            **pipe_args,
        )
        
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")
        self.pipeline.to(self.__class__.device)

    def get_img2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_path.as_posix(),
            use_safetensors=True,
            torch_dtype=torch.float16,
            **pipe_args,
        )
    
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")
        self.pipeline.to(self.__class__.device)

    def get_img2img_result(self, seed, image_path):
        self.init_img2img_pipe()
        params = self.params
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        prompt = params.output_prompt
        output_params = OutputParams(
            prompt=prompt,
            height=params.output_height,
            width=params.output_width,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps,
            negative_prompt=params.negative_prompt,
            generator=generator,
            strength=params.strength,
            num_images_per_prompt=params.num_images_per_prompt,
            model=self.__class__.option,
        )
        to_pipe = output_params.to_kandyimg2img_pipe()
        logging.info(pprint.pformat(output_params.to_output()))
        result = self.pipeline(
            **to_pipe,
            image=load_image(image_path, size=(512, 512)),
        )
        return (result, output_params)
