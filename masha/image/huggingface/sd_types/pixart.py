from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers.pipelines import PixArtAlphaPipeline
import logging
from humanfriendly import format_size
from torch.mps import current_allocated_memory
from pathlib import Path
from rich import print


class StableDiffusionPixArt(BaseStableDiffusion):
    pipeline: PixArtAlphaPipeline

    def __get_output_params(self, seed) -> OutputParams:
        params = self.params
        prompt = params.output_prompt
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
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
        )

    def get_txt2img_result(self, seed):
        self.pipeline = self.txt2img_pipe
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_pixart_pipe()
        print(output_params.to_output())
        result = self.pipeline(**to_pipe)
        return (result, output_params)

    def get_text2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        assert model_path
        sd_pipe = PixArtAlphaPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path.as_posix(),
            use_safetensors=True,
            torch_dtype=torch.float16,
            **pipe_args,
        )
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")
        return sd_pipe.to(self.__class__.device) 

    def loadLoraWeights(self, sd_pipe: PixArtAlphaPipeline, lora_path: Path):
        pass

    def loadTextualInversion(
        self,
        sd_pipe: PixArtAlphaPipeline,
    ):
        pass
