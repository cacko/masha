from masha.image.huggingface.sd_types.base import BaseStableDiffusion
from masha.image.models import OutputParams
import torch
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image
import logging
from humanfriendly import format_size
from torch.mps import current_allocated_memory
from masha.image.huggingface.utils import get_embeddings, get_lora_models, load_image
from pathlib import Path
from masha.config import app_config
from masha.image.config import image_config
from rich import print


class StableDiffusionLCM(BaseStableDiffusion):
    _params = dict(negative_prompt="blurry")

    @property
    def text_inversion_root(self) -> Path:
        return self.__class__.dataRoot / app_config.image.textual_inversionxl_root

    def __get_output_params(self, seed) -> OutputParams:
        params = self.params
        prompt = params.output_prompt
        generator = torch.Generator(self.__class__.device).manual_seed(seed)
        return OutputParams(
            prompt=prompt,
            guidance_scale=params.guidance_scale,
            num_inference_steps=params.num_inference_steps,
            negative_prompt=params.negative_prompt,
            generator=generator,
            width=params.output_width,
            height=params.output_height,
            seed=seed,
            num_images_per_prompt=params.num_images_per_prompt,
            model=self.__class__.option,
        )

    def get_img2img_result(self, seed, image_path):
        self.pipeline = self.img2img_pipe
        output_params = self.__get_output_params(seed)
        to_pipe = output_params.to_pipe_img2img_xl()
        print(output_params.to_output())
        result = self.pipeline(
            **{**to_pipe, "image": load_image(image_path, size=(768, 768))}
        )
        return (result, output_params)

    def get_img2img_pipeline(self, pipe_args):
        model_path = self.__class__.modelPath
        logging.info(f"MODEL PATH {model_path}")
        logging.info(f"MEM START - {format_size(current_allocated_memory())}")
        assert model_path
        pipe_args = dict(use_safetensors=True, **pipe_args)
        sd_base = AutoPipelineForImage2Image.from_pretrained(
            model_path.as_posix(),
            torch_dtype=torch.float16,
            **pipe_args,
        )
        logging.info(f"SCHEDULER {sd_base.scheduler}")
        pytorch_pipe = sd_base.to(self.__class__.device)  # type: ignore
        logging.info(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.loadLoraWeights(pytorch_pipe, self.__class__.loraxlPath)
        except Exception as e:
            logging.exception(e)
            logging.info("failed")
        logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        return pytorch_pipe

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
        logging.info(f"MEM START - {format_size(current_allocated_memory())}")
        assert model_path
        pipe_args = dict(use_safetensors=True, **pipe_args)
        sd_base = DiffusionPipeline.from_pretrained(
            model_path.as_posix(),
            add_watermarker=False,
            torch_dtype=torch.float16,
            **pipe_args,
        )
        logging.info(f"SCHEDULER {sd_base.scheduler}")
        pytorch_pipe = sd_base.to(self.__class__.device)  # type: ignore
        logging.info(f"MEM PIPE - {format_size(current_allocated_memory())}")
        try:
            self.loadLoraWeights(pytorch_pipe, self.__class__.loraxlPath)
            self.loadTextualInversion(pytorch_pipe)
        except Exception as e:
            logging.exception(e)
            logging.info("failed")
        logging.info(f"MEM LORA - {format_size(current_allocated_memory())}")
        return pytorch_pipe


    def loadLoraWeights(self, lora_path: Path):
        try:
            assert self.params
            loras = list(get_lora_models(self.params.prompt, lora_path))
            assert len(loras)
            adapter_names = []
            adapter_weights = []
            for lora, wgt in loras:
                logging.info(f">> LOADING {lora.as_posix()}")
                adapter_name = lora.stem
                weights_root = lora.parent.as_posix()
                weights_file = lora.name
                self.pipeline.load_lora_weights(
                    weights_root, weight_name=weights_file, adapter_name=adapter_name
                )
                adapter_names.append(f"adapter_{adapter_name}")
                adapter_weights.append(wgt)
                self.pipeline.fuse_lora(lora_scale=wgt, adapter_names=[adapter_name])
            logging.info("...LOADED")
        except AssertionError:
            pass

    def loadTextualInversion(
        self,
    ):
        try:
            txt = f"{self.params.prompt} {self.params.negative_prompt}"
            embeddings = list(
                get_embeddings(
                    prompt=txt,
                    root=self.text_inversion_root,
                    embeddings=image_config.embeddings,
                )
            )
            if not embeddings:
                return
            assert hasattr(self.pipeline, "load_textual_inversion")
            logging.info("LOADING TEXTUAL INVERSIONS")
            for embedding in embeddings:
                state_dict = load_image(embedding.path)
                logging.info(f">> LOADING {embedding.path} -> {embedding.token}")
                self.pipeline.load_textual_inversion(
                    state_dict["clip_g"],
                    token=embedding.token,
                    text_encoder=self.pipeline.text_encoder_2,
                    tokenizer=self.pipeline.tokenizer_2,
                )
                self.pipeline.load_textual_inversion(
                    state_dict["clip_l"],
                    token=embedding.token,
                    text_encoder=self.pipeline.text_encoder,
                    tokenizer=self.pipeline.tokenizer,
                )
        except AssertionError:
            pass
