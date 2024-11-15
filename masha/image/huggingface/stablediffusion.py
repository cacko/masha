from pathlib import Path
from tkinter.tix import IMAGE

from corestring import to_int
from masha.image.diffusers import DEFAULT_IMAGE_FORMAT, IMAGE_FORMAT, Diffusers
import logging
from diffusers import (
    DiffusionPipeline,
    SemanticStableDiffusionPipeline,
)
from masha.image.models import Face2ImageResult, Image2ImageResult, PipelineParams
from typing import Optional
import torch
from torch.mps import (
    empty_cache,
    current_allocated_memory,
)
from corefile import TempPath
from masha.core import perftime
from masha.image.models import ImageResult
from humanfriendly import format_size
from masha.image.prompt import Prompt
import gc


class StableDiffusion(Diffusers):
    option = "default"
    params: Optional[PipelineParams] = None
    pipeline: Optional[DiffusionPipeline] = None

    def __del__(self):
        self.do_release()

    def do_release(self):
        logging.info("Releasing cache")
        try:
            self.pipeline = None
            del self.pipeline
        except Exception:
            pass
        gc.collect()
        empty_cache()
        logging.info(f"Memory allocated - {format_size(current_allocated_memory())}")

    def set_text2img_pipeline(self, pipe_args):
        raise NotImplementedError
    
    def get_txt2img_result(self, seed):
        raise NotImplementedError

    def set_img2img_pipeline(self, pipe_args):
        raise NotImplementedError

    def get_img2img_result(self, seed, image_path):
        raise NotImplementedError

    def set_face2img_pipeline(self, pipe_args):
        raise NotImplementedError

    def get_face2img_result(self, seed, faceid_embeds, **kwds):
        raise NotImplementedError

    def set_coreml_pipeline(self, pipe_args):
        pass
    
    def get_image_suffix(self, image_format: IMAGE_FORMAT = None):
        try:
            assert image_format
            return image_format.value
        except AssertionError:
            return DEFAULT_IMAGE_FORMAT.value

    def init_face2img_pipe(self):
        assert self.params
        logging.info(f"Memory allocated -  {format_size(current_allocated_memory())}")
        pipe_args = dict()
        self.set_face2img_pipeline(pipe_args)

    def init_img2img_pipe(self):
        logging.debug(self.__class__.pipelineClass)
        assert self.params
        params = self.params
        logging.info(f"Memory allocated -  {format_size(current_allocated_memory())}")
        pipe_args = dict()
        if StableDiffusion.is_superuser:
            logging.warn("SUPERUSER > DISABLING SAFETY CHECKER")
            pipe_args = {**pipe_args, **dict(safety_checker=None)}  # type: ignore
        if params.editing_prompt:
            self.pipeline =  SemanticStableDiffusionPipeline.from_pretrained(
                **pipe_args,
            )
        else:
            self.set_img2img_pipeline(pipe_args)

    def init_txt2img_pipe(self):
        if self.pipeline:
            return self.pipeline
        logging.debug(self.__class__.pipelineClass)
        assert self.params
        params = self.params
        logging.info(f"Memory allocated -  {format_size(current_allocated_memory())}")
        pipe_args = dict()
        if StableDiffusion.is_superuser:
            logging.warn("SUPERUSER > DISABLING SAFETY CHECKER")
            pipe_args = {**pipe_args, **dict(safety_checker=None)}  # type: ignore
        if params.editing_prompt:
            return SemanticStableDiffusionPipeline.from_pretrained(
                **pipe_args,
            )
        
        self.set_text2img_pipeline(pipe_args)

    def generate_from_text(
        self,
        params: PipelineParams
    ):
        self.params = params
        params = self.params
        paths = []
        tmp_path = TempPath(
            f"{self.__class__.get_filestem(params=params)}.{self.image_suffix}"
        )
        seed = params.seed
        if not seed:
            seed = torch.Generator(self.__class__.device).seed()
        with perftime(f"{self.__class__.modelPath.stem}, seed: {seed}"):
            output_params = None
            res = None
            params = self.params
            if params.auto_prompt:
                self.params.prompt = Prompt.cls_for_option(params.auto_prompt).generate(
                    params.prompt
                )
                logging.debug(f"auto prompt -> {self.params.prompt}")

            res, output_params = self.get_txt2img_result(seed)
            assert res
            for idx, image in enumerate(res.images):
                if params.upscale:
                    image = self.__class__.upscale(image, 4)
                pth = TempPath(f"{tmp_path.stem}{idx:02d}{tmp_path.suffix}")
                image.save(pth.as_posix())
                assert pth.exists()
                paths.append(pth)

        result = ImageResult(image=paths, params=output_params, seed=seed)
        result.write_exif()
        self.do_release()
        logging.debug(f"MEM END - {format_size(current_allocated_memory())}")
        return result

    def generate_from_image(
        self,
        image_path: Path,
        params: PipelineParams,
        **kwds,
    ):
        self.params = params
        paths = []
        tmp_path = TempPath(
            f"{self.__class__.get_filestem(params=self.params)}.{self.image_suffix}"
        )
        seed = self.params.seed
        if not seed:
            seed = torch.Generator(self.__class__.device).seed()
        with perftime(f"{self.__class__.modelPath.stem}, seed: {seed}"):
            output_params = None
            res = None
            # params = self.params
            # if params.auto_prompt:
            #     self.params.prompt = Prompt.cls_for_option(params.auto_prompt).generate(
            #         params.prompt
            #     )
            #     logging.debug(f"auto prompt -> {self.params.prompt}")

            res, output_params = self.get_img2img_result(seed, image_path)
            assert res
            for idx, image in enumerate(res.images):
                if params.upscale:
                    image = self.__class__.upscale(image, scale=params.upscale)
                pth = TempPath(f"{tmp_path.stem}{idx:02d}{tmp_path.suffix}")
                image.save(pth.as_posix())
                paths.append(pth)
        result = Image2ImageResult(image=paths, params=output_params, seed=seed)
        result.write_exif(kwds.get("extra_exif", {}))
        self.do_release()
        logging.debug(f"MEM END - {format_size(current_allocated_memory())}")
        return result

    def generate_from_face(
        self,
        faceid_embeds,
        params: PipelineParams,
        **kwds,
    ):
        self.params = params
        params = self.params
        paths = []
        tmp_path = TempPath(
            f"{self.__class__.get_filestem(params=params)}.{self.image_suffix}"
        )
        seed = to_int(params.seed, -1)
        if seed < 1:
            seed = torch.Generator(self.__class__.device).seed()
        with perftime(f"{self.__class__.modelPath.stem}, seed: {seed}"):
            output_params = None
            params = self.params
            if params.auto_prompt:
                self.params.prompt = Prompt.cls_for_option(params.auto_prompt).generate(
                    params.prompt
                )
                logging.debug(f"auto prompt -> {self.params.prompt}")

            images, output_params = self.get_face2img_result(
                seed, faceid_embeds, **kwds
            )
            assert images
            for idx, image in enumerate(images):
                if params.upscale:
                    image = self.__class__.upscale(image, scale=params.upscale)
                pth = TempPath(f"{tmp_path.stem}{idx:02d}{tmp_path.suffix}")
                image.save(pth.as_posix())
                paths.append(pth)
        result = Face2ImageResult(image=paths, params=output_params, seed=seed)
        result.write_exif(kwds.get("extra_exif", {}))
        self.do_release()
        logging.debug(f"MEM END - {format_size(current_allocated_memory())}")
        return result
