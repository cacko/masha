import logging
from pathlib import Path
from safetensors.torch import load_file, safe_open
import os
from sympy import EX
import torch
from diffusers import LCMScheduler
from diffusers.pipelines import StableDiffusionPipeline
from masha.image.models import PipelineParams
import torch
from diffusers import LCMScheduler
from diffusers.pipelines import StableDiffusionPipeline
import logging
from masha.image.huggingface.utils import get_embeddings, get_lora_models
from pathlib import Path
from masha.image.config import image_config
from diffusers.models.modeling_utils import load_state_dict


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def loadLoraWeights(pipeline: StableDiffusionPipeline, prompt: str, lora_path: Path):
    try:
        assert prompt
        assert hasattr(pipeline, "load_lora_weights")
        loras = list(get_lora_models(prompt, lora_path))
        adapter_names = []
        adapter_weights = []
        if not loras:
            return pipeline
        for lora, wgt in loras:
            try:
                logging.debug(f">> LOADING {lora.as_posix()}")
                adapter_name = lora.stem
                weights_root = lora.parent.as_posix()
                weights_file = lora.name
                pipeline.load_lora_weights(
                    weights_root, weight_name=weights_file, adapter_name=adapter_name
                )
                adapter_names.append(adapter_name)
                adapter_weights.append(wgt)
                if adapter_name.startswith("lcm"):
                    pipeline.scheduler = LCMScheduler.from_config(
                        pipeline.scheduler.config
                    )
            except Exception as e:
                logging.error(e)
        pipeline.set_adapters(
            adapter_names=adapter_names, adapter_weights=adapter_weights
        )
        # pipeline.fuse_lora(lora_scale=1, adapter_names=adapter_names)
        # pipeline.unload_lora_weights()
        logging.info(f">> LOADED {adapter_names} {adapter_weights}")
    except AssertionError:
        pass
    except Exception as e:
        logging.error(e)
    return pipeline


def loadTextualInversion(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    text_inversion_root: Path,
):
    try:
        logging.info
        txt = f"{prompt} {negative_prompt}"
        embeddings = list(
            get_embeddings(
                prompt=txt,
                root=text_inversion_root,
                embeddings=image_config.embeddings,
            )
        )
        if not embeddings:
            return pipeline
        assert hasattr(pipeline, "load_textual_inversion")
        logging.debug("LOADING TEXTUAL INVERSIONS")
        for embedding in embeddings:
            pipeline.load_textual_inversion(
                embedding.path,
                token=embedding.token,
            )
            logging.info(f">> LOADED {embedding.path} -> {embedding.token}")
    except AssertionError:
        pass
    except Exception as e:
        logging.error(e)
    return pipeline
