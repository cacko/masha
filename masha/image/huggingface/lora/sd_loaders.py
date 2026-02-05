import logging
from safetensors.torch import safe_open
import os
from diffusers.pipelines import StableDiffusionPipeline
import logging
from masha.image.huggingface.utils import get_embeddings, get_lora_models
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

class LoadersSDMixin(object):
    def loadLoraWeights(self):
        try:
            assert isinstance(self.pipeline, StableDiffusionPipeline)
            assert self.params
            prompt = self.params.prompt
            assert hasattr(self.pipeline, "load_lora_weights")
            loras = list(get_lora_models(prompt, self.lora_path))
            adapter_names = []
            adapter_weights = []
            assert loras
            for lora, wgt in loras:
                try:
                    logging.debug(f">> LOADING {lora.as_posix()}")
                    adapter_name = lora.stem
                    weights_root = lora.parent.as_posix()
                    weights_file = lora.name
                    self.pipeline.load_lora_weights(
                        weights_root, weight_name=weights_file, adapter_name=adapter_name
                    )
                    adapter_names.append(adapter_name)
                    adapter_weights.append(wgt)
                except Exception as e:
                    logging.error(e)
            self.pipeline.set_adapters(
                adapter_names=adapter_names, adapter_weights=adapter_weights
            )
            # pipeline.fuse_lora(lora_scale=1, adapter_names=adapter_names)
            # pipeline.unload_lora_weights()
            logging.info(f">> LOADED {adapter_names} {adapter_weights}")
        except AssertionError:
            pass
        except Exception as e:
            logging.error(e)


    def loadTextualInversion(self):
        try:
            assert self.params
            prompt = self.params.prompt
            negative_prompt = self.params.negative_prompt
            txt = f"{prompt} {negative_prompt}"
            embeddings = list(
                get_embeddings(
                    prompt=txt,
                    root=self.text_inversion_root,
                    embeddings=image_config.embeddings,
                )
            )
            assert embeddings
            assert hasattr(self.pipeline, "load_textual_inversion")
            logging.debug("LOADING TEXTUAL INVERSIONS")
            for embedding in embeddings:
                self.pipeline.load_textual_inversion(
                    embedding.path,
                    token=embedding.token,
                )
                logging.info(f">> LOADED {embedding.path} -> {embedding.token}")
        except AssertionError:
            pass
        except Exception as e:
            logging.error(e)
