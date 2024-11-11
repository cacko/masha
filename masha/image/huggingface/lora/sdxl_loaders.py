import logging
from masha.image.huggingface.utils import (
    get_lora_models,
    get_embeddings,
)
from diffusers import StableDiffusionXLPipeline
from masha.image.config import image_config
from diffusers.models.modeling_utils import load_state_dict


class LoadersSDXLMixin(object):

    def loadLoraWeights(self):
        try:
            assert self.params or prompt
            if self.params:
                prompt = self.params.prompt
            loras = list(get_lora_models(prompt, self.lora_path))
            assert len(loras)
            adapter_names = []
            adapter_weights = []
            for lora, wgt in loras:
                try:
                    logging.debug(f">> LOADING {lora.as_posix()}")
                    adapter_name: str = f"adapter_{lora.stem}"
                    weights_root = lora.parent.as_posix()
                    weights_file = lora.name
                    self.pipeline.load_lora_weights(
                        weights_root,
                        weight_name=weights_file,
                        adapter_name=adapter_name,
                    )
                    adapter_names.append(adapter_name)
                    adapter_weights.append(wgt)
                except Exception as e:
                    logging.warning(f">> LOADING OF {lora.name} FAILED")
                    logging.error(str(e))
            self.pipeline.set_adapters(
                adapter_names=adapter_names, adapter_weights=adapter_weights
            )
            # pipeline.fuse_lora(adapter_names=adapter_names)
            # pipeline.unload_lora_weights()
            logging.info(f">> LOADED {adapter_names} {adapter_weights}")
        except AssertionError:
            pass
        except Exception as e:
            logging.exception(e)

    def loadTextualInversion(self):
        if self.params:
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
        try:
            assert self.pipeline
            assert isinstance(self.pipeline, StableDiffusionXLPipeline)
            assert embeddings
            assert hasattr(self.pipeline, "load_textual_inversion")
            logging.debug("LOADING TEXTUAL INVERSIONS")
            for embedding in embeddings:
                state_dict = load_state_dict(embedding.path.as_posix())
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
                logging.info(f">> LOADED {embedding.path} -> {embedding.token}")
        except AssertionError:
            pass
