
import logging
from masha.image.huggingface.utils import (
    get_lora_models,
)


class LoadersQwenMixin(object):

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
            logging.info(f">> LOADED {adapter_names} {adapter_weights}")
        except AssertionError:
            pass
        except Exception as e:
            logging.exception(e)