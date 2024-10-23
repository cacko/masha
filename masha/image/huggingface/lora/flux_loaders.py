import logging
from os import path
from typing import Optional
from masha.image.huggingface.utils import (
    get_lora_models,
)


class LoadersFluxMixin(object):

    def loadLoraWeights(
        self, prompt=None
    ) -> tuple[list[str], list[float]]:
        assert self.params or prompt
        if self.params:
            prompt = self.params.prompt
        loras = list(get_lora_models(prompt, self.lora_path))
        assert len(loras)
        paths = []
        scales = []
        for lora, wgt in loras:
            paths.append(lora.as_posix())
            scales.append(wgt)
        logging.info(f">> FOUND LORAS {paths} {scales}")
        return paths, scales