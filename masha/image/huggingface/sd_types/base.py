from masha.image.huggingface.stablediffusion import StableDiffusion
from masha.image.models import PipelineParams


class BaseStableDiffusion(StableDiffusion):
    _params = {}
    custom_params: dict[str, str] = {}

    def pipelineParams(self, **params):
        if not self.custom_params:
            self.custom_params = {}
        if not params:
            params = {}
        return PipelineParams(
            **{
                **self._params,
                **self.custom_params,
                **{k: v for k, v in params.items() if v is not None},
            }
        )
