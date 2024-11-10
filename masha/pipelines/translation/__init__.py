from pathlib import Path
from transformers import pipeline, Pipeline
from masha.core import perftime
from masha.pipelines import TORCH_DEVICE, BasePipeline
from typing import Optional


class Translation(BasePipeline):

    _model_path: Optional[Path] = None
    _pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        if not self._pipeline:
            self._pipeline = pipeline(
                'translation',
                model=self.modelPath.as_posix(),
                device=TORCH_DEVICE
            )
        return self._pipeline

    def translate(
        self,
        context,
    ) -> str:
        with perftime("LOCAL translation"):
            res = self.pipeline(context, return_text=True)
            return res[0]['translation_text']  # type: ignore
