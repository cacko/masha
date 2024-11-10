from transformers import pipeline, set_seed
from random import randint
from masha.pipelines import BasePipeline
from masha.pipelines import TORCH_DEVICE


class TextGeneration(BasePipeline):

    _pipeline = None

    @property
    def pipeline(self) -> str:
        if not self._pipeline:
            self._pipeline = pipeline(
                'text-generation',
                device=TORCH_DEVICE,
                model=self.modelPath.as_posix()
            )
        return self._pipeline

    def getGeneratedResult(
        self,
        context="",
        num_return_sequences=1
    ):
        set_seed(randint(40, 100))
        res = self.pipeline(
            context,
            num_return_sequences=num_return_sequences,
        )
        return ". ".join(res[0]['generated_text'].split("\n"))
