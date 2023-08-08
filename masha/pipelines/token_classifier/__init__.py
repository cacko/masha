from transformers import pipeline, Pipeline
from masha.pipelines import BasePipeline


class TokenClassifier(BasePipeline):

    _pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        if not self._pipeline:
            self._pipeline = pipeline(
                'ner',
                model=self.modelPath.as_posix(),
                tokenizer=self.modelPath.as_posix(),
                stride=256,
                aggregation_strategy="simple"
            )
        return self._pipeline

    def getGeneratedResult(
        self,
        context="",
    ):
        tokens = self.pipeline(
            context,
        )
        return tokens
