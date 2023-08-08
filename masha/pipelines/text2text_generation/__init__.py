from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from masha.pipelines import BasePipeline


class Text2TextGeneration(BasePipeline):
    _pipeline = None
    _model = None
    _tokenizer = None


    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = T5Tokenizer.from_pretrained(
                pretrained_model_name_or_path=self.modelPath.as_posix()
            )
        return self._tokenizer

    @property
    def model(self):
        if not self._model:

            self._model = T5ForConditionalGeneration.from_pretrained(
                self.modelPath.as_posix()
            )
            self._model.eval()
        return self._model

    @property
    def pipeline(self) -> str:
        if not self._pipeline:
            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        return self._pipeline

    def getGeneratedResult(
        self,
        prompt: str,
        max_length=2048,
        temperature=1.8,
        top_p=0.9,
        max_time=200,
        **model_args
    ):
        args = {
            "max_length": max_length,
            # "temperature": temperature,
            # "top_p": top_p,
            # "early_stopping": True,
            "max_time": max_time,
            "do_sample": True,  # do_sample=False to force deterministic output
            "num_return_sequences": 1,  # number of samples to return
            "min_length": 512,
            # "num_beams": 4,
            # "no_repeat_ngram_size": 3,
            **model_args
        }

        res = self.pipeline(prompt, **args)
        return res[0]["generated_text"]
