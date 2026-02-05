from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from torch.utils.data import DataLoader, TensorDataset
import torch
from scipy.special import softmax
from masha.core import perftime
from masha.pipelines import BasePipeline



class TextClassifier(BasePipeline):

    _model = None
    _tokenizer = None
    _pipeline = None

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.tokenizerPath.as_posix()
            )
            self._tokenizer.model_max_length = 1024
        return self._tokenizer

    @property
    def model(self):
        if not self._model:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.modelPath.as_posix()
            )
            self._model.eval()
        return self._model

    @property
    def pipeline(self):
        if not self._pipeline:
            self._pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer
            )
        return self._pipeline

    def classify(self, text) -> dict[str, float]:
        with perftime("LOCAL translation"):
            encoded = self.tokenizer(
                [(
                    text,
                    ' '.join(str(text)).replace('  ', ' ').replace('  ', ' ')
                )],
                return_attention_mask=True,
                padding=True, return_tensors='pt')
            dataset = TensorDataset(
                encoded['input_ids'],
                encoded['attention_mask'],
                torch.tensor([0])
            )
            dataloader = DataLoader(dataset, batch_size=1)
            batch = next(iter(dataloader))

            inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                      'labels': batch[2], }
            output = softmax(self.model(**inputs)[1].detach().tolist()[0])
            return self._formatresult(output)

    def _formatresult(self, output):
        res = {label: output[idx]
               for idx, label in self.model.config.id2label.items()}
        return res
