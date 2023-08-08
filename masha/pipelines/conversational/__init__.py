import rich
from transformers import PreTrainedTokenizer
from masha.pipelines import BasePipeline
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Conversational(BasePipeline):

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        raise NotImplementedError

    @property
    def model(self):
        raise NotImplementedError

    def conversation_response(
        self,
        text: str,
        **kwargs
    ) -> str:
        while True:
            if resp := self.get_response(text, **kwargs):
                return resp

    def get_response(self, text: str) -> str:
        tokenizer = self.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        model = self.model
        new_user_input_ids = tokenizer.encode(
            text + tokenizer.eos_token,
            return_tensors='pt'
        ).to(device)

        chat_history_ids = model.generate(
            new_user_input_ids,
            do_sample=True,
            max_length=1000,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(
            chat_history_ids[:, new_user_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
