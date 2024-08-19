

from masha.pipelines.image2text import Image2Text
from masha.image.config import image_config

class InvoiceReader(Image2Text):
    
    
    def __init__(self):
        model_name  = image_config.image2text.model
        tokenizer_path = image_config.image2text.tokenizer
        super().__init__(model_name, tokenizer_path)