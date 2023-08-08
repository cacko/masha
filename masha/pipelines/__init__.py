from pathlib import Path
from os import environ
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
DATA_ROOT = Path(environ.get("HUGGINGROOT", "."))
TORCH_DEVICE = environ.get("DEVICE", "mps")

class BasePipeline:

    __model_path: Optional[Path] = None
    __tokenizer_path: Optional[Path] = None
    __model_name: str
    __tokenizer_name: Optional[str] = None
    __data_root: Optional[Path] = None

    def __init__(self, model_name, tokenizer_path: Optional[str] = None):
        self.__model_name = model_name
        self.__tokenizer_name = tokenizer_path
        if not self.__tokenizer_name:
            self.__tokenizer_name = model_name

    @property
    def dataRoot(self) -> Path:
        if not self.__data_root:
            self.__data_root = DATA_ROOT
        return self.__data_root

    @property
    def modelPath(self) -> Path:
        if not self.__model_path:
            self.__model_path = self.dataRoot / self.__model_name
            if "/" in self.__model_name:
                self.__model_path = self.__model_name
        return self.__model_path

    @property
    def tokenizerPath(self) -> Path:
        if not self.__tokenizer_path:
            assert isinstance(self.__tokenizer_name, str)
            self.__tokenizer_path = self.dataRoot / self.__tokenizer_name
        return self.__tokenizer_path

    @property
    def modelName(self) -> str:
        return self.__model_name

    @property
    def device(cls):
        return environ.get("DEVICE", "mps")
