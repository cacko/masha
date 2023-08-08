from masha.config import TranslationConfig
from masha.pipelines.translation import Translation
from pathlib import Path
from typing import Optional


class TranslatorMeta(type):

    _instances: dict[str, 'Translator'] = {}
    _model_root: Optional[Path] = None
    _config: TranslationConfig

    def __call__(cls, *args, **kwds):
        lang = args[0] if len(args) > 0 else kwds.get("lang")
        assert lang
        if lang not in cls._instances:
            model_name = getattr(cls._config, (f"{lang}_model"))
            cls._instances[lang] = type.__call__(cls, model_name, *args, **kwds)
        return cls._instances[lang]

    @classmethod
    def register(cls, config: TranslationConfig):
        cls._config = config

    def translated(cls, lang, text) -> str:
        return cls(lang).translate(text)


class Translator(Translation, metaclass=TranslatorMeta):
    pass
