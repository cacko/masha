import os
from pathlib import Path
from cachable.storage.redis import RedisStorage
from cachable.storage.file import FileStorage
from masha.name.demographics import Demographics
from masha.chat import ChatDialog
from masha.translate.translator import Translator
from masha.knowledge_base.ask import Ask
from masha.knowledge_base.summarization import WikiSummarization
from masha.knowledge_base.wolfram import Wolfram
from masha.config import app_config
import logging
# from masha.text.synonims imoport
from corelog import register, Handlers


register(os.environ.get("MASHA_LOG_LEVEL", "INFO"), handler_type=Handlers.RICH)
logging.getLogger("pytorch").setLevel(os.environ.get("PYTORCH_LOG_LEVEL", "INFO"))
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(os.environ.get("PYTORCH_LOG_LEVEL", "INFO"))
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


RedisStorage.register(app_config.redis.url)
Ask.register(app_config.wiki)
WikiSummarization.register(app_config.wiki)
ChatDialog.register(app_config.chat)
Demographics.register(app_config.demographics)
Translator.register(app_config.translation)
Wolfram.register(app_config.wolfram)


cache_root = Path(app_config.cache.base_root)
if not cache_root.exists():
    cache_root.mkdir(parents=True, exist_ok=True)
FileStorage.register(cache_root)
