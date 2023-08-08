import string
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class HuggingConfig(BaseModel):
    root: str


class ChatConfig(BaseModel):
    dialog_model: str
    dialog_tokenizer: str
    dialog_sarcastic_model: str
    dialog_sarcastic_tokenizer: str
    dialog_phrase_model: str
    dialog_phrase_tokenizer: str


# class TextConfig(BaseModel):
#     generator_model: str
#     detector_lang_model: str
#     skill_extractor_model: str
#     cover_letter_model: str
#     synonims_model: str
    

class DemographicsConfig(BaseModel):
    gender_model: str
    gender_tokenizer: str
    race_model: str


class WikiConfig(BaseModel):
    qa_model: str
    summarization_model: str


class TranslationConfig(BaseModel):
    en_es_model: str
    es_en_model: str
    bg_en_model: str
    en_bg_model: str
    en_cs_model: str
    cs_en_model: str
    pl_en_model: str
    en_pl_model: str
    yoda_model: str

class GoogleConfig(BaseModel):
    maps_api_key: str
    maps_api_secret: str

class WolframConfig(BaseModel):
    app_name: str
    app_id: str


class RedisConfig(BaseModel):
    url: str


class BeatsConfig(BaseModel):
    store_root: str


class ServerConfig(BaseModel):
    host: str
    port: int
    workers: int
    loop: str


class CacheConfig(BaseModel):
    base_root: str
    generated_root: str


class S3Config(BaseModel):
    cloudfront_host: str
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str
    storage_bucket_name: str
    directory: str

class HueConfig(BaseModel):
    username: str
    hostname: str
    clientkey: str

class FirebaseConfig(BaseModel):
    rdb_host: str
    service_json: str


class Settings(BaseSettings):
    hugging: HuggingConfig
    chat: ChatConfig
    beats: BeatsConfig
    redis: RedisConfig
    wolfram: WolframConfig
    wiki: WikiConfig
    translation: TranslationConfig
    demographics: DemographicsConfig
    server: ServerConfig
    s3: S3Config
    google: GoogleConfig
    cache: CacheConfig
    hue: HueConfig
    firebase: FirebaseConfig

    class Config:
        env_nested_delimiter = "__"


app_config = Settings()  # type: ignore
