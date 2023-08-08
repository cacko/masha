from pathlib import Path
from pydantic import BaseModel
import yaml


class GeminiConfig(BaseModel):
    cookies: list[str]
    
class OpenRouterConfig(BaseModel):
    api_key: str
    model: str
    
class GeneratorConfig(BaseModel):
    model: str
    
class DetectorConfig(BaseModel):
    lang_model: str
    
class SkillsConfig(BaseModel):
    extractor_model: str
    
class CoverLetterConfig(BaseModel):
    model: str
    
class SynonimsConfig(BaseModel):
    model: str

class TextConfig(BaseModel):
    gemini: GeminiConfig
    openrouter: OpenRouterConfig
    generator: GeneratorConfig
    detector: DetectorConfig
    skills: SkillsConfig
    cover_letter: CoverLetterConfig
    synonims: SynonimsConfig

config_path = Path(__file__).parent / "config.yaml"
data = yaml.full_load(config_path.read_text())


text_config = TextConfig(**data)
