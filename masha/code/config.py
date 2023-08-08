from pathlib import Path
from pydantic import BaseModel
import yaml


class SummarizationConfig(BaseModel):
    model: str
    
class InstructConfig(BaseModel):
    model: str
    
class InstructPythonConfig(InstructConfig):
    pass

class InstructPHPConfig(InstructConfig):
    pass

class InstructJavascriptConfig(InstructConfig):
    pass

class CodeConfig(BaseModel):
    summarization: SummarizationConfig
    instruct: InstructConfig
    instruct_python: InstructPythonConfig
    instruct_php: InstructPHPConfig
    instruct_javascript: InstructJavascriptConfig

config_path = Path(__file__).parent / "config.yaml"
data = yaml.full_load(config_path.read_text())


code_config = CodeConfig(**data)
