from coreface import DeepConfig, FaceConfig
from pydantic import BaseModel, validator

def string_to_config(configuration: str) -> DeepConfig:
    assert configuration
    fc = FaceConfig[configuration.upper()].value
    return DeepConfig(
        name=configuration, model=fc[0], detector=fc[1], metric=fc[2], color=fc[3]
    )



