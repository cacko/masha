from fuzzelinho import Match, MatchMethod
from pydantic import BaseModel

from masha.config import DemographicsConfig
from .classifiers import (
    RaceClassifier,
    Gender,
    Race,
    GenderClassifier
)


class NameMatch(Match):
    minRatio = 80
    method = MatchMethod.PARTIALSET


class NameNeedle(BaseModel):
    name: str


class DemographicsMeta(type):

    config: DemographicsConfig
    _instance = None
    _genderClassifier = None
    _raceClassifier = None

    def __call__(cls, *args, **kwds):
        if not cls._instance:
            cls._genderClassifier = GenderClassifier()
            cls._raceClassifier = RaceClassifier(
                cls.config.race_model
            )
            cls._instance = type.__call__(cls, *args, **kwds)
        return cls._instance

    def register(cls, config: DemographicsConfig):
        cls.config = config

    @property
    def genderClassifier(cls) -> GenderClassifier:
        assert cls._genderClassifier
        return cls._genderClassifier

    @property
    def raceClassifier(cls) -> RaceClassifier:
        assert cls._raceClassifier
        return cls._raceClassifier

    def gender(cls, name: str) -> Gender:
        return cls().getTopGender(name)

    def race(cls, name: str) -> Race:
        return cls().getTopRace(name)


class Demographics(object, metaclass=DemographicsMeta):

    def getTopGender(self, name: str) -> Gender:
        try:
            return Gender[Demographics.genderClassifier.classify(name)]
        except ValueError:
            return Gender.U

    def getTopRace(self, name: str) -> Race:
        results = Demographics.raceClassifier.classify(name)
        assert results
        return Race(sorted(list(results.items()), key=lambda r: r[1], reverse=True)[0][0])
