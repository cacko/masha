from typing import Any
from corestring import split_with_quotes
from masha.pipelines.text_classifier import TextClassifier
from enum import StrEnum
from gender_predictor import GenderPredictor


class Gender(StrEnum):
    M = "male"
    F = "female"
    G = "faggot"
    U = "unknown"


class Race(StrEnum):
    BRITISH = "GreaterEuropean,British"
    FRENCH = "GreaterEuropean,WestEuropean,French"
    ITALIAN = "GreaterEuropean,WestEuropean,Italian"
    HISPANIC = "GreaterEuropean,WestEuropean,Hispanic"
    JEWISH = "GreaterEuropean,Jewish"
    EASTEURO = "GreaterEuropean,EastEuropean"
    INDOPAK = "Asian,IndianSubContinent"
    JAPANESE = "Asian,GreaterEastAsian,Japanese"
    MUSLIM = "GreaterAfrican,Muslim"
    ASIAN = "Asian,GreaterEastAsian,EastAsian"
    NORDIC = "GreaterEuropean,WestEuropean,Nordic"
    GERMAN = "GreaterEuropean,WestEuropean,Germanic"
    AFRICAN = "GreaterAfrican,Africans"
    FAGGOT = "faggot"
    
class Ethnicity(StrEnum):
    BLACK = "african"
    ASIAN = "asian"
    HISPANIC = "hispanic"
    INDIAN = "indian"
    ARABIC="arabic"
    EAST_ASIAN="east asian"
    WHITE="white"
    MIDDLE_EASTERN="middle_eastern",
    LATINO_HISPANIC="latino_hispanic",
    SOUTHEAST_ASIAN="southeast_asian"
    
    @classmethod
    def _missing_(cls, value: object) -> Any:
        value = value.lower()
        match value:
            case "black":
                return cls.BLACK
            case _:
                return cls.WHITE

class GenderClassifier:

    def __init__(self):
        self.gp = GenderPredictor()

    def classify(self, name):
        name = split_with_quotes(name)
        return self.gp.classify(name[0])


class RaceClassifier(TextClassifier):
    pass
