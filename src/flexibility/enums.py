from enum import Enum, auto


class Method(Enum):
    INDIVIDUAL = auto()
    REPRESENTATIVE = auto()


class Keyword(Enum):
    SUM = auto()
    UPPER = auto()
    LOWER = auto()


class Scenario(Enum):
    HP100 = auto()  # 100% heat pumps
    HP75 = auto()
    HP50 = auto()


class TimeResolution(Enum):
    """Provide factors to convert time data in seconds to minutes, hours, etc."""

    SECONDS = 1, "s"
    MINUTES = 60, "min"
    HOURS = 60 * 60, "h"
    DAYS = 60 * 60 * 24, "d"

    def __init__(self, factor: int, label: str) -> None:
        self.factor = factor
        self.label = label


class PowerResolution(Enum):
    """Provide factors to convert power data in kW to MW, GW, etc."""

    KILOWATT = 1, "kW"
    MEGAWATT = 1_000, "MW"
    GIGAWATT = 1_000_000, "GW"
    TERAWATT = 1_000_000_000, "TW"

    def __init__(self, factor: int, label: str) -> None:
        self.factor = factor
        self.label = label


class Cunit(Enum):
    """Provide label and units to be used in charts"""

    OAT = "Outdoor air temperature", "ºC", "number"
    IAT = "Indoor air temperature", "ºC", "number"
    THERMALCAP = "Thermal capacity level", "", "string"
    EE = "Energy efficiency", "", "string"
    QUANTITY = "", "", "number"

    def __init__(self, label: str, unit: str, variable_type: str) -> None:

        self.label = label
        self.unit = unit
        self.variable_type = variable_type
