from dataclasses import dataclass

from ..base import RCModel


@dataclass
class Ti_RAcv(RCModel):
    states = [("TEMPERATURE", "xi", "indoor temperature")]

    params = [
        ("THERMAL_RESISTANCE", "R", "between the outdoor and the indoor"),
        ("THERMAL_CAPACITY", "C", "effective overall capacity"),
        ("SOLAR_APERTURE", "A", "effective solar aperture"),
        ("COEFFICIENT", "cv", "scaling of the heat from the ventilation"),
        ("STATE_DEVIATION", "sigw", "of the indoor dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0", "of the infoor temperature"),
    ]

    inputs = [
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
        ("POWER", "Qv", "heat from the ventilation system"),
    ]

    outputs = [("TEMPERATURE", "xi", "indoor temperature")]

    def __post_init__(self):
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        self.C[0, 0] = 1.0

    def update_continuous_ssm(self):
        R, C, A, cv, sigw, sigv, x0, sigx0, *_ = self.parameters.theta

        self.A[0, 0] = -1.0 / (C * R)
        self.B[0, :] = [1.0 / (C * R), A / C, 1.0 / C, cv / C]
        self.Q[0, 0] = sigw
        self.R[0, 0] = sigv
        self.x0[0, 0] = x0
        self.P0[0, 0] = sigx0
