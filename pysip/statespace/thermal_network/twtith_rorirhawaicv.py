from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTiTh_RoRiRhAwAicv(RCModel):
    """Third order RC model"""

    states = [
        ("TEMPERATURE", "xw", "envelope temperature"),
        ("TEMPERATURE", "xi", "indoor temperature"),
        ("TEMPERATURE", "xh", "heaters temperature"),
    ]

    params = [
        ("THERMAL_RESISTANCE", "Ro", "between the outdoor and the envelope"),
        ("THERMAL_RESISTANCE", "Ri", "between the envelope and the indoor"),
        ("THERMAL_RESISTANCE", "Rh", "between the heaters and the indoor"),
        ("THERMAL_CAPACITY", "Cw", "of the envelope"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("THERMAL_CAPACITY", "Ch", "of the heaters"),
        ("SOLAR_APERTURE", "Aw", "of the envelope"),
        ("SOLAR_APERTURE", "Ai", "of the windows"),
        ("COEFFICIENT", "cv", "scaling of the heat from the ventilation"),
        ("STATE_DEVIATION", "sigw_w", "of the envelope dynamic"),
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("STATE_DEVIATION", "sigw_h", "of the heaters dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0_w", "of the envelope temperature"),
        ("INITIAL_MEAN", "x0_i", "of the infoor temperature"),
        ("INITIAL_MEAN", "x0_h", "of the heater temperature"),
        ("INITIAL_DEVIATION", "sigx0_w", "of the envelope temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_h", "of the heater temperature"),
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
        self.C[0, 1] = 1.0

    def update_continuous_ssm(self):
        (
            Ro,
            Ri,
            Rh,
            Cw,
            Ci,
            Ch,
            Aw,
            Ai,
            cv,
            sigw_w,
            sigw_i,
            sigw_h,
            sigv,
            x0_w,
            x0_i,
            x0_h,
            sigx0_w,
            sigx0_i,
            sigx0_h,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 0.0],
            [1.0 / (Ci * Ri), -(Ri + Rh) / (Ci * Ri * Rh), 1.0 / (Ci * Rh)],
            [0.0, 1.0 / (Ch * Rh), -1.0 / (Ch * Rh)],
        ]
        self.B[:] = [
            [1.0 / (Cw * Ro), Aw / Cw, 0.0, 0.0],
            [0.0, Ai / Ci, 0.0, cv / Ci],
            [0.0, 0.0, 1.0 / Ch, 0.0],
        ]
        self.Q[self._diag] = [sigw_w, sigw_i, sigw_h]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i, x0_h]
        self.P0[self._diag] = [sigx0_w, sigx0_i, sigx0_h]
