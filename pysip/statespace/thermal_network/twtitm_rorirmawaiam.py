from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TwTiTm_RoRiRmAwAiAm(RCModel):
    """Third order RC model"""

    states = [
        ("TEMPERATURE", "xw", "envelope temperature"),
        ("TEMPERATURE", "xi", "indoor temperature"),
        ("TEMPERATURE", "xm", "internal mass temperature"),
    ]

    params = [
        ("THERMAL_RESISTANCE", "Ro", "between the outdoor and the envelope"),
        ("THERMAL_RESISTANCE", "Ri", "between the envelope and the indoor"),
        ("THERMAL_RESISTANCE", "Rm", "between the indoor and the internal mass"),
        ("THERMAL_CAPACITY", "Cw", "of the envelope"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("THERMAL_CAPACITY", "Cm", "of the internal mass"),
        ("SOLAR_APERTURE", "Aw", "of the envelope"),
        ("SOLAR_APERTURE", "Ai", "of the windows"),
        ("SOLAR_APERTURE", "Am", "of the internal mass"),
        ("STATE_DEVIATION", "sigw_w", "of the envelope dynamic"),
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("STATE_DEVIATION", "sigw_m", "of the internal mass dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0_w", "of the envelope temperature"),
        ("INITIAL_MEAN", "x0_i", "of the infoor temperature"),
        ("INITIAL_MEAN", "x0_m", "of the internal mass temperature"),
        ("INITIAL_DEVIATION", "sigx0_w", "of the envelope temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_m", "of the internal mass temperature"),
    ]

    inputs = [
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
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
            Rm,
            Cw,
            Ci,
            Cm,
            Aw,
            Ai,
            Am,
            sigw_w,
            sigw_i,
            sigw_m,
            sigv,
            x0_w,
            x0_i,
            x0_m,
            sigx0_w,
            sigx0_i,
            sigx0_m,
            *_,
        ) = self.parameters.theta

        self.A[:] = [
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 0.0],
            [1.0 / (Ci * Ri), -(Ri + Rm) / (Ci * Ri * Rm), 1.0 / (Ci * Rm)],
            [0.0, 1.0 / (Cm * Rm), -1.0 / (Cm * Rm)],
        ]
        self.B[:] = [
            [1.0 / (Cw * Ro), Aw / Cw, 0.0],
            [0.0, Ai / Ci, 1.0 / Ci],
            [0.0, Am / Cm, 0.0],
        ]
        self.Q[self._diag] = [sigw_w, sigw_i, sigw_m]
        self.R[0, 0] = sigv
        self.x0[:, 0] = [x0_w, x0_i, x0_m]
        self.P0[self._diag] = [sigx0_w, sigx0_i, sigx0_m]
