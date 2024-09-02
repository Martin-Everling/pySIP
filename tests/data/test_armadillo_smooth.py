import copy

import pandas as pd
import pytest

from pysip.params.prior import LogNormal, Normal
from pysip.regressors import Regressor
from pysip.statespace.thermal_network import TwTi_RoRi


@pytest.fixture
def data_armadillo():
    sT = 3600.0 * 24.0
    df = pd.read_csv("data/armadillo/armadillo_data_H2.csv").set_index("Time")
    df.drop(df.index[-1], axis=0, inplace=True)
    df.index /= sT
    return df


@pytest.fixture
def parameters_armadillo():
    sT = 3600.0 * 24.0
    return [
        dict(name="Ro", scale=1e-2, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name="Ri", scale=1e-3, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name="Cw", scale=1e7 / sT, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name="Ci", scale=1e6 / sT, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(
            name="sigw_w",
            scale=1e-2 * sT ** 0.5,
            bounds=(0, None),
            prior=LogNormal(1, 1),
        ),
        dict(name="sigw_i", value=0, transform="fixed"),
        dict(name="sigv", scale=1e-2, bounds=(0, None), prior=LogNormal(1, 1)),
        dict(name="x0_w", loc=25, scale=5, prior=Normal(0, 1)),
        dict(name="x0_i", value=26.701, transform="fixed"),
        dict(name="sigx0_w", value=1, transform="fixed"),
        dict(name="sigx0_i", value=0.1, transform="fixed"),
    ]


@pytest.fixture
def parameters_armadillo_bounded():
    sT = 3600.0 * 24.0
    return [
        dict(name="Ro", scale=1e-2, bounds=(0.1, 10.0), prior=LogNormal(1, 1)),
        dict(name="Ri", scale=1e-3, bounds=(0.1, 10.0), prior=LogNormal(1, 1)),
        dict(name="Cw", scale=1e7 / sT, bounds=(0.1, 10.0), prior=LogNormal(1, 1)),
        dict(name="Ci", scale=1e6 / sT, bounds=(0.1, 10.0), prior=LogNormal(1, 1)),
        dict(
            name="sigw_w",
            scale=1e-2 * sT**0.5,
            bounds=(0.1, 100.0),
            prior=LogNormal(1, 1),
        ),
        dict(name="sigw_i", value=0, transform="fixed"),
        dict(name="sigv", scale=1e-2, bounds=(0.1, 100.0), prior=LogNormal(1, 1)),
        dict(name="x0_w", loc=25, scale=5, bounds=(0.1, 10.0), prior=Normal(0, 1)),
        dict(name="x0_i", value=26.701, transform="fixed"),
        dict(name="sigx0_w", value=1, transform="fixed"),
        dict(name="sigx0_i", value=0.1, transform="fixed"),
    ]


@pytest.fixture
def statespace_armadillo(parameters_armadillo):
    return TwTi_RoRi(parameters_armadillo, hold_order=1)


@pytest.fixture
def regressor_armadillo(statespace_armadillo: TwTi_RoRi):
    return Regressor(
        ss=statespace_armadillo, outputs="T_int", inputs=["T_ext", "P_hea"]
    )


@pytest.fixture
def statespace_armadillo_bounded(parameters_armadillo_bounded):
    return TwTi_RoRi(parameters_armadillo_bounded, hold_order=1)


@pytest.fixture
def regressor_armadillo_bounded(statespace_armadillo_bounded: TwTi_RoRi):
    return Regressor(
        ss=statespace_armadillo_bounded, outputs="T_int", inputs=["T_ext", "P_hea"]
    )


def test_fit_predict(data_armadillo, regressor_armadillo):
    regressor_armadillo = copy.deepcopy(regressor_armadillo)
    summary, corr, scipy_summary = regressor_armadillo.fit(
        df=data_armadillo,
    )
    regressor_armadillo.predict(df=data_armadillo, smooth=False, use_outputs=True)
    regressor_armadillo.predict(df=data_armadillo, smooth=True, use_outputs=True)
    regressor_armadillo.predict(df=data_armadillo, smooth=False, use_outputs=False)
    regressor_armadillo.predict(df=data_armadillo, smooth=True, use_outputs=False)