import copy

import numpy as np
import pandas as pd
import pytest

from pysip.params.prior import LogNormal, Normal
from pysip.regressors import Regressor
from pysip.statespace.thermal_network import TwTi_RoRi
from pysip.utils.math import fit, mad, mae, ned, rmse, smape
from pysip.utils.statistics import aic, ccf, check_ccf, check_cpgram, cpgram, lrtest


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
            scale=1e-2 * sT**0.5,
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

    loglik = regressor_armadillo.log_likelihood(df=data_armadillo)

    y = data_armadillo["T_int"].values

    ym = regressor_armadillo.predict(
        df=data_armadillo, use_outputs=False
    ).y_mean.values.squeeze()
    res = regressor_armadillo.eval_residuals(
        df=data_armadillo
    ).residual.values.squeeze()

    assert scipy_summary.fun == pytest.approx(-316.68812014562525, rel=1e-2)
    assert loglik == pytest.approx(-328.97507135305074, rel=1e-2)
    Np = int(np.sum(regressor_armadillo.ss.parameters.free))
    assert aic(loglik, Np) == pytest.approx(-643.9501427061015, rel=1e-2)
    assert lrtest(loglik - 2, loglik, Np + 1, Np) == pytest.approx(
        0.04550026389635857, rel=1e-2
    )
    assert fit(y, ym) == pytest.approx(0.8430719684902173, rel=1e-2)
    assert rmse(y, ym) == pytest.approx(0.7434264911761783, rel=1e-2)
    assert mae(y, ym) == pytest.approx(0.6578653810019472, rel=1e-2)
    assert mad(y, ym) == pytest.approx(1.3561205016928923, rel=1e-2)
    assert smape(y, ym) == pytest.approx(1.0027822035683103, rel=1e-2)
    assert ned(y, ym) == pytest.approx(0.011151665679293845, rel=1e-2)
    # assert check_model(regressor_armadillo, summary, corr, verbose=False)
    assert res.mean() == pytest.approx(0, abs=5e-2)
    assert check_ccf(*ccf(res))[0]
    assert check_cpgram(*cpgram(res))[0]


def test_fit_predict_shgo(data_armadillo, regressor_armadillo_bounded):
    regressor_armadillo = copy.deepcopy(regressor_armadillo_bounded)
    summary, corr, scipy_summary = regressor_armadillo.fit(
        df=data_armadillo,
        optimizer="shgo",
        method="L-BFGS-B",
    )
    assert scipy_summary.fun == pytest.approx(-316.68688278425225, rel=1e-2)


def test_fit_predict_dual_annealing(data_armadillo, regressor_armadillo):
    regressor_armadillo = copy.deepcopy(regressor_armadillo)
    summary, corr, scipy_summary = regressor_armadillo.fit(
        df=data_armadillo, optimizer="dual_annealing"
    )
    assert scipy_summary.fun == pytest.approx(-316.68812014562525, rel=1e-2)


def test_fit_predict_basinhopping(data_armadillo, regressor_armadillo):
    regressor_armadillo = copy.deepcopy(regressor_armadillo)
    summary, corr, scipy_summary = regressor_armadillo.fit(
        df=data_armadillo, optimizer="basinhopping"
    )
    assert scipy_summary.fun == pytest.approx(-316.68812014562525, rel=1e-2)
