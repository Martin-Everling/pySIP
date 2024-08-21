import copy

import numpy as np
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
def statespace_armadillo(parameters_armadillo):
    return TwTi_RoRi(parameters_armadillo, hold_order=1)


@pytest.fixture
def regressor_armadillo(statespace_armadillo: Regressor):
    return Regressor(
        ss=statespace_armadillo, outputs="T_int", inputs=["T_ext", "P_hea"]
    )


def test_fit_predict_n_step(data_armadillo, regressor_armadillo):
    regressor_armadillo_1 = copy.deepcopy(regressor_armadillo)
    summary_1, corr_1, scipy_summary_1 = regressor_armadillo_1.fit(
        df=data_armadillo,
    )
    results_1 = regressor_armadillo_1.predict(
        df=data_armadillo,
        use_outputs=False
    )
    y_pred_1 = results_1.y_mean.values

    # Handle the alternative fit
    regressor_armadillo_2 = copy.deepcopy(regressor_armadillo)
    summary_2, corr_2, scipy_summary_2 = regressor_armadillo_2.fit(
        df=data_armadillo,
        k_simulations=3,
        n_simulation=48,  # 1 represent 0.5 hours, so 48 represents a full day
    )
    results_2 = regressor_armadillo_2.predict(
        df=data_armadillo,
        use_outputs=False
    )
    y_pred_2 = results_2.y_mean.values

    # Compare with weights
    regressor_armadillo_3 = copy.deepcopy(regressor_armadillo)
    summary_3, corr_3, scipy_summary_3 = regressor_armadillo_3.fit(
        df=data_armadillo,
        k_simulations=3,
        n_simulation=48,  # 1 represent 0.5 hours, so 48 represents a full day
        simulation_weights=(48 - np.arange(48)) / 48,
    )
    results_3 = regressor_armadillo_3.predict(
        df=data_armadillo,
        use_outputs=False
    )
    y_pred_3 = results_3.y_mean.values

    y = data_armadillo[regressor_armadillo_1.outputs[0]]
    y_compare = pd.DataFrame(
        index=data_armadillo.index,
        data={
            'y': y,
            'y_pred_1': y_pred_1[:, 0],
            'y_pred_2': y_pred_2[:, 0],
            'y_pred_3': y_pred_3[:, 0],
            'y_error_1': y - y_pred_1[:, 0],
            'y_error_2': y - y_pred_2[:, 0],
            'y_error_3': y - y_pred_3[:, 0],
        }
    )

    print(f"First error 1-step target function: {y_compare['y_error_1'].iloc[1]}")
    print(f"First error n-step target function: {y_compare['y_error_2'].iloc[1]}")
    print(
        f"First_error n-step weighted target function: {y_compare['y_error_3'].iloc[1]}"
    )

    print(f"MAE 1-step target function: {y_compare['y_error_1'].abs().mean()}")
    print(f"MAE n-step target function: {y_compare['y_error_2'].abs().mean()}")
    print(f"MAE n-step weighted target function: {y_compare['y_error_3'].abs().mean()}")

    # Check individual model fits
    assert (y_compare['y_error_1'].iloc[1]
            == pytest.approx(0.017802806421535422, rel=1e-2))
    assert (y_compare['y_error_2'].iloc[1]
            == pytest.approx(-0.1265164893965256, rel=1e-2))
    assert (y_compare['y_error_3'].iloc[1]
            == pytest.approx(-0.10903685109297356, rel=1e-2))
    assert (y_compare['y_error_1'].abs().mean()
            == pytest.approx(0.6578543247342358, rel=1e-2))
    assert (y_compare['y_error_2'].abs().mean()
            == pytest.approx(0.5800431751209509, rel=1e-2))
    assert (y_compare['y_error_3'].abs().mean()
            == pytest.approx(0.59751486149397, rel=1e-2))
    assert scipy_summary_1.fun == pytest.approx(-316.687870, rel=1e-2)
    assert scipy_summary_2.fun == pytest.approx(-217.849030, rel=1e-2)
    assert scipy_summary_3.fun == pytest.approx(-205.296176, rel=1e-2)

    # Compare the prediction error of the first step ahead
    assert (y_compare['y_error_1'].abs().iloc[1]
            < y_compare['y_error_2'].abs().iloc[1]), \
        'The 1-step ahead prediction should be better with a normal fit'
    assert (y_compare['y_error_1'].abs().iloc[1]
            < y_compare['y_error_3'].abs().iloc[1]), \
        'The 1-step ahead prediction should be better with a normal fit'
    assert (y_compare['y_error_3'].abs().iloc[1]
            < y_compare['y_error_2'].abs().iloc[1]), \
        'The 1-step ahead prediction should be better when using weights'

    # Compare the MAE of the n-step ahead
    assert y_compare['y_error_2'].abs().mean() < y_compare['y_error_1'].abs().mean(), \
        'The n-step ahead MAE should be better with the simulation fit'
    assert y_compare['y_error_3'].abs().mean() < y_compare['y_error_1'].abs().mean(), \
        'The n-step ahead MAE should be better with the simulation fit'
    assert y_compare['y_error_2'].abs().mean() < y_compare['y_error_3'].abs().mean(), \
        'The n-step ahead MAE should be better without using weights'
