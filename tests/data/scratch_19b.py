import numpy as np
import pandas as pd

from pysip.params.prior import LogNormal, Normal
from pysip.regressors import Regressor
from pysip.statespace.thermal_network import TwTi_RoRi


def get_data_armadillo():
    sT = 3600.0 * 24.0
    df = pd.read_csv("data/armadillo/armadillo_data_H2.csv").set_index("Time")
    df.drop(df.index[-1], axis=0, inplace=True)
    df.index /= sT
    return df


def get_parameters_armadillo():
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


def get_parameters_armadillo_bounded():
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


def get_statespace_armadillo(parameters_armadillo):
    return TwTi_RoRi(parameters_armadillo, hold_order=1)


def get_regressor_armadillo(statespace_armadillo):
    return Regressor(
        ss=statespace_armadillo, outputs="T_int", inputs=["T_ext", "P_hea"]
    )


def main():
    data_armadillo = get_data_armadillo()

    # Do the normal fit
    parameters_armadillo_1 = get_parameters_armadillo()
    statespace_armadillo_1 = get_statespace_armadillo(parameters_armadillo_1)
    regressor_armadillo_1 = get_regressor_armadillo(statespace_armadillo_1)

    summary_1, corr_1, scipy_summary_1 = regressor_armadillo_1.fit(
        df=data_armadillo,
        include_prior=False,
        include_penalty=False,
    )

    results_1 = regressor_armadillo_1.predict(df=data_armadillo, use_outputs=False)
    y_pred_1 = results_1.y_mean.values

    # BOUNDED
    parameters_armadillo_1b = get_parameters_armadillo_bounded()
    statespace_armadillo_1b = get_statespace_armadillo(parameters_armadillo_1b)
    regressor_armadillo_1b = get_regressor_armadillo(statespace_armadillo_1b)

    summary_1b, corr_1b, scipy_summary_1b = regressor_armadillo_1b.fit(
        df=data_armadillo,
        include_prior=False,
        include_penalty=False,
    )
    regressor_armadillo_1b.predict(df=data_armadillo, use_outputs=False)

    parameters_armadillo_1bp = get_parameters_armadillo_bounded()
    statespace_armadillo_1bp = get_statespace_armadillo(parameters_armadillo_1bp)
    regressor_armadillo_1bp = get_regressor_armadillo(statespace_armadillo_1bp)

    summary_1bp, corr_1bp, scipy_summary_1bp = regressor_armadillo_1bp.fit(
        df=data_armadillo,
        include_prior=False,
        include_penalty=True,
    )
    regressor_armadillo_1bp.predict(df=data_armadillo, use_outputs=False)

    # Handle the alternative fit
    parameters_armadillo_2 = get_parameters_armadillo()
    statespace_armadillo_2 = get_statespace_armadillo(parameters_armadillo_2)
    regressor_armadillo_2 = get_regressor_armadillo(statespace_armadillo_2)

    summary_2, corr_2, scipy_summary_2 = regressor_armadillo_2.fit(
        df=data_armadillo,
        k_simulations=3,
        n_simulation=48,  # 1 represent 0.5 hours, so 48 represents a full day
    )
    results_2 = regressor_armadillo_2.predict(df=data_armadillo, use_outputs=False)
    y_pred_2 = results_2.y_mean.values

    # Compare with weights
    parameters_armadillo_3 = get_parameters_armadillo()
    statespace_armadillo_3 = get_statespace_armadillo(parameters_armadillo_3)
    regressor_armadillo_3 = get_regressor_armadillo(statespace_armadillo_3)

    summary_3, corr_3, scipy_summary_3 = regressor_armadillo_3.fit(
        df=data_armadillo,
        k_simulations=3,
        n_simulation=48,  # 1 represent 0.5 hours, so 48 represents a full day
        simulation_weights=(48 - np.arange(48)) / 48,
    )
    results_3 = regressor_armadillo_3.predict(df=data_armadillo, use_outputs=False)
    y_pred_3 = results_3.y_mean.values

    y = data_armadillo[regressor_armadillo_1.outputs[0]]
    y_compare = pd.DataFrame(
        index=data_armadillo.index,
        data={
            "y": y,
            "y_pred_1": y_pred_1[:, 0],
            "y_pred_2": y_pred_2[:, 0],
            "y_pred_3": y_pred_3[:, 0],
            "y_error_1": y - y_pred_1[:, 0],
            "y_error_2": y - y_pred_2[:, 0],
            "y_error_3": y - y_pred_3[:, 0],
        },
    )

    print(f"First error 1-step target function: {y_compare['y_error_1'].iloc[1]}")
    print(f"First error n-step target function: {y_compare['y_error_2'].iloc[1]}")
    print(
        f"First_error n-step weighted target function: "
        f"{y_compare['y_error_3'].iloc[1]}"
    )

    print(f"MAE 1-step target function: {y_compare['y_error_1'].abs().mean()}")
    print(f"MAE n-step target function: {y_compare['y_error_2'].abs().mean()}")
    print(f"MAE n-step weighted target function: {y_compare['y_error_3'].abs().mean()}")

    pass


if __name__ == "__main__":
    main()
