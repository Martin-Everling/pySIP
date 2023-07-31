from copy import deepcopy
import math
from typing import NamedTuple, Sequence
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .statespace.base import StateSpace

from numba.core.errors import NumbaPerformanceWarning
import xarray as xr


#### Models, compatible with the Numba implementation of the Kalman filter ####


class States(NamedTuple):
    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray


def _allocate_filter_res(n_timesteps, nx, ny):
    return FilteringResult(
        np.empty((n_timesteps, nx, 1)),
        np.empty((n_timesteps, nx, nx)),
        np.empty((n_timesteps, nx, 1)),
        np.empty((n_timesteps, nx, nx)),
        np.empty((n_timesteps, ny, 1)),
        np.empty((n_timesteps, ny, ny)),
    )


def _get_filter_estimate(filter_estimate, i):
    return FilteringResult(
        filter_estimate.x_update[i],
        filter_estimate.P_update[i],
        filter_estimate.x_predict[i],
        filter_estimate.P_predict[i],
        filter_estimate.k[i],
        filter_estimate.S[i],
    )


def _set_filter_estimate(filter_estimate, i, value):
    (
        filter_estimate.x_update[i],
        filter_estimate.P_update[i],
        filter_estimate.x_predict[i],
        filter_estimate.P_predict[i],
        filter_estimate.k[i],
        filter_estimate.S[i],
    ) = value


class FilteringResult(NamedTuple):
    x_update: np.ndarray
    P_update: np.ndarray
    x_predict: np.ndarray
    P_predict: np.ndarray
    k: np.ndarray
    S: np.ndarray

    def __getitem__(self, i):
        return _get_filter_estimate(self, i)

    def __setitem__(self, i, value):
        _set_filter_estimate(self, i, value)

    def to_xarray(self, idx_name, time, states, outputs):
        coords = {
            idx_name: time,
            "states": states,
            "outputs": outputs,
        }
        ds = xr.Dataset(
            {
                "x_update": ((idx_name, "states"), self.x_update[:, :, 0]),
                "x_predict": ((idx_name, "states"), self.x_predict[:, :, 0]),
                "P_update": ((idx_name, "states", "states"), self.P_update),
                "P_predict": ((idx_name, "states", "states"), self.P_predict),
                "k": ((idx_name, "outputs"), self.k[:, :, 0]),
                "S": ((idx_name, "outputs", "outputs"), self.S),
            },
            coords=coords,
        )
        return ds


def _smooth_from_filter(filter_estimate):
    return SmoothingResult(
        filter_estimate.x_update,
        filter_estimate.P_update,
        filter_estimate.x_predict,
        filter_estimate.P_predict,
        filter_estimate.x_predict.copy(),
        filter_estimate.P_predict.copy(),
        filter_estimate.k,
        filter_estimate.S,
    )


def _get_smoother_estimate(smoother_estimate, i):
    return SmoothingResult(
        smoother_estimate.x_update[i],
        smoother_estimate.P_update[i],
        smoother_estimate.x_predict[i],
        smoother_estimate.P_predict[i],
        smoother_estimate.x_smooth[i],
        smoother_estimate.P_smooth[i],
        smoother_estimate.k[i],
        smoother_estimate.S[i],
    )


def _set_smoother_estimate(smoother_estimate, i, value):
    (
        smoother_estimate.x_update[i],
        smoother_estimate.P_update[i],
        smoother_estimate.x_predict[i],
        smoother_estimate.P_predict[i],
        smoother_estimate.x_smooth[i],
        smoother_estimate.P_smooth[i],
        smoother_estimate.k[i],
        smoother_estimate.S[i],
    ) = value


class SmoothingResult(NamedTuple):
    x_update: np.ndarray
    P_update: np.ndarray
    x_predict: np.ndarray
    P_predict: np.ndarray
    x_smooth: np.ndarray
    P_smooth: np.ndarray
    k: np.ndarray
    S: np.ndarray

    def __getitem__(self, i):
        return _get_smoother_estimate(self, i)

    def __setitem__(self, i, value):
        _set_smoother_estimate(self, i, value)

    def to_xarray(self, idx_name, time, states, outputs):
        coords = {
            idx_name: time,
            "states": states,
            "outputs": outputs,
        }
        ds = xr.Dataset(
            {
                "x_update": ((idx_name, "states"), self.x_update[:, :, 0]),
                "x_predict": ((idx_name, "states"), self.x_predict[:, :, 0]),
                "x_smooth": ((idx_name, "states"), self.x_smooth[:, :, 0]),
                "P_update": ((idx_name, "states", "states"), self.P_update),
                "P_predict": ((idx_name, "states", "states"), self.P_predict),
                "P_smooth": ((idx_name, "states", "states"), self.P_smooth),
                "k": ((idx_name, "outputs"), self.k[:, :, 0]),
                "S": ((idx_name, "outputs", "outputs"), self.S),
            },
            coords=coords,
        )
        return ds


def _allocate_simulation_res(n_timesteps, nx, ny):
    return SimulationResults(
        np.empty((n_timesteps, ny, 1)),
        np.empty((n_timesteps, nx, 1)),
    )


def _get_simulation_res(simulation_res, i):
    return SimulationResults(
        simulation_res.y[i],
        simulation_res.x[i],
    )


def _set_simulation_res(simulation_res, i, value):
    (
        simulation_res.y[i],
        simulation_res.x[i],
    ) = value


class SimulationResults(NamedTuple):
    y: np.ndarray
    x: np.ndarray

    def __getitem__(self, i):
        return _get_simulation_res(self, i)

    def __setitem__(self, i, value):
        _set_simulation_res(self, i, value)

    def to_xarray(self, idx_name, time, states, outputs):
        coords = {
            idx_name: time,
            "states": states,
            "outputs": outputs,
        }
        ds = xr.Dataset(
            {
                "x": ((idx_name, "states"), self.x[:, :, 0]),
                "y": ((idx_name, "outputs"), self.y[:, :, 0]),
            },
            coords=coords,
        )
        return ds


def _output_from_filter(filter_res, ny):
    return OutputEstimateResult(
        filter_res.x_update,
        filter_res.P_update,
        np.empty((filter_res.x_update.shape[0], ny, 1)),
        np.empty((filter_res.x_update.shape[0], ny, 1)),
    )


class OutputEstimateResult(NamedTuple):
    x: np.ndarray
    P: np.ndarray
    y: np.ndarray
    y_std: np.ndarray

    def __getitem__(self, i):
        return OutputEstimateResult(
            self.x[i],
            self.P[i],
            self.y[i],
            self.y_std[i],
        )

    def __setitem__(self, i, value):
        (
            self.x[i],
            self.P[i],
            self.y[i],
            self.y_std[i],
        ) = value

    def to_xarray(self, idx_name, time, states, outputs):
        coords = {
            idx_name: time,
            "states": states,
            "outputs": outputs,
        }
        ds = xr.Dataset(
            {
                "x": ((idx_name, "states"), self.x[:, :, 0]),
                "P": ((idx_name, "states", "states"), self.P),
                "y": ((idx_name, "outputs"), self.y[:, :, 0]),
                "y_std": ((idx_name, "outputs"), self.y_std[:, :, 0]),
            },
            coords=coords,
        )
        return ds


#### Pure python / numpy implementation of the Kalman filter ####
# they are not efficient par their own, and are therefore jitted by numba


def _solve_triu_inplace(A, b):
    for i in range(b.shape[0]):
        b[i] = (b[i] - A[i, i + 1 :] @ b[i + 1 :]) / A[i, i]
    return b


def _update(
    C, D, R, x, P, u, y, _Arru
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ny, _ = C.shape
    _Arru[:ny, :ny] = R
    _Arru[ny:, :ny] = P @ C.T
    _Arru[ny:, ny:] = P
    _, r_fact = np.linalg.qr(_Arru)
    S = r_fact[:ny, :ny]
    if ny == 1:
        k = (y - C @ x - D @ u) / S[0, 0]
        x = x + r_fact[:1, 1:].T * k
    else:
        k = _solve_triu_inplace(S, y - C @ x - D @ u)
        x = x + r_fact[:ny, ny:].T @ k
    P = r_fact[ny:, ny:]
    return x, P, k, S


def _predict(A, B0, B1, Q, x, P, u, dtu) -> tuple[np.ndarray, np.ndarray]:
    _, r = np.linalg.qr(np.vstack((P @ A.T, Q)))
    x = A @ x + B0 @ u + B1 @ dtu
    return x, r


def _kalman_step(
    x, P, u, dtu, y, states, _Arru
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ~np.isnan(y).any():
        x_up, P_up, k, S = _update(states.C, states.D, states.R, x, P, u, y, _Arru)
    else:
        x_up, P_up = x, P
        k = np.full((states.C.shape[0], 1), np.nan)
        S = np.full((states.C.shape[0], states.C.shape[0]), np.nan)
    x_pred, P_pred = _predict(
        states.A, states.B0, states.B1, states.Q, x_up, P_up, u, dtu
    )
    return x_up, P_up, k, S, x_pred, P_pred


def _unpack_states(states, i) -> States:
    return States(
        states.A[:, :, i],
        states.B0[:, :, i],
        states.B1[:, :, i],
        states.C,
        states.D,
        states.Q[:, :, i],
        states.R,
    )


def _log_likelihood(x0, P0, u, dtu, y, states) -> float:
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
    log_likelihood = 0.5 * n_timesteps * math.log(2.0 * math.pi)
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        if ~np.isnan(y_i).any():
            x, P, k, S = _update(
                states_i.C, states_i.D, states_i.R, x, P, u_i, y_i, _Arru
            )
            if ny == 1:
                Si = S[0, 0]
                log_likelihood += (
                    math.log(abs(Si.real) + abs(Si.imag)) + 0.5 * k[0, 0] ** 2
                )
            else:
                log_likelihood += np.linalg.slogdet(S)[1] + 0.5 * (k.T @ k)[0, 0]
        x, P = _predict(
            states_i.A, states_i.B0, states_i.B1, states_i.Q, x, P, u_i, dtu_i
        )
    return log_likelihood


def _filtering(x0, P0, u, dtu, y, states) -> FilteringResult:
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
    res = _allocate_filter_res(n_timesteps, nx, ny)
    for i in range(n_timesteps):
        y_i = np.ascontiguousarray(y[i]).reshape(-1, 1)
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        x_up, P_up, k, S, x, P = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        _set_filter_estimate(res, i, (x_up, P_up.T @ P_up, x, P.T @ P, k, S))
    return res


def _smoothing(x0, P0, u, dtu, y, states) -> SmoothingResult:
    n_timesteps = y.shape[0]
    res = _smooth_from_filter(_filtering(x0, P0, u, dtu, y, states))

    xf = res.x_smooth
    Pf = res.P_smooth
    xp = np.vstack((x0[None, ...], res.x_predict))
    Pp = np.vstack(((P0.T @ P0)[None, ...], res.P_predict))

    for i in range(n_timesteps - 2, -1, -1):
        G = np.linalg.solve(Pp[i + 1], states.A[:, :, i] @ Pf[i]).T
        xf[i, :, :] += G @ (xf[i + 1, :, :] - xp[i + 1, :, :])
        Pf[i, :, :] += G @ (Pf[i + 1, :, :] - Pp[i + 1, :, :]) @ G.T
    return res


def _simulate(x0, u, dtu, states) -> SimulationResults:
    x = x0
    n_timesteps = u.shape[0]
    ny, nx = states.C.shape
    res = _allocate_simulation_res(n_timesteps, nx, ny)
    for i in range(n_timesteps):
        u_i = np.ascontiguousarray(u[i]).reshape(-1, 1)
        dtu_i = np.ascontiguousarray(dtu[i]).reshape(-1, 1)
        states_i = _unpack_states(states, i)
        y = states_i.C @ x - states_i.D @ u_i
        x = (
            states_i.A @ x
            + states_i.B0 @ u_i
            + states_i.B1 @ dtu_i
            + states_i.Q @ np.random.randn(nx, 1)
        )
        _set_simulation_res(res, i, (y, x))
    res.y += states.R @ np.random.randn(n_timesteps, ny, 1)
    return res


def _estimate_output(x0, P0, u, dtu, y, states) -> OutputEstimateResult:
    n_timesteps = y.shape[0]
    ny = states.C.shape[0]
    res = _output_from_filter(_filtering(x0, P0, u, dtu, y, states), ny)
    for i in range(n_timesteps):
        x = res.x[i]
        P = res.P[i]
        states_i = _unpack_states(states, i)
        y = states_i.C @ x
        y_std = np.sqrt(states_i.C @ P @ states_i.C.T) + states_i.R
        res.y[i] = y
        res.y_std[i] = y_std

    return res


# All above will be jitted by numba, if available. Otherwise, the pure python / numpy
# implementation will be used.
try:
    from numba import jit_module

    jit_module(nopython=True, nogil=True, cache=True)
except ImportError:
    warnings.warn("Numba not installed, using pure python implementation")


@dataclass
class KalmanQR:
    """Bayesian Filter (Kalman Square Root filter)

    Parameters
    ----------
    ss : StateSpace
        State space model.

    Notes
    -----

    All the method except _proxy_params are jitted by numba, if available. Otherwise,
    the pure (but slower) python / numpy implementation will be used.
    """

    ss: StateSpace

    def _proxy_params(
        self,
        dt: pd.Series,
        vars: Sequence[pd.DataFrame],
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ):
        ss = self.ss
        ss.update()
        # use lru to avoid re_computation of discretization for identical dt
        dts, idx = np.unique(dt, return_inverse=True)
        A = np.zeros((ss.nx, ss.nx, dt.size))
        B0 = np.zeros((ss.nx, ss.nu, dt.size))
        B1 = np.zeros((ss.nx, ss.nu, dt.size))
        Q = np.zeros((ss.nx, ss.nx, dt.size))
        Ai, B0i, B1i, Qi = map(np.dstack, zip(*map(ss.discretization, dts)))
        A[:] = Ai[:, :, idx]
        B0[:] = B0i[:, :, idx]
        B1[:] = B1i[:, :, idx]
        Q[:] = Qi[:, :, idx]

        vars = [var.to_numpy() for var in vars]
        states = States(A, B0, B1, ss.C, ss.D, Q, ss.R)
        x0 = x0 if x0 is not None else ss.x0
        P0 = P0 if P0 is not None else ss.P0
        return tuple([x0, P0, *vars, states])

    def update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update the state and covariance of the current time step.

        Note that this function will not update the statespace model to ensure the
        consistency with the model parameters : to do so, use the `filter.ss.update`
        method.

        Parameters
        ----------
        x : np.ndarray
            State (or endogeneous) vector.
        P : np.ndarray
            Covariance matrix.
        u : np.ndarray
            Output (or exogeneous) vector.
        y : np.ndarray
            Measurement (or observation) vector.

        Returns
        -------
        x : np.ndarray
            State (or endogeneous) vector.
        P : np.ndarray
            Covariance matrix.
        """
        x, P = deepcopy(x), deepcopy(P)
        nx = self.ss.nx
        ny = self.ss.ny
        C = self.ss.C
        D = self.ss.D
        R = self.ss.R
        _Arru = np.zeros((nx + ny, nx + ny))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _update(C, D, R, x, P, u, y, _Arru)

    def predict(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the state and covariance of the next time step.

        Parameters
        ----------
        x : np.ndarray
            State (or endogeneous) vector.
        P : np.ndarray
            Covariance matrix.
        u : np.ndarray
            Output (or exogeneous) vector.
        dtu : np.ndarray
            Time derivative of the output vector.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Predicted state vector.
        np.ndarray
            Predicted covariance matrix.
        """
        x, P = deepcopy(x), deepcopy(P)
        A, B0, B1, Q = self.ss.discretization(dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _predict(A, B0, B1, Q, x, P, u, dtu)

    def log_likelihood(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ) -> float:
        """Compute the log-likelihood of the model.

        Parameters
        ----------
        dt : pd.Series
            Time steps.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.

        Returns
        -------
        float
            Log-likelihood of the model.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, states = self._proxy_params(dt, (u, dtu, y))
            return _log_likelihood(x0, P0, u, dtu, y, states)

    def filtering(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> FilteringResult:
        """Filter the data using the state space model and the Kalman filter.

        Parameters
        ----------
        dt : pd.Series
            Time steps.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.
        x0 : np.ndarray, optional
            Initial state vector, by default None. If None, the initial state vector
            provided by the statespace model is used.
        P0 : np.ndarray, optional
            Initial covariance matrix, by default None. If None, the initial covariance
            matrix provided by the statespace model is used.

        Returns
        -------
        FilteringResults
            Result of the filtering in the form of a namedtuple with the following
            fields:

            - x_update: np.ndarray[n_timesteps, nx, 1]
                Updated state vector.
            - P_update: np.ndarray[n_timesteps, nx, nx]
                Updated covariance matrix.
            - x_predict: np.ndarray[n_timesteps, nx, 1]
                Predicted state vector.
            - P_predict: np.ndarray[n_timesteps, nx, nx]
                Predicted covariance matrix.
            - k: np.ndarray[n_timesteps, ny, 1]
                Kalman gain.
            - S: np.ndarray[n_timesteps, ny, ny]
                Innovation covariance matrix.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, states = self._proxy_params(dt, (u, dtu, y), x0, P0)
            return _filtering(x0, P0, u, dtu, y, states)

    def smoothing(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> SmoothingResult:
        """Smooth the data using the state space model and the Kalman filter.

        Parameters
        ----------
        dt : pd.Series
            Time steps.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.
        x0 : np.ndarray, optional
            Initial state vector, by default None. If None, the initial state vector
            provided by the statespace model is used.
        P0 : np.ndarray, optional
            Initial covariance matrix, by default None. If None, the initial covariance
            matrix provided by the statespace model is used.

        Returns
        -------
        SmoothingResults
            Result of the smoothing in the form of a namedtuple with the following
            fields:

            - x_update: np.ndarray[n_timesteps, nx, 1]
                Updated state vector.
            - P_update: np.ndarray[n_timesteps, nx, nx]
                Updated covariance matrix.
            - x_predict: np.ndarray[n_timesteps, nx, 1]
                Predicted state vector.
            - P_predict: np.ndarray[n_timesteps, nx, nx]
                Predicted covariance matrix.
            - x_smooth: np.ndarray[n_timesteps, nx, 1]
                Smoothed state vector.
            - P_smooth: np.ndarray[n_timesteps, nx, nx]
                Smoothed covariance matrix.
            - k: np.ndarray[n_timesteps, ny, 1]
                Kalman gain.
            - S: np.ndarray[n_timesteps, ny, ny]
                Innovation covariance matrix.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, states = self._proxy_params(dt, (u, dtu, y), x0, P0)
            return _smoothing(x0, P0, u, dtu, y, states)

    def simulate(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> SimulationResults:
        """ Simulate the data using the state space model. The Kalman filter is not
        involved in this process.

        The formula reads:

        .. math::
            y = C x + D u + R \epsilon \\
            x = A x + B_0 u + B_1 \dot{u} + Q \eta

        where :math:`\epsilon` and :math:`\eta` are independent white noise with
        covariance matrices :math:`R` and :math:`Q` respectively.

        Parameters
        ----------
        dt : pd.Series
            Time steps.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        x0 : np.ndarray, optional
            Initial state vector, by default None. If None, the initial state vector
            provided by the statespace model is used.
        P0 : np.ndarray, optional
            Initial covariance matrix, by default None. If None, the initial covariance
            matrix provided by the statespace model is used.

        Returns
        -------
        SimulationResults
            Simulated data in the form of a namedtuple with the following fields:

            - x: np.ndarray[n_timesteps, nx, 1]
                State vector.
            - y: np.ndarray[n_timesteps, ny, 1]
                Output vector.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, _, u, dtu, states = self._proxy_params(dt, (u, dtu), x0, P0)
            return _simulate(x0, u, dtu, states)

    def estimate_output(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> OutputEstimateResult:
        """Estimate the output using the state space model and the Kalman filter.

        Parameters
        ----------
        dt : pd.Series
            Time steps.
        u : pd.DataFrame
            Output (or exogeneous) vector.
        dtu : pd.DataFrame
            Time derivative of the output vector.
        y : pd.DataFrame
            Measurement (or observation) vector.
        x0 : np.ndarray, optional
            Initial state vector, by default None. If None, the initial state vector
            provided by the statespace model is used.
        P0 : np.ndarray, optional
            Initial covariance matrix, by default None. If None, the initial covariance
            matrix provided by the statespace model is used.

        Returns
        -------
        OutputEstimate
            Estimated output in the form of a namedtuple with the following fields:

            - x: np.ndarray[n_timesteps, nx, 1]
                State vector.
            - P: np.ndarray[n_timesteps, nx, nx]
                Covariance matrix.
            - y: np.ndarray[n_timesteps, ny, 1]
                Output vector.
            - y_std: np.ndarray[n_timesteps, ny, 1]
                Standard deviation of the output vector.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u, dtu, y, states = self._proxy_params(dt, (u, dtu, y), x0, P0)
            return _estimate_output(x0, P0, u, dtu, y, states)
