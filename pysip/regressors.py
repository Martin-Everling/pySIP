from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
import scipy.optimize as scipy_optimize

from .params.parameters import Parameters
from .statespace.base import StateSpace
from .statespace_estimator import KalmanQR
from .utils.statistics import ttest


def _make_estimate(theta, data, estimator):
    estimator = deepcopy(estimator)
    estimator.ss.parameters.theta_free = theta
    res = estimator.estimate_output(*data)
    return res.y[None, ..., 0]


@dataclass
class Regressor:
    """A regressor use a statespace model and a bayesian filter to predict the states of
    the system given the exogeneous (or boundary) inputs and the initial conditions.

    This base class does not have the ability to estimate the parameters of the
    statespace model : see the derived classes for that.


    Parameters
    ----------
    ss : StateSpace()
        State-space model
    inputs : str or list of str, optional
        Input column names, by default None. If None, they are inferred from the
        state-space model.
    outputs : str or list of str, optional
        Output column names, by default None. If None, they are inferred from the
        state-space model.
    time_scale : str
        Time series frequency, e.g. 's': seconds, 'D': days, etc.
    """

    ss: StateSpace
    inputs: Optional[Union[str, Sequence[str]]] = None
    outputs: Optional[Union[str, Sequence[str]]] = None
    time_scale: str = "s"

    def __post_init__(self):
        self.estimator = KalmanQR(self.ss)
        if self.inputs is None:
            self.inputs = [node.name for node in self.ss.inputs]
        if self.outputs is None:
            self.outputs = [node.name for node in self.ss.outputs]
        self.states = [node.name for node in self.ss.states]
        if isinstance(self.inputs, str):
            self.inputs = [self.inputs]
        if isinstance(self.outputs, str):
            self.outputs = [self.outputs]

    @property
    def parameters(self) -> Parameters:
        return self.ss.parameters

    def prepare_data(self, df, with_outputs=True) -> Tuple[pd.DataFrame, ...]:
        """Prepare data for training

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data
        with_outputs : bool, optional
            Whether to return the outputs, by default True

        Returns
        -------
        DataFrame:
            time steps
        DataFrame:
            input data
        DataFrame:
            derivative of input data
        DataFrame:
            output data (filled with NaNs if with_outputs=False)
        """
        return self.ss.prepare_data(
            df, self.inputs, self.outputs if with_outputs else False, self.time_scale
        )

    def simulate(self, df: pd.DataFrame) -> xr.Dataset:
        """Stochastic simulation of the state-space model

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the inputs and outputs
        inputs : str or list of str, optional
            Input column names, by default None. If None, the regressor's `inputs`
            attribute is used.
        time_scale : str, optional
            Time series frequency, e.g. 's': seconds, 'D': days, etc., by default None.
            If None, the regressor's `time_scale` attribute is used.

        Returns
        -------
        xarray.Dataset
            Dataset containing the simulated outputs and states
        """

        dt, u, dtu, _ = self.prepare_data(df, with_outputs=False)
        return self.estimator.simulate(dt, u, dtu).to_xarray(
            df.index.name or "time", df.index, self.states, self.outputs
        )

    def estimate_output(
        self,
        df: pd.DataFrame,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        use_outputs: bool = True,
    ) -> xr.Dataset:
        """Estimate the output filtered distribution

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        x0 : numpy.ndarray, optional
            Initial state. If not provided, the initial state is taken from the
            state-space model defaults.
        P0 : numpy.ndarray, optional
            Initial state covariance. If not provided, the initial state covariance is
            taken from the state-space model defaults.
        use_outputs : bool, optional
            Whether to use the data outputs to do the estimation, by default True

        Returns
        -------
        xr.Dataset
            Dataset containing the estimated outputs and their covariance
        """

        dt, u, dtu, y = self.prepare_data(df, with_outputs=use_outputs)
        return self.estimator.estimate_output(dt, u, dtu, y, x0, P0).to_xarray(
            df.index.name or "time", df.index, self.states, self.outputs
        )

    def estimate_states(
        self,
        df: pd.DataFrame,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        smooth: bool = False,
        use_outputs: bool = True,
    ) -> xr.Dataset:
        """Estimate the state filtered/smoothed distribution

        Parameters
        ----------
        dt : float
            Sampling time
        u : array_like, shape (n_u, n_steps)
            Input data
        u1 : array_like, shape (n_u, n_steps)
            Forward finite difference of the input data
        y : array_like, shape (n_y, n_steps)
            Output data
        x0 : array_like, shape (n_x, )
            Initial state mean different from `ss.x0`
        P0 : array_like, shape (n_x, n_x)
            Initial state deviation different from `ss.P0`
        smooth : bool, optional
            Use RTS smoother

        Returns
        -------
        xr.Dataset
            Dataset containing the estimated states and their covariance
        """

        dt, u, dtu, y = self.prepare_data(df, with_outputs=use_outputs)

        return (
            self.estimator.smoothing(dt, u, dtu, y, x0, P0)
            if smooth
            else self.estimator.filtering(dt, u, dtu, y, x0, P0)
        ).to_xarray(df.index.name or "time", df.index, self.states, self.outputs)

    def _target(
        self,
        eta: np.ndarray,
        dt: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        y: np.ndarray,
        include_prior,
        include_penalty,
    ) -> float:
        """Evaluate the negative log-posterior

        Parameters
        ----------
        eta : array_like, shape (n_eta, )
            Unconstrained parameters
        dt : float
            Sampling time
        u : array_like, shape (n_u, n_steps)
            Input data
        dtu : array_like, shape (n_u, n_steps)
            Forward finite difference of the input data
        y : array_like, shape (n_y, n_steps)
            Output data

        Returns
        -------
        log_posterior : float
            The negative log-posterior
        """
        estimator = deepcopy(self.estimator)
        estimator.ss.parameters.eta_free = eta
        log_likelihood = estimator.log_likelihood(dt, u, dtu, y)

        # Determine log posterior
        log_posterior = log_likelihood
        if include_prior:
            log_posterior -= estimator.ss.parameters.prior
        if include_penalty:
            log_posterior += estimator.ss.parameters.penalty

        return log_posterior

    def _target_n_step(
        self,
        eta: np.ndarray,
        dt: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        y: np.ndarray,
        include_prior,
        include_penalty,
        k_simulations: int,
        n_simulation: int,
        simulation_weights: np.ndarray | None,
        alpha: float | None = None,
    ) -> float:
        """Evaluate the negative log-posterior

        Parameters
        ----------
        eta : array_like, shape (n_eta, )
            Unconstrained parameters
        dt : float
            Sampling time
        u : array_like, shape (n_u, n_steps)
            Input data
        dtu : array_like, shape (n_u, n_steps)
            Forward finite difference of the input data
        y : array_like, shape (n_y, n_steps)
            Output data

        Returns
        -------
        log_posterior : float
            The negative log-posterior
        """
        # Determine default alpha (weighting factor prediction versus simulations)
        if alpha is None:
            alpha = len(y) / (len(y) + k_simulations * n_simulation)

        # First run the full system
        estimator = deepcopy(self.estimator)
        estimator.ss.parameters.eta_free = eta
        log_likelihood, x, P = estimator.log_likelihood_with_outputs(dt, u, dtu, y)

        # Handle the simulations
        i_start_simulations = np.floor(
            np.linspace(0, len(y) - n_simulation, k_simulations)
        ).astype(int)
        log_likelihood_simulations = np.zeros((k_simulations))
        for i_simulation in range(k_simulations):
            i_start_sim = i_start_simulations[i_simulation]
            i_end_sim = i_start_sim + n_simulation

            log_likelihood_simulations[i_simulation] += estimator.log_likelihood(
                dt.iloc[i_start_sim:i_end_sim],
                u.iloc[i_start_sim:i_end_sim],
                dtu.iloc[i_start_sim:i_end_sim],
                y.iloc[i_start_sim:i_end_sim],
                x[i_start_sim, :].reshape([-1, 1]),
                P[i_start_sim, :],
                weights=simulation_weights,
                use_outputs=False,
            )
        log_likelihood = log_likelihood * alpha + (1 - alpha) * np.mean(
            log_likelihood_simulations
        )

        # Determine log posterior
        log_posterior = log_likelihood
        if include_prior:
            log_posterior -= estimator.ss.parameters.prior
        if include_penalty:
            log_posterior += estimator.ss.parameters.penalty

        return log_posterior

    def fit(
        self,
        df: pd.DataFrame,
        options: dict = None,
        *,
        init: Literal["unconstrained", "prior", "zero", "fixed", "value"] = "fixed",
        hpd: float = 0.95,
        jac="3-point",
        method="BFGS",
        optimizer: Optional[str] = None,
        optimizer_kwargs: Optional[dict] = None,
        include_prior: bool = True,
        include_penalty: bool = True,
        k_simulations: Optional[int] = None,
        n_simulation: Optional[int] = None,
        simulation_weights: Optional[np.ndarray] = None,
        alpha: float | None = None,
        **minimize_options,
    ) -> Union[pd.DataFrame, pd.DataFrame, dict]:
        """Estimate the parameters of the state-space model.

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        outputs : str or list of str, optional
            Output name(s)
        inputs : str or list of str, optional
            Input name(s)
        options : dict, optional, deprecated
            Options for the minimization method. You should use named arguments instead.
            Usage of this argument will raise a warning and will be removed in the
            future.
        init : str, optional
            Method to initialize the parameters. Options are:
            - 'unconstrained': initialize the parameters in the unconstrained space
            - 'prior': initialize the parameters using the prior distribution
            - 'zero': initialize the parameters to zero
            - 'fixed': initialize the parameters to the fixed values
            - 'value': initialize the parameters to the given values
        hpd : float, optional
            Highest posterior density interval. Used only when `init='prior'`.
        k_simulations: int, optional
            Optionally, the amount of simulations to include in the target function
        n_simulation: int, optional
            Optionally, the simulation length for all simulations in the target function
        simulation_weights: numpy.ndarray, optional
            Optionally, provide weights for the simulations, e.g. a decreasing weight
            over the simulation steps.
        minimize_options : dict, optional
            Options for the minimization method. See `scipy.optimize.minimize` for
            details. Compared to the original `scipy.optimize.minimize` function, the
            following options are set by default:
            - `method='BFGS'`
            - `jac='3-point'`
            - `disp=True`
            - `gtol=1e-4`

        Returns
        -------
        pandas.DataFrame
            Dataframe with the estimated parameters, their standard deviation, the
            p-value of the t-test and penalty values.
        pandas.DataFrame
            Dataframe with the correlation matrix of the estimated parameters.
        dict
            Results object from the minimization method. See `scipy.optimize.minimize`
            for details.
        """
        if options is not None:
            warnings.warn(
                "Use of the options argument is deprecated and will raise an error in "
                "the future. Prefer using named arguments (kwargs) instead.",
                DeprecationWarning,
            )
            minimize_options.update(options)
        minimize_options = {"disp": True, "gtol": 1e-4} | minimize_options

        self.parameters.eta_free = self.parameters.init_parameters(1, init, hpd)
        data = self.prepare_data(df)

        # Set arguments and target function
        if k_simulations is None and n_simulation is None:
            args = (*data, include_prior, include_penalty)
            target = self._target
        elif k_simulations is not None and n_simulation is not None:
            args = (
                *data,
                include_prior,
                include_penalty,
                k_simulations,
                n_simulation,
                simulation_weights,
                alpha,
            )
            target = self._target_n_step
        else:
            raise KeyError(
                "Please provide both a number of simulations "
                '"k_simulations" and the simulation length "n_simulation" '
                "arguments when including simulations in the target "
                "function."
            )

        # Minimize/Optimize
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if optimizer is None:
            results = minimize(
                fun=target,
                x0=self.parameters.eta_free,
                args=args,
                method=method,
                jac=jac,
                options=minimize_options,
            )
        elif optimizer.lower() == "shgo":
            # Advised is to run shgo only with all parameters bounded and "L-BFGS-B"
            all_bounded = all(
                [
                    all([b is not None for b in p.bounds])
                    for p in self.parameters.parameters_free
                ]
            )
            if not all_bounded:
                raise KeyError(
                    "Please make sure all parameters are bounded when running shgo."
                )
            bounds = [(-1e1, 1e1) for _ in self.parameters.eta_free]
            # Advised is to use "L-BFGS-B" when running
            bounded_methods = ["nelder-mead", "l-bfgs-b", "tnc", "slsqp", "powell"]
            if not any([method.lower() == _method for _method in bounded_methods]):
                raise KeyError(
                    f"Please make sure the minimization method is bounded when"
                    f" running shgo, any of: {bounded_methods}."
                )

            # Set defaults
            optimizer_kwargs["sampling_method"] = optimizer_kwargs.get(
                "sampling_method", "sobol"
            )
            # Run
            min_kwargs = {"method": method, "jac": jac, "options": minimize_options}
            results_opt = scipy_optimize.shgo(
                func=target,
                bounds=bounds,
                args=args,
                minimizer_kwargs=min_kwargs,
                **optimizer_kwargs,
            )
            results = minimize(
                fun=target,
                x0=results_opt.x,
                args=args,
                method=method,
                jac=jac,
                bounds=bounds,
                options=minimize_options,
            )
        elif optimizer.lower() == "dual_annealing":
            # Set defaults
            optimizer_kwargs["maxiter"] = optimizer_kwargs.get("maxiter", 100)
            optimizer_kwargs["maxfun"] = optimizer_kwargs.get("maxfun", 1e4)
            # Run
            min_kwargs = {"method": method, "jac": jac, "options": minimize_options}
            results_opt = scipy_optimize.dual_annealing(
                func=target,
                bounds=[(-1e50, 1e50) for _ in self.parameters.eta_free],
                args=args,
                x0=self.parameters.eta_free,
                minimizer_kwargs=min_kwargs,
                **optimizer_kwargs,
            )
            results = minimize(
                fun=target,
                x0=results_opt.x,
                args=args,
                method=method,
                jac=jac,
                options=minimize_options,
            )
        elif optimizer.lower() == "basinhopping":
            # Set defaults
            optimizer_kwargs["niter"] = optimizer_kwargs.get("niter", 10)
            # Run
            min_kwargs = {
                "args": args,
                "method": method,
                "jac": jac,
                "options": minimize_options,
            }
            results_opt = scipy_optimize.basinhopping(
                func=target,
                x0=self.parameters.eta_free,
                minimizer_kwargs=min_kwargs,
                **optimizer_kwargs,
            )
            results = results_opt.lowest_optimization_result
        else:
            raise KeyError()

        self.parameters.eta_free = results.x

        if hasattr(results, "hess_inv"):
            hess_inv = results.hess_inv
            if not isinstance(hess_inv, np.ndarray):
                hess_inv = hess_inv.todense()

            # inverse jacobian of the transform eta = f(theta)
            inv_jac = np.diag(1.0 / np.array(self.parameters.eta_jacobian))

            # covariance matrix in the constrained space (e.g. theta)
            cov_theta = inv_jac @ hess_inv @ inv_jac

            # standard deviation of the constrained parameters
            sig_theta = np.sqrt(np.diag(cov_theta)) * self.parameters.scale
            sig_theta_test = ttest(self.parameters.theta_free, sig_theta, len(data[0]))
            inv_sig_theta = np.diag(1.0 / np.sqrt(np.diag(cov_theta)))

            # correlation matrix of the constrained parameters
            corr_matrix = inv_sig_theta @ cov_theta @ inv_sig_theta
        else:
            sig_theta = [np.nan for _ in self.parameters.theta_free]
            sig_theta_test = [np.nan for _ in self.parameters.theta_free]
            corr_matrix = np.nan
        if hasattr(results, "jac"):
            jac = results.jac
        else:
            jac = [np.nan for _ in self.parameters.theta_free]

        df = pd.DataFrame(
            data=np.vstack(
                [
                    self.parameters.theta_free,
                    sig_theta,
                    sig_theta_test,
                    np.abs(jac),
                    np.abs(self.parameters.d_penalty),
                ]
            ).T,
            columns=["θ", "σ(θ)", "pvalue", "|g(η)|", "|dpen(θ)|"],
            index=self.parameters.names_free,
        )
        df_corr = pd.DataFrame(
            data=corr_matrix,
            index=self.parameters.names_free,
            columns=self.parameters.names_free,
        )

        self.summary_ = df
        self.corr_ = df_corr
        self.results_ = results

        return df, df_corr, results

    def eval_residuals(
        self,
        df: pd.DataFrame,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ) -> xr.Dataset:
        """Compute the standardized residuals

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        outputs : str or list of str, optional
            Output name(s)
        inputs : str or list of str, optional
            Input name(s)
        x0 : numpy.ndarray, optional
            Initial state. If not provided, the initial state is taken from the
            state-space model defaults.
        P0 : numpy.ndarray, optional
            Initial state covariance. If not provided, the initial state covariance is
            taken from the state-space model defaults.

        Returns
        -------
        xr.Dataset
            Dataset containing the residuals and their standard deviation
        """

        dt, u, dtu, y = self.prepare_data(df)
        res = self.estimator.filtering(dt, u, dtu, y, x0, P0).to_xarray(
            df.index.name or "time", df.index, self.states, self.outputs
        )
        return res[["k", "S"]].rename({"k": "residual", "S": "residual_std"})

    def log_likelihood(self, df: pd.DataFrame) -> Union[float, np.ndarray]:
        """
        Evaluate the log-likelihood of the model.

        Parameters
        ----------
        df : pandas.DataFrame
            Data.
        outputs : str or list of str, optional
            Outputs name(s). If None, all outputs are used.
        inputs : str or list of str, optional
            Inputs name(s). If None, all inputs are used.
        pointwise : bool, optional
            Evaluate the log-likelihood pointwise.

        Returns
        -------
        float or numpy.ndarray
            Negative log-likelihood or predictive density evaluated point-wise.
        """

        dt, u, dtu, y = self.prepare_data(df)
        return self.estimator.log_likelihood(dt, u, dtu, y)

    def log_posterior(
        self,
        df: pd.DataFrame,
    ) -> Union[float, np.ndarray]:
        """Evaluate the negative log-posterior, defined as

        .. math::
            -\\log p(\\theta | y) = -\\log p(y | \\theta) - \\log p(\\theta) + \\log

        with :math:`y` the data, :math:`\\theta` the parameters, :math:`p(y | \\theta)`
        the likelihood, :math:`p(\\theta)` the prior and :math:`\\log p(\\theta | y)`
        the log-posterior.

        Parameters
        ----------
        df : pandas.DataFrame
            Training data

        Returns
        -------
        log_posterior : float
            The negative log-posterior
        """
        return (
            self.log_likelihood(df)
            - self.ss.parameters.prior
            + self.ss.parameters.penalty
        )

    def predict(
        self,
        df: pd.DataFrame,
        tnew: Union[np.ndarray, pd.Series] = None,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        smooth: bool = False,
        use_outputs: bool = True,
    ) -> xr.Dataset:
        """State-space model output prediction

        Parameters
        ----------
        df : pandas.DataFrame
            Training data
        tnew : numpy.ndarray or pandas.Series, optional
            New time instants
        x0 : numpy.ndarray, optional
            Initial state. If not provided, the initial state is taken from the
            state-space model.
        P0 : numpy.ndarray, optional
            Initial state covariance. If not provided, the initial state covariance is
            taken from the state-space model.
        smooth : bool, optional
            If True, the Kalman smoother is used instead of the Kalman filter
        use_outputs : bool, optional
            If True, the outputs are used to do the estimation. Default is False.

        Returns
        -------
        xr.Dataset
            Dataset containing the predicted outputs and their covariance

        """

        if self.ss.ny > 1:
            raise NotImplementedError
        if tnew is not None:
            itp_df = pd.concat(
                [df, df.reindex(tnew).drop(df.index, errors="ignore")]
            ).sort_index()
            itp_df[self.inputs] = itp_df[self.inputs].interpolate(method="linear")
        else:
            itp_df = df.copy()
        ds = self.estimate_states(
            itp_df, smooth=smooth, x0=x0, P0=P0, use_outputs=use_outputs
        )
        idx_name = df.index.name or "time"
        if tnew is not None:
            ds = ds.sel(**{idx_name: tnew})

        if tnew is None:
            u = itp_df[self.inputs]
        else:
            u = itp_df.loc[tnew, self.inputs]
        u_reshaped = np.expand_dims(u.values, axis=2)
        ds["y_mean"] = (
            (idx_name, "outputs"),
            (self.ss.C @ ds.x.values.reshape(-1, self.ss.nx, 1))[..., 0]
            + (self.ss.D @ u_reshaped)[..., 0],
        )

        ds["y_std"] = (
            (idx_name, "outputs"),
            np.sqrt(
                np.einsum(
                    "...yx,...xx,...xy->...y",
                    self.ss.C,
                    ds.P.values,
                    self.ss.C.T,
                )
            )
            + self.ss.R,
        )
        return ds

    def posterior_predictive(self, df, **kwargs):
        """Sample from the posterior predictive distribution

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the inputs and outputs

        Returns
        -------
        xarray.Dataset

        Raises
        ------
        RuntimeError
            If the model has not been sampled yet
        """
        try:
            trace = self.trace
        except AttributeError:
            raise RuntimeError("No trace available : run `sample` first")
        parameters = trace.posterior.to_dataframe().to_numpy()
        data = self.prepare_data(df)

        data = self.prepare_data(df)
        with ProcessPoolExecutor() as executor:
            results = np.vstack(
                list(
                    executor.map(
                        partial(_make_estimate, data=data, estimator=self.estimator),
                        parameters,
                    )
                )
            )
        results = results.reshape(
            trace.posterior.chain.shape[0],
            trace.posterior.draw.shape[0],
            *results.shape[1:],
        )

        idx_name = df.index.name or "time"
        return xr.DataArray(
            results,
            dims=("chain", "draw", idx_name, "outputs"),
            coords={
                "chain": trace.posterior.chain,
                "draw": trace.posterior.draw,
                "outputs": self.outputs,
                idx_name: df.index,
            },
        ).to_dataset("outputs")

    @property
    def trace(self):
        main_trace = getattr(self, "_trace", None)
        prior_trace = getattr(self, "_prior_trace", None)

        if main_trace is not None:
            if prior_trace is not None:
                main_trace.extend(prior_trace)
            return main_trace
        if prior_trace is not None:
            return prior_trace
        raise AttributeError("No trace available")
