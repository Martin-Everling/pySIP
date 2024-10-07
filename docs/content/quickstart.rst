Quickstart
==========

Installation
------------

Two options are available:

1. Install the package from git main branch:

.. code-block:: bash

    pip install git+https://github.com/locie/pySIP

2. Install the package from source:

.. code-block:: bash

    git clone https://github.com/locie/pySIP
    cd pysip
    pip install .

Note that these two options need git on your system.

Usage
-----

As a first example, we will use the `pysip` package to estimate the parameters
of a simple model from a set of artificial data, generated from a sinusoidal
function with additive Gaussian noise.

The model will be a Matern32 kernel.

Data preparation
~~~~~~~~~~~~~~~~

First, we have to create the data, and store it in a `pandas.DataFrame` object.

.. code:: python

    >>> import numpy as np
    >>> from pysip import Regressor
    >>> from pysip.statespace import Matern32

    >>> N = 200
    >>> t = np.linspace(0, 1, N)
    >>> y = np.sin(12 * t) + 0.66 * np.cos(25 * t) + np.random.randn(N) * 0.1
    >>> df = pd.DataFrame(index=t, data=y, columns=["y"])

In that simple case, we only have one output variable, and no internal state or
input but the package can handle multiple inputs, states and outputs.

Parameters configuration and model initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameters are defined as a list of dictionaries, each dictionary
containing the following keys:

- ``name``: the name of the parameter
- ``value``: the initial value of the parameter (optional, default: 0)
- ``bounds``: the bounds of the parameter (optional, default: no bounds)
- ``scale`` (optional): the scale of the parameter (default: 1)
- ``transform``: the transformation to apply to the parameter (optional, deduced
  from the bounds if not provided).
- ``prior``: the prior distribution of the parameter (optional, but needed for
  Bayesian parameter estimation).

Our model needs three parameters: the Matern32 length scale, the Matern32 scale
and the noise standard deviation. First, no need to include priors : they are
only needed for Bayesian parameter estimation.

.. code:: python

    >>> par = [
    ...     dict(name="mscale", value=1.11, bounds=(0, None)),
    ...     dict(name="lscale", value=0.15, bounds=(0, None)),
    ...     dict(name="sigv", value=0.1, bounds=(0, None)),
    ... ]
    >>> model = Matern32(par)
    Matern32(hold_order=0, method='mfd', name='Matern32')

Model evaluation
~~~~~~~~~~~~~~~~

The model, using the provided parameters, can be evaluated using a statespace
estimator. The one provided by the package is a (fast) Square root Kalman
filter. The `Regressor` class is used to evaluate the model via that estimator
(estimate the state, filter the output or evaluate the state or outputs at
in-between time points).

The output and input names are provided at the initialization of the
`Regressor`. Most of the `Regressor` methods return a `xarray.Dataset` object :
they are a collection of numpy arrays with named dimensions that allow easy
plotting and post-processing. More detail are available in the method docstrings
(available in the API part of the documentation).

.. code:: python

    >>> reg = Regressor(model, outputs=["y"])
    >>> nofit_estimation = reg.estimate_output(df)


Simple (frequentist) parameter estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provided a first guess for the parameters. We have seen that the evaluation
result is far from the data, as our guess is away from the true parameters. We
can now estimate the parameters using the `fit` method of the `Regressor` class.
This method will use the `scipy.optimize.minimize` function to find the
parameters that minimize the negative log-likelihood of the model given the
data.
After that, the model (and the regressor)

.. code:: python

    >>> summary = reg.fit(df)[0]
    >>> summary
                θ      σ(θ)    pvalue    |g(η)|     |dpen(θ)|
    mscale  1.041769  1.041769  0.322431  0.000020  9.214196e-17
    lscale  0.142355  0.142355  0.322431  0.000020  4.934663e-15
    sigv    0.090430  0.090430  0.322431  0.000003  1.222867e-14
    >>> fit_estimation = reg.estimate_output(df)

We can see how are the prediction compared to the data

.. figure:: ../_static/figures/quickstart_matern_estimation.svg
    :align: center

    Fit comparison

The starting values have been chosen to be not too close to the true values, so
we can see that the difference between the fit and the initial values.

Bayesian parameter estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This part of the code has been removed to reduce the package dependencies