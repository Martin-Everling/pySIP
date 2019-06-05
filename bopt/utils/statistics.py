import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as tStudent
from scipy.stats import norm as normal
from scipy.stats import chi2
from scipy.signal import correlate


def ttest(theta, sigma, N):
    """t-test: statistical signifiance of the maximum likelihood estimates
    :math:`\\hat{\\theta}`

    The following hypothesis is tested:
    :math:`H_0: \\hat{\\theta}=0` against :math:`H_1: \\hat{\\theta} \neq 0`

    Under the null hypothesis :math:`H_0`, the test quantity
    :math:`z=\\frac{\\hat{\\theta}}{\\sigma_{\\hat{\\theta}}}
    follows a t-distribution, centered on the null hypothesis value, with
    :math:`N-N_p` degrees of freedom where N is the sample size and
    :math:`N_p` the number of estimated parameters

    The null hypothesis :math:`H_0`is rejected if the :math:`p_{value}` of the
    test quantity :math:`z` is less or equal to the signifiance level defined.
    """
    if not isinstance(theta, np.ndarray):
        theta = np.array(theta)

    if not isinstance(sigma, np.ndarray):
        sigma = np.array(sigma)

    idx = sigma > 0
    if np.any(idx is False):
        print("Negative standard deviation in the t-test, the p-value is NaN")

    Np = len(theta)
    pvalue = np.full(Np, np.nan)
    pvalue[idx] = 2 * (1 - tStudent.cdf(theta[idx] / sigma[idx], N - Np))

    return pvalue


def ccf(x, y=None, n_lags=None, ci=0.95, half=True, show_zero=True):
    """Cross correlation (CCF) between two time-series x and y.

    If only one time-series is specified, the auto correlation of x is computed.

    Args:
        x:  time series of length N
        y: (optional) time series of length N
        n_lags: number of lags, if n_lags = None --> n_lags = N - 1
        ci: confidence interval [0, 1], by default 95%
        half: if True, plot only the positive lags
        show_zero: if True the lag at 0 is plotted (acf[0] = 1)
    """

    if y is None:
        y = x

    N = len(x)
    n_lags = int(n_lags) if n_lags else N - 1

    if len(x) != len(y):
        raise ValueError('x and y must have equal length')

    if n_lags >= N or n_lags < 0:
        raise ValueError(f'maxlags must belong to [1 {n_lags - 1}]')

    lags = np.arange(-N + 1, N)

    correlation_coeffs = (
        correlate(x - np.mean(x), y - np.mean(y), method='direct')
        / (np.std(x) * np.std(y) * N)
    )

    t = int(show_zero) if half else 1 + n_lags
    cut = range(N - t, N + n_lags)

    # confidence interval lines
    confidence = np.ones(len(cut)) * normal.ppf((1 + ci) / 2) / np.sqrt(N)

    return lags[cut], correlation_coeffs[cut], confidence


def check_ccf(lags, coeffs, confidence, threshold=0.95):
    """CCF Test
    Given cross-correlation coefficients values and corresponding confidence
    interval values, check that at least a minimum of the absolute ccf values
    fits within the confidence intervals.

    Args:
        lags, coeffs, confidences: outputs of the ccf function
        threshold (float): Threshold level (0.0 - 1.0)

    Returns:
        Boolean. True if the test has succeed, False otherwise.
    """
    in_band = np.sum(np.abs(coeffs) < confidence) / len(coeffs)
    return in_band >= threshold


def plot_ccf(lags, coeffs, confidence):
    """Plot cross-correlation coefficients

    Args:
        lags, coeffs, confidences: outputs of the ccf function

    Returns:
        Matplotlib axe
    """

    _, ax = plt.subplots()

    _, stemlines, baseline = ax.stem(lags, coeffs, '-', markerfmt='None')
    ax.fill_between(lags, -confidence, confidence,
                    facecolor='r', edgecolor='None', alpha=0.2)

    ax.set_xlabel('Lags', fontsize=12)
    ax.set_ylabel('Normalized Correlation Coefficients', fontsize=12)
    plt.setp(baseline, color='b', linewidth=1)
    plt.setp(stemlines, color='b', linewidth=1.5)

    return ax


def cpgram(ts):
    """Cumulative periodogram with 95% confidence intervals

    Args:
        ts: residual time series

    Notes:
        Adapted from the `R cpgram function
        <https://www.rdocumentation.org/packages/stats/versions/3.5.3/topics/cpgram>`_
    """

    spectrum = np.fft.fft(ts)
    n = len(ts)
    y = (np.sqrt(spectrum.real ** 2 + spectrum.imag ** 2)) ** 2 / n
    if n % 2 == 0:
        n -= 1
        y = y[:n]

    freq = np.linspace(0, 0.5, n, endpoint=True)
    crit = 1.358 / (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))

    return y, freq, crit


def check_cpgram(y, freq, crit, threshold=0.95):
    """Cpgram test

    Given a spectrum and a criterion, check that the cumulative periodogram
    values fits within 95% confidence intervals

    Args:
        y, freq, crit: outputs of the cpgram function
        threshold (float): Threshold level (0.0 - 1.0)

    Returns:
        Boolean. True if the test has succeed, False otherwise.
    """
    cum_sum = np.cumsum(y) / np.sum(y) - 2 * freq
    in_band = np.sum(np.abs(cum_sum) < crit) / len(freq)
    return in_band >= threshold


def plot_cpgram(y, freq, crit):
    """Plot cumulative periodogram with confidence intervals

    Args:
        lags, coeffs, confidences: outputs of the ccf function

    Returns:
        Matplotlib axe
    """
    _, ax = plt.subplots()

    ax.plot(freq, np.cumsum(y) / np.sum(y))
    ax.fill_between(freq, 2 * freq - crit, 2 * freq + crit,
                    facecolor='r', edgecolor='None', alpha=0.2)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Normalized Nyquist Frequency')
    ax.set_ylabel('Cumulated Periodogram')

    return ax


def autocorrf(x):
    """Compute the unbiased autocorrelation function by using the Fast Fourier
    Transform (FFT) with zeros padding

    Argument
    --------
    x: array-like
      x is a one dimensional time series

    Return
    ------
    acorr: the normalized autocorrelation function
    """
    # check for one dimensional time series
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("the time series x must be one dimensional")

    # length of the time series
    n = len(x)

    # find the next power of two for FFT computation efficiency
    next_pow_2 = 2**np.ceil(np.log2(n)).astype("int")

    # compute the FFT of the signal with the mean removed
    f = np.fft.fft(x - np.mean(x), n=2 * next_pow_2)

    # compute the autocorrelation
    acorr = np.fft.ifft(f * np.conjugate(f))[:n].real

    # unbiased estimator
    acorr /= np.arange(n, 0, -1)

    # Normalize to unity (divide by variance gives the same results)
    acorr /= acorr[0]

    return acorr


def autocovf(x):
    """Compute the unbiased autocovariance function by using the Fast Fourier
    Transform (FFT) with zeros padding

    Argument
    --------
    x: array-like
      x is a one dimensional time series

    Return
    ------
    acov: the normalized autocorrelation function
    """
    # estimate the autocorrelation function
    acorr = autocorrf(x)

    # multiply by the variance with zero degree of freedom (sample variance)
    # to obtain the autocorrelation function
    return acorr * np.var(x)


def likelihood_ratio_test(ll, ll_sub, N, N_sub):
    """Compute the pvalue of the likelihood ratio test

    The likelihood ratio test compares two nested models, M_sub and M,
    such that M_sub ⊂ M, e.g. M_sub can be obtained by setting some
    parameters of M to 0.

    Parameters
    ----------
    ll : float
        log-likelihood of the model M
    ll_sub : float
        log-likelihood of the sub-model, M_sub
    N : int
        number of free parameters in M
    N_sub : int
        number of free parameters in M_sub

    """
    if N_sub > N:
        raise ValueError("The sub-model must have less parameters "
                         "than the larger model")

    return chi2.sf(-2.0 * (np.abs(ll_sub) - np.abs(ll)), N - N_sub)
