import numpy as np


def random_recursive_signal(n_samples: np.int32, start: np.float64,
                             scale: np.float64 = 1.0) -> np.ndarray:
    """
    Generates a synthetic time series by accumulating Gaussian noise step by step.

    Each sample is computed as the previous value plus a random increment drawn
    from a normal distribution. This produces a random walk that resembles realistic
    signals such as power demand curves.

    Parameters
    ----------
    n_samples : int
        Number of samples in the output signal.
    start : float
        Initial value of the signal (first sample, without any noise applied).
    scale : float, optional
        Standard deviation of the Gaussian noise added at each step. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) with the generated signal.
    """
    signal = np.zeros(shape=n_samples, dtype=np.float64)
    noise = np.random.normal(loc=0.0, scale=scale, size=n_samples - 1)

    signal[0] = start
    for i in range(1, n_samples):
        signal[i] = signal[i - 1] + noise[i - 1]

    return signal


def scale_signal(signal: np.ndarray, method: str = 'MinMax') -> np.ndarray:
    """
    Normalizes a signal using one of the supported scaling methods.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to normalize.
    method : str, optional
        Normalization method to apply. Supported values are:
        - 'MinMax': rescales the signal to the range [0, 1].
        - 'STD': standardizes the signal to zero mean and unit variance.
        Comparison is case-insensitive. Defaults to 'MinMax'.

    Returns
    -------
    np.ndarray
        Normalized signal with the same shape as the input.

    Raises
    ------
    ValueError
        If the specified method is not recognized.
    """
    method_lower = method.lower()

    match method_lower:
        case 'minmax':
            _min, _max = np.min(a=signal), np.max(a=signal)
            return (signal - _min) / (_max - _min)

        case 'std':
            mu, sigma = np.mean(a=signal), np.std(a=signal)
            return (signal - mu) / sigma

        case _:
            raise ValueError(f"Normalization method '{method}' is not recognized.")


def moving_average_filter(signal: np.ndarray, window_size: np.int32 = 7) -> np.ndarray:
    """
    Smooths a signal by applying a moving average filter.

    The filter replaces each sample with the mean of the surrounding window,
    effectively removing high-frequency noise. The signal length is preserved
    by padding the end with the last known value before convolution.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to filter.
    window_size : int, optional
        Number of samples to include in each averaging window. Must be greater
        than zero. Defaults to 7.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as the input.

    Raises
    ------
    ValueError
        If window_size is not a positive integer.
    """
    if window_size <= 0:
        raise ValueError("The window size must be a positive integer.")

    p_size = window_size - 1

    # Pad the signal at the end by repeating the last value to preserve output length
    signal_padded = np.zeros(shape=signal.shape[0] + p_size, dtype=np.float64)
    signal_padded[:signal.shape[0]] = signal
    signal_padded[signal.shape[0]:] = signal[-1]

    output_signal = np.zeros(shape=signal.shape[0], dtype=np.float64)
    for i in range(signal.shape[0]):
        output_signal[i] = np.mean(a=signal[i : i + window_size])

    return output_signal


def generate_demand(n_samples: np.int32, start: np.float64 = None,
                    scale: np.float64 = None, apply_filtering: bool = True) -> np.ndarray:
    """
    Generates a realistic normalized power demand signal.

    The signal is constructed as a random walk (via Gaussian noise accumulation),
    then rescaled to [0, 1] using Min-Max normalization. Optionally, a moving
    average filter is applied to smooth out high-frequency fluctuations, producing
    a more natural-looking demand curve.

    Parameters
    ----------
    n_samples : int
        Number of time steps in the demand signal.
    start : float, optional
        Starting power value. If not provided, a random value in [0, 100] is used.
    scale : float, optional
        Noise scale for the random walk. If not provided, defaults to 1.0.
    apply_filtering : bool, optional
        Whether to apply a moving average filter after normalization. Defaults to True.

    Returns
    -------
    np.ndarray
        Normalized demand signal of shape (n_samples,), with values in [0, 1].
    """
    demand_signal = random_recursive_signal(
        n_samples=n_samples,
        start=start if start is not None else np.random.uniform(low=0.0, high=100.0),
        scale=scale if scale is not None else 1.0
    )

    demand_signal_norm = scale_signal(signal=demand_signal, method='MinMax')

    if apply_filtering:
        return moving_average_filter(signal=demand_signal_norm)

    return demand_signal_norm
