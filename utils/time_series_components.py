import numpy as np

BASELINE = 10
AMPLITUDE = 40
SLOPE = 0.005
NOISE_LEVEL = 3
PERIOD = 365


def trend(time: np.array, slope: float):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(
        season_time < 0.1, np.cos(season_time * 6 * np.pi), 2 / np.exp(9 * season_time)
    )


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def create_time_series_with_noise(time: np.array):
    # Create the series
    series = (
        BASELINE
        + trend(time, SLOPE)
        + seasonality(time, period=PERIOD, amplitude=AMPLITUDE)
    )
    # Update with noise
    series += noise(time, NOISE_LEVEL, seed=51)
    return series
