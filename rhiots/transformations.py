from pathlib import Path
import numpy as np
from typing import Type

from scipy.interpolate import CubicSpline
import numpy as np
from typing import Union, List


class ManipulateData:
    """
    A class for applying various data transformation techniques to time series data.

    Parameters
    ----------
    x : Union[np.ndarray, List[List[float]]]
        The time series data to transform, expected as a 2D array-like structure
        where rows are time steps and columns are series.
    transformation : str
        The transformation to apply. Options are: "jitter", "scaling",
        "magnitude_warp", "time_warp".
    parameters : List[float]
        Transformation parameters, where each parameter corresponds to
        the effect size of each transformation type.

    Methods
    -------
    apply_transf() -> np.ndarray
        Applies the specified transformation and returns the transformed data.
    """

    def __init__(
        self,
        x: Union[np.ndarray, List[List[float]]],
        transformation: str,
        parameters: List[float],
        version: int,
    ):
        self.x = np.array(x)
        self.parameters = parameters
        self.version = version
        self.transformation = transformation
        self.orig_steps = np.arange(self.x.shape[0])

        # Map of transformation functions
        self.transformation_map = {
            "jitter": self._jitter,
            "scaling": self._scaling,
            "magnitude_warp": self._magnitude_warp,
            "time_warp": self._time_warp,
        }
        if transformation not in self.transformation_map:
            raise ValueError(f"Transformation '{transformation}' is not supported.")

    def _jitter(self) -> np.ndarray:
        """Adds random noise to the time series data, scaled by the standard deviation of each series."""
        #  scale noise relative to series standard deviation and detrend series
        x_diff = np.diff(self.x, axis=0)
        series_std = np.std(x_diff, axis=0, keepdims=True)
        jitter_amount = self.parameters[0] * series_std * self.version
        return self.x + np.random.normal(0, jitter_amount, size=self.x.shape)

    def _scaling(self) -> np.ndarray:
        """Scales the time series data by a random factor."""
        x_diff = np.diff(self.x, axis=0)
        series_std = np.std(x_diff, axis=0, keepdims=True)
        scaling_amount = self.parameters[1] * series_std * self.version
        scaling_factor = np.random.normal(1, scaling_amount, size=self.x.shape)
        return self.x * scaling_factor

    def _magnitude_warp(self, knots: int = 4) -> np.ndarray:
        """Applies random magnitude warping to the time series data using cubic splines."""
        x_diff = np.diff(self.x, axis=0)
        series_std = np.std(x_diff, axis=0, keepdims=True)
        warping_amount = self.parameters[0] * series_std * self.version
        warp_strength = np.random.normal(
            1, warping_amount, (knots + 2, self.x.shape[1])
        )
        warp_steps = np.linspace(0, self.x.shape[0] - 1, knots + 2)
        warped_data = np.empty_like(self.x)

        for col in range(self.x.shape[1]):
            spline = CubicSpline(warp_steps, warp_strength[:, col])
            warp_values = spline(self.orig_steps)
            warped_data[:, col] = self.x[:, col] * warp_values

        return warped_data

    def _time_warp(self, knots: int = 4) -> np.ndarray:
        """Applies random time warping to the time series data by stretching and compressing intervals."""
        x_diff = np.diff(self.x, axis=0)
        series_std = np.std(x_diff, axis=0, keepdims=True)
        twarping_amount = self.parameters[3] * series_std * self.version
        warp_strength = np.random.normal(
            1, twarping_amount, (knots + 2, self.x.shape[1])
        )
        warp_steps = np.linspace(0, self.x.shape[0] - 1, knots + 2)
        time_warped_data = np.empty_like(self.x)

        for col in range(self.x.shape[1]):
            time_spline = CubicSpline(warp_steps, warp_steps * warp_strength[:, col])
            warped_time_steps = time_spline(self.orig_steps)
            time_warped_data[:, col] = np.interp(
                self.orig_steps, warped_time_steps, self.x[:, col]
            )

        return time_warped_data

    def apply_transf(self) -> np.ndarray:
        """Applies the chosen transformation to the data."""
        transform_func = self.transformation_map[self.transformation]
        return transform_func()
