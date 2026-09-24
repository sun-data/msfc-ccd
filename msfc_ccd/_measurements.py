"""Helpers shared by the modules that measure the properties of a camera."""

from typing import TypeVar
import numpy as np
import named_arrays as na
from ._images.abc import AbstractCameraData, AbstractTapData

__all__ = [
    "CameraDataT",
    "TapDataT",
]

CameraDataT = TypeVar("CameraDataT", bound=AbstractCameraData)
"""Images from either the whole sensor or each tap."""

TapDataT = TypeVar("TapDataT", bound=AbstractTapData)
"""Images from each tap."""


def _message_taps(images: object) -> str:
    """Explain that a measurement needs the images from each tap."""
    return (
        "`images` must be the images from each tap, "
        "an instance of `msfc_ccd.abc.AbstractTapData`, "
        f"got `{type(images).__name__}`. "
        "Pass `images.taps` instead."
    )


def _variance_difference(
    difference: na.AbstractScalarArray,
    axis: tuple[str, str],
    threshold: float,
) -> na.AbstractScalarArray:
    """Half the variance of a difference image, rejecting spikes."""
    median = np.median(difference, axis=axis)
    deviation = np.abs(difference - median)
    mad = np.median(deviation, axis=axis)

    # The ratio of the standard deviation to the median absolute deviation
    # for a normal distribution
    factor = 1.482602218505602
    where = deviation < threshold * factor * mad

    return np.square(difference.std(axis=axis, where=where)) / 2
