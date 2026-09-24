"""Helpers shared by the modules that measure the properties of a camera."""

from typing import Callable, TypeVar
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


def _axes_pixel(images: AbstractCameraData) -> dict[str, int]:
    """Index the pixel axes of the header of a result which has none."""
    return {images.axis_x: 0, images.axis_y: 0}


def _fit_taps(
    images: AbstractTapData,
    fit: Callable[..., tuple[float, ...]],
    num: int,
    **arrays: na.AbstractScalar,
) -> tuple[list[na.ScalarArray], tuple[str, ...]]:
    """
    Fit each tap separately, pooling every other axis.

    `arrays` are broadcast against each other, and `fit` is called once for
    each tap with the values of each array in that tap as a flat
    :class:`numpy.ndarray`.
    It must return `num` numbers, which are gathered into arrays over the tap
    axes.
    The axes which were pooled are returned as well.
    """
    shape = na.shape_broadcasted(*arrays.values())
    axes_tap = (images.axis_tap_y, images.axis_tap_x)
    shape_tap = {a: shape[a] for a in axes_tap if a in shape}
    axes_pooled = tuple(a for a in shape if a not in shape_tap)

    results = [na.ScalarArray.empty(shape_tap) for _ in range(num)]
    for index in na.ndindex(shape_tap):
        values = {
            name: na.broadcast_to(array, shape)[index].ndarray.reshape(-1)
            for name, array in arrays.items()
        }
        for result, value in zip(results, fit(**values)):
            result[index] = value

    return results, axes_pooled


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
