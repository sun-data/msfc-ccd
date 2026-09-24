"""
Measure the noise of each tap of a camera.

:func:`readout` needs a sequence of darks,
and :func:`photon_transfer` a sequence of images gathered with the same
illumination and exposure length.
"""

import dataclasses
import numpy as np
import named_arrays as na
from ._images.abc import AbstractTapData
from ._measurements import TapDataT, _message_taps, _variance_difference

__all__ = [
    "readout",
    "photon_transfer",
]


def readout(
    images: TapDataT,
    axis: str,
    threshold: float = 5,
) -> TapDataT:
    r"""
    Estimate the readout noise of each tap from a sequence of dark images.

    The difference between each image and the previous image along `axis`
    is computed, which removes the bias, the dark current and the fixed
    pattern noise, and leaves only the readout noise of the two images.
    The readout noise is then the standard deviation of the active pixels
    in each difference image, divided by :math:`\sqrt{2}`,
    after rejecting pixels affected by cosmic rays or other spikes.

    The result has one fewer element along `axis` than the input,
    since it is defined for each pair of adjacent images.
    Take the mean along `axis` to estimate the readout noise of the camera.

    Parameters
    ----------
    images
        A sequence of dark images from each tap, such as the
        :attr:`~msfc_ccd.abc.AbstractSensorData.taps` of a sequence of images.
    axis
        The logical axis along which the images are a sequence of darks.
    threshold
        Pixels in the difference image further than this many standard
        deviations from the median are rejected.
        The standard deviation used for the rejection is estimated from the
        median absolute deviation, which is not affected by the spikes.

    Returns
    -------
    A copy of `images`, of the same type, whose outputs are the readout
    noise of each tap for each pair of adjacent images.

    Raises
    ------
    TypeError
        If `images` is not the images from each tap.
    """
    if not isinstance(images, AbstractTapData):
        raise TypeError(_message_taps(images))

    outputs = images.active.outputs
    outputs_1 = outputs[{axis: slice(1, None)}]
    outputs_0 = outputs[{axis: slice(None, -1)}]

    variance = _variance_difference(
        difference=outputs_1 - outputs_0,
        axis=images.axis_xy,
        threshold=threshold,
    )

    return dataclasses.replace(
        images,
        inputs=images.inputs[{axis: slice(1, None)}],
        outputs=np.sqrt(variance),
    )


def photon_transfer(
    images: AbstractTapData,
    axis: str,
    threshold: float = 5,
) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:
    r"""
    Compute the photon transfer curve of each tap from a sequence of flats.

    The images must be a sequence of uniformly illuminated images, or
    flats, gathered using the same illumination and exposure length.
    For each pair of adjacent images along `axis`,
    the signal is the mean of the two images and the variance is half the
    variance of their difference,
    computed over the active pixels outside the masked rows,
    :meth:`~msfc_ccd.abc.AbstractTapData.where_masked`.

    Differencing the two images removes everything the two images have in
    common, including the pattern of the illumination and the variation in
    response from pixel to pixel,
    and leaves only the shot noise and the readout noise.
    The shot noise has a variance, in electrons, equal to the signal,
    so the variance in data numbers is

    .. math::

        \sigma^2 = \frac{S}{g} + \sigma_r^2,

    where :math:`S` is the signal, :math:`g` is the gain,
    and :math:`\sigma_r` is the readout noise.
    Gathering pairs at several levels of illumination traces out the
    photon transfer curve, and :func:`msfc_ccd.gain.photon_transfer` uses it
    to measure the gain.

    The result has one fewer element along `axis` than the input,
    since it is defined for each pair of adjacent images.

    Parameters
    ----------
    images
        A sequence of flats from each tap, such as the
        :attr:`~msfc_ccd.abc.AbstractSensorData.taps` of a sequence of images.
    axis
        The logical axis along which the images are a sequence of flats.
    threshold
        Pixels in the difference image further than this many standard
        deviations from the median are rejected, as in :func:`readout`,
        to remove cosmic rays.

    Raises
    ------
    TypeError
        If `images` is not the images from each tap.

    Examples
    --------
    Compute the signal and the variance of a pair of images of a diffuse
    LED source.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na
        import msfc_ccd

        # Load two consecutive images with the same illumination
        path = na.ScalarArray(
            ndarray=np.array([
                msfc_ccd.samples.path_led_esis1,
                msfc_ccd.samples.path_led_esis1_next,
            ]),
            axes="time",
        )
        images = msfc_ccd.fits.open(path)

        # Compute the signal and the variance of each tap
        ptc = msfc_ccd.noise.photon_transfer(images.taps, "time")

        ptc.inputs.ndarray

    .. jupyter-execute::

        ptc.outputs.ndarray
    """
    if not isinstance(images, AbstractTapData):
        raise TypeError(_message_taps(images))

    num_masked = images.camera.sensor.num_masked
    outputs = images.unbiased.active.outputs
    outputs = outputs[{images.axis_y: slice(num_masked, None)}]
    outputs_1 = outputs[{axis: slice(1, None)}]
    outputs_0 = outputs[{axis: slice(None, -1)}]

    signal = ((outputs_1 + outputs_0) / 2).mean(images.axis_xy)

    variance = _variance_difference(
        difference=outputs_1 - outputs_0,
        axis=images.axis_xy,
        threshold=threshold,
    )

    return na.FunctionArray(
        inputs=signal,
        outputs=variance,
    )
