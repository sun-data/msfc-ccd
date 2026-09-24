"""
Measure the dark signal of a camera.

:func:`master` and :func:`current` both need a sequence of darks,
and :func:`current` needs them at a range of exposure lengths.
"""

import dataclasses
import numpy as np
import named_arrays as na
from ._images.abc import AbstractTapData
from ._measurements import CameraDataT, TapDataT, _message_taps

__all__ = [
    "master",
    "current",
]


def master(
    images: CameraDataT,
    axis: str,
    proportion: float = 0.25,
) -> CameraDataT:
    """
    Estimate the master dark image of each tap from a sequence of dark images.

    The master dark is the trimmed mean of the sequence along `axis`,
    computed independently for each pixel.
    Trimming the largest and smallest values rejects cosmic rays and
    other spikes which affect a single image, while hot pixels and the
    fixed pattern of the sensor, which are present in every image,
    are preserved.

    The images should have had their bias removed before calling this
    function, so that the result can be subtracted from other images which
    have had their bias removed in the same way.

    Parameters
    ----------
    images
        A sequence of dark images, either for the whole sensor or for each
        tap.
    axis
        The logical axis along which the images are a sequence of darks.
    proportion
        The fraction of the largest and smallest values to remove from each
        pixel before taking the mean.
        The number of images removed is rounded down,
        so with fewer than ``1 / proportion`` images
        nothing is removed.

    Returns
    -------
    A copy of `images`, of the same type, whose outputs are the master dark,
    without `axis`.
    """
    outputs = na.mean_trimmed(
        a=images.outputs,
        q=proportion,
        axis=axis,
    )
    return dataclasses.replace(
        images,
        inputs=images.inputs[{axis: 0}],
        outputs=outputs,
    )


def current(
    images: TapDataT,
    axis: str,
    proportion: float = 0.01,
) -> TapDataT:
    """
    Estimate the dark current rate of each tap.

    The images must be a sequence of darks gathered using a range of
    exposure lengths.
    The signal in each image is the trimmed mean of the bias-subtracted
    active pixels, and the rate is the slope of a linear fit of that signal
    against the measured exposure time of each image,
    :attr:`msfc_ccd.ImageHeader.timedelta`.

    The masked rows at the start of each tap,
    :meth:`~msfc_ccd.abc.AbstractTapData.where_masked`,
    are left out of the mean.
    They hold charge from the frame store rather than from the image area,
    and their dark current rate is about 80 times higher,
    so although they are less than two percent of the active pixels,
    including them would roughly double the result.

    The intercept of the fit is discarded.
    The blank columns used to compute the bias are offset from the dark
    level of the active pixels by about 1 DN,
    and the fixed pattern of the sensor is not removed,
    so the intercept is not the dark current at zero exposure.
    Only the slope is independent of both.

    The dark current accumulated during a single image is much smaller
    than the readout noise, about 0.05 DN in a two-second image,
    so it is only measurable after averaging over all the active pixels,
    and a long exposure is needed to separate it from the bias.

    Parameters
    ----------
    images
        A sequence of dark images from each tap, gathered using a range of
        exposure lengths, such as the
        :attr:`~msfc_ccd.abc.AbstractSensorData.taps` of a sequence of images.
    axis
        The logical axis along which the images are a sequence of darks.
    proportion
        The fraction of the brightest and darkest active pixels to remove
        from each image before taking the mean,
        which protects the mean from cosmic rays.

        On the dark tests of 2017-07-12, cosmic rays are too rare to
        matter, and the result is the same to within one percent for any
        proportion from 0 to 0.05.
        Trimming much harder removes the hot pixels too,
        which are part of the dark current.

    Returns
    -------
    A copy of `images`, of the same type, whose outputs are the dark current
    rate of each tap.

    Raises
    ------
    TypeError
        If `images` is not the images from each tap.

    Examples
    --------
    Estimate the dark current rate of the ESIS channel 1 camera using
    a pair of dark images with different exposure lengths.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na
        import msfc_ccd

        # Load a two-second dark image and a twelve-second dark image
        path = na.ScalarArray(
            ndarray=np.array([
                msfc_ccd.samples.path_dark_2s_esis1,
                msfc_ccd.samples.path_dark_12s_esis1,
            ]),
            axes="time",
        )
        images = msfc_ccd.fits.open(path)

        # Estimate the dark current rate of each tap
        msfc_ccd.dark.current(images.taps, "time").outputs.to("DN / s").ndarray

    Two images give only a rough estimate, since the uncertainty in the
    bias of each image is about a fifth of the signal accumulated between
    them.
    Averaging many images at each of five exposure lengths gives
    0.020 to 0.025 DN / s across the four taps of this camera,
    within six percent of the independent analysis of the same test by
    MSFC.
    """
    if not isinstance(images, AbstractTapData):
        raise TypeError(_message_taps(images))

    num_masked = images.camera.sensor.num_masked
    signal = images.unbiased.active.outputs
    signal = signal[{images.axis_y: slice(num_masked, None)}]
    signal = na.mean_trimmed(
        a=signal,
        q=proportion,
        axis=images.axis_xy,
    )
    timedelta = images.inputs.timedelta

    signal = signal - signal.mean(axis)
    timedelta = timedelta - timedelta.mean(axis)

    rate = (timedelta * signal).sum(axis) / np.square(timedelta).sum(axis)

    return dataclasses.replace(
        images,
        inputs=images.inputs[{axis: 0}],
        outputs=rate,
    )
