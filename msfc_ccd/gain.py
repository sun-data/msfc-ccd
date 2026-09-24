r"""
Measure the gain of each tap of a camera.

:func:`fe55` needs one or more exposures of an :math:`^{55}\text{Fe}` source,
and :func:`photon_transfer` a sequence of flats.
"""

import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
from ._fe55 import Fe55, _fit_gain
from ._images.abc import AbstractTapData
from ._measurements import TapDataT, _message_taps, _variance_difference
from . import noise

__all__ = [
    "fe55",
    "photon_transfer",
]


def fe55(
    images: TapDataT,
    threshold: float = 5,
    gain_min: u.Quantity = 2 * u.electron / u.DN,
    gain_max: u.Quantity = 5 * u.electron / u.DN,
    source: None | Fe55 = None,
) -> TapDataT:
    r"""
    Measure the gain of each tap from one or more Fe 55 exposures.

    The isolated events found by :meth:`~msfc_ccd.abc.AbstractTapData.hits`
    are pooled over every axis except the two tap axes, and the gain of each
    tap is fit to the charges of those events.

    The model is a pair of Gaussians, one for each
    :math:`^{55}\text{Fe}` line, whose separation and relative height are
    fixed by the line energies and the emission probabilities, on a flat
    background of events which lost part of their charge to a neighboring
    pixel or to the surface.
    The only free parameters are the gain, the width of the lines and the
    size of that background.
    The fit is an unbinned maximum likelihood, so there is no bin width to
    choose, which matters because a single image yields only a few tens of
    events per tap.

    Parameters
    ----------
    images
        One or more Fe 55 exposures from each tap, such as the
        :attr:`~msfc_ccd.abc.AbstractSensorData.taps` of an image.
    threshold
        Passed to :meth:`~msfc_ccd.abc.AbstractTapData.hits`.
    gain_min
        The smallest gain to consider.
        This and `gain_max` bracket where the K-:math:`\alpha` peak can
        lie, which is what lets the peak be found without a starting
        guess.
    gain_max
        The largest gain to consider.
    source
        The properties of the :math:`^{55}\text{Fe}` source.
        If :obj:`None`, :class:`msfc_ccd.Fe55` is used.

    Returns
    -------
    A copy of `images`, of the same type, whose outputs are the gain of each
    tap, or :obj:`numpy.nan` for a tap with too few events to fit.

    Raises
    ------
    TypeError
        If `images` is not the images from each tap.

    Examples
    --------
    Measure the gain of the ESIS channel 3 camera.

    .. jupyter-execute::

        import msfc_ccd

        image = msfc_ccd.fits.open(msfc_ccd.samples.path_fe55_esis3)

        msfc_ccd.gain.fe55(image.taps).outputs.ndarray
    """
    if not isinstance(images, AbstractTapData):
        raise TypeError(_message_taps(images))

    if source is None:
        source = Fe55()

    hits = images.hits(threshold)

    unit = u.electron / u.DN
    gain_min = gain_min.to_value(unit)
    gain_max = gain_max.to_value(unit)

    charge = hits.outputs
    axes = tuple(charge.shape)
    axes_tap = (images.axis_tap_y, images.axis_tap_x)
    axes_tap = tuple(a for a in axes_tap if a in axes)
    axes_pooled = tuple(a for a in axes if a not in axes_tap)

    shape_tap = tuple(charge.shape[a] for a in axes_tap)
    ndarray = charge.ndarray_aligned(axes_tap + axes_pooled)
    ndarray = ndarray.to_value(u.DN).reshape(shape_tap + (-1,))

    result = np.empty(shape_tap)
    for index in np.ndindex(*shape_tap):
        result[index] = _fit_gain(
            charge=ndarray[index],
            gain_min=gain_min,
            gain_max=gain_max,
            fe55=source,
        )

    return dataclasses.replace(
        images,
        inputs=images.inputs[{a: 0 for a in axes_pooled}],
        outputs=na.ScalarArray(result * unit, axes=axes_tap),
    )


def photon_transfer(
    images: TapDataT,
    axis: str,
    threshold: float = 5,
) -> TapDataT:
    r"""
    Measure the gain of each tap from a sequence of flat images.

    The gain is the ratio of the signal to the shot noise variance
    computed by :func:`msfc_ccd.noise.photon_transfer`,

    .. math::

        g = \frac{S}{\sigma^2 - \sigma_r^2},

    pooled over every axis except the two tap axes,
    so pairs of flats at several levels of illumination can be combined.
    The readout noise, :math:`\sigma_r`, is measured from the blank columns
    of the same pairs of images, where there is no shot noise,
    so it follows any change in the noise of the camera during the test.

    This function is independent of :func:`fe55`, which measures the gain
    from the charge released by :math:`^{55}\text{Fe}` X-rays.
    On the tests of 2017-07-12 the two agree to within 3 percent,
    with the flats giving the lower gain on every tap.

    The flats should be well below full well.
    Above about 5,000 DN, the variance of the ESIS cameras grows more
    slowly than the signal, so the apparent gain rises,
    by 1 to 2 percent at 15,000 DN and about 5 percent at 30,000 DN.
    The sample images in the example below are near 15,000 DN.

    Parameters
    ----------
    images
        A sequence of flats from each tap, such as the
        :attr:`~msfc_ccd.abc.AbstractSensorData.taps` of a sequence of images.
    axis
        The logical axis along which the images are a sequence of flats.
    threshold
        Pixels in the difference images further than this many standard
        deviations from the median are rejected, to remove cosmic rays.

    Returns
    -------
    A copy of `images`, of the same type, whose outputs are the gain of each
    tap.

    Raises
    ------
    TypeError
        If `images` is not the images from each tap.

    Examples
    --------
    Measure the gain of the ESIS channel 1 camera from a pair of images of
    a diffuse LED source.

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

        # Measure the gain of each tap
        msfc_ccd.gain.photon_transfer(images.taps, "time").outputs.ndarray
    """
    if not isinstance(images, AbstractTapData):
        raise TypeError(_message_taps(images))

    ptc = noise.photon_transfer(images, axis, threshold)

    outputs = images.outputs
    outputs = outputs[{images.axis_x: images.where_blank(num=25)}]
    outputs_1 = outputs[{axis: slice(1, None)}]
    outputs_0 = outputs[{axis: slice(None, -1)}]
    variance_readout = _variance_difference(
        difference=outputs_1 - outputs_0,
        axis=images.axis_xy,
        threshold=threshold,
    )

    axes_tap = (images.axis_tap_x, images.axis_tap_y)
    axes_pooled = tuple(a for a in ptc.outputs.shape if a not in axes_tap)

    signal = ptc.inputs.sum(axes_pooled)
    variance = (ptc.outputs - variance_readout).sum(axes_pooled)

    return dataclasses.replace(
        images,
        inputs=images.inputs[{a: 0 for a in axes_pooled}],
        outputs=(u.electron * signal / variance).to(u.electron / u.DN),
    )
