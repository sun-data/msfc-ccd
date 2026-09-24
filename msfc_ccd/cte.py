r"""
Measure the charge transfer efficiency of each tap of a camera.

:func:`fe55` needs several hundred exposures of an :math:`^{55}\text{Fe}`
source, and :func:`eper` a flat.
"""

import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
from ._fe55 import Fe55, _fit_cte
from ._images.abc import AbstractTapData

__all__ = [
    "fe55",
    "eper",
]


def fe55(
    images: AbstractTapData,
    threshold: float = 5,
    threshold_split: float = 3,
    gain_min: u.Quantity = 2 * u.electron / u.DN,
    gain_max: u.Quantity = 5 * u.electron / u.DN,
    source: None | Fe55 = None,
) -> AbstractTapData:
    r"""
    Estimate the charge transfer efficiency of each tap from Fe 55 images.

    Every :math:`^{55}\text{Fe}` K-:math:`\alpha` X-ray releases the same
    charge, but an event far from the amplifier is transferred more times
    before it is read out, and loses a little of its charge to each
    imperfect transfer.
    The single-pixel events found by
    :meth:`~msfc_ccd.abc.AbstractTapData.hits` are pooled over every axis
    except the two tap axes, and fit with the model used by
    :func:`msfc_ccd.gain.fe55`, except that the center of each line falls
    linearly with the number of serial and parallel transfers of the event,

    .. math::

        q = q_0 \left(1 - \text{CTI}_x n_x - \text{CTI}_y n_y \right),

    where :math:`n_x` counts the blank columns as well as the active
    columns between the event and the amplifier,
    and :math:`n_y` counts the rows.

    Only the center pixel of each single-pixel event is used,
    which is the standard form of the method.
    MSFC summed the :math:`3 \times 3` region around each event, which
    recovers the charge left in the neighboring pixel and so hides the
    loss; their fits gave a charge transfer efficiency above 100 percent.

    The result is a vector for each tap,
    whose :math:`x` component is the serial charge transfer efficiency and
    whose :math:`y` component is the parallel charge transfer efficiency.

    The loss is tiny, about a data number across the whole tap,
    so this function needs thousands of events per tap,
    or several hundred images from the ESIS cameras.

    Subtract a master dark from the images first, using
    :func:`msfc_ccd.dark.master`.
    The dark level of the ESIS cameras rises by 0.3 to 0.4 DN from the
    first row read out to the last, which the median removed by
    :meth:`~msfc_ccd.abc.AbstractTapData.hits` cannot follow,
    and which would otherwise look like a negative parallel inefficiency of
    about :math:`10^{-6}`.

    The fit also averages over any columns with charge traps.
    On a sensor with many of them, the traps dominate the result,
    and the serial component should be taken from :func:`eper` instead.

    Parameters
    ----------
    images
        Several hundred Fe 55 exposures.
    threshold
        How many readout noises above the dark level a pixel must be to
        start an event.
    threshold_split
        How many readout noises above the dark level each of the eight
        neighbors of an event must stay below for it to count as a
        single-pixel event.
    gain_min
        The smallest gain allowed, which sets where the K-:math:`\alpha`
        peak can be.
    gain_max
        The largest gain allowed.
    source
        The properties of the :math:`^{55}\text{Fe}` source.
        If :obj:`None`, the default :class:`msfc_ccd.Fe55` is used.

    Examples
    --------
    Estimate the charge transfer efficiency of the ESIS channel 3 camera
    from four Fe 55 exposures.
    Four images are far too few for a precise result,
    so this only shows how the function is called.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na
        import msfc_ccd

        path = na.ScalarArray(
            ndarray=np.array(msfc_ccd.samples.paths_fe55_esis3),
            axes="time",
        )
        images = msfc_ccd.fits.open(path)

        cte = msfc_ccd.cte.fe55(images.taps).outputs

        # The serial charge transfer efficiency
        cte.x.ndarray

    .. jupyter-execute::

        # The parallel charge transfer efficiency
        cte.y.ndarray
    """
    if source is None:
        source = Fe55()

    hits = images.hits(threshold, threshold_split)

    unit = u.electron / u.DN
    gain_min = gain_min.to_value(unit)
    gain_max = gain_max.to_value(unit)

    # The masked rows are not part of the image
    charge = hits.outputs
    index_x = charge.indices[images.axis_x]
    index_y = charge.indices[images.axis_y]
    where_image = index_y >= images.camera.sensor.num_masked
    charge = charge * np.where(where_image, 1, np.nan)

    # The number of transfers before each pixel reaches the amplifier
    transfers_x = index_x + images.camera.sensor.num_blank + 1
    transfers_y = index_y + 1

    shape = charge.shape
    axes_tap = (images.axis_tap_y, images.axis_tap_x)
    axes_tap = tuple(a for a in axes_tap if a in shape)
    axes_pooled = tuple(a for a in shape if a not in axes_tap)
    shape_tap = tuple(shape[a] for a in axes_tap)

    def flatten(a: na.AbstractScalar) -> np.ndarray:
        a = na.broadcast_to(a, shape)
        a = a.ndarray_aligned(axes_tap + axes_pooled)
        return np.asarray(a).reshape(shape_tap + (-1,))

    charge = flatten(charge.to(u.DN).value)
    transfers_x = flatten(transfers_x)
    transfers_y = flatten(transfers_y)

    cti_x = np.empty(shape_tap)
    cti_y = np.empty(shape_tap)
    for index in np.ndindex(*shape_tap):
        cti_x[index], cti_y[index] = _fit_cte(
            charge=charge[index],
            transfers_x=transfers_x[index],
            transfers_y=transfers_y[index],
            gain_min=gain_min,
            gain_max=gain_max,
            fe55=source,
        )

    return dataclasses.replace(
        images,
        inputs=images.inputs[{a: 0 for a in axes_pooled}],
        outputs=na.Cartesian2dVectorArray(
            x=na.ScalarArray((1 - cti_x) * 100 * u.percent, axes=axes_tap),
            y=na.ScalarArray((1 - cti_y) * 100 * u.percent, axes=axes_tap),
        ),
    )


def eper(
    images: AbstractTapData,
    num_active: int = 10,
) -> AbstractTapData:
    r"""
    Estimate the serial charge transfer efficiency of each tap from a flat.

    This is the extended pixel edge response (EPER) method.
    Each serial transfer leaves a small fraction of the charge behind,
    which is released into the pixels read out afterwards.
    In a flat, the charge left behind by the last active pixel of each row
    spills into the overscan columns, so the charge transfer inefficiency
    is

    .. math::

        \text{CTI} = \frac{Q_o}{S N},

    where :math:`Q_o` is the total charge in the overscan columns of a
    row, :math:`S` is the signal in the last active pixels of that row,
    and :math:`N` is the number of serial transfers of the last active
    pixel.
    The result is the charge transfer efficiency,
    :math:`\text{CTE} = 1 - \text{CTI}`, for each image.

    :math:`N` counts the blank columns as well as the active columns,
    since the blank columns are elements of the serial register between the
    active pixels and the amplifier.
    MSFC counted only the active columns, which makes their inefficiency
    about 5 percent larger for the same data.

    The masked rows, :meth:`~msfc_ccd.abc.AbstractTapData.where_masked`,
    are ignored.
    Only the serial transfers are measured,
    since the sensor is read out without any overscan rows.

    The charge left behind is not a fixed fraction of the signal.
    On the tests of 2017-07-12, the fraction is nearly constant on some
    taps and falls with signal on others, and on every tap it rises below
    about a thousand data numbers,
    so the result describes the camera at the signal level of the flat.
    The overscan columns of a dark image sit within about 0.4 DN of the
    bias, which matters only for faint flats;
    subtract a dark image first to remove it.

    Parameters
    ----------
    images
        One or more flats.
    num_active
        The number of active columns at the end of each row used to
        measure the signal.

    Examples
    --------
    Estimate the serial charge transfer efficiency of the ESIS channel 1
    camera from an image of a diffuse LED source.

    .. jupyter-execute::

        import msfc_ccd

        image = msfc_ccd.fits.open(msfc_ccd.samples.path_led_esis1)

        msfc_ccd.cte.eper(image.taps).outputs.ndarray
    """
    sensor = images.camera.sensor

    outputs = images.unbiased.outputs
    outputs = outputs[{images.axis_y: slice(sensor.num_masked, None)}]

    num_transfers = images.num_x - sensor.num_overscan
    slice_active = slice(num_transfers - num_active, num_transfers)
    slice_overscan = slice(num_transfers, None)

    signal = outputs[{images.axis_x: slice_active}].mean(images.axis_xy)
    deferred = outputs[{images.axis_x: slice_overscan}].sum(images.axis_x)
    deferred = deferred.mean(images.axis_y)

    cti = deferred / (signal * num_transfers)

    return dataclasses.replace(
        images,
        outputs=(1 - cti).to(u.percent),
    )
