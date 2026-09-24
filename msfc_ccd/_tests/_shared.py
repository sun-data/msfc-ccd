"""Cameras, images and synthetic data shared by the test modules."""

import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd

camera = msfc_ccd.Camera(
    gain=na.ScalarArray(
        ndarray=[[2.5, 2.6], [2.7, 2.8]] * u.electron / u.DN,
        axes=("tx", "ty"),
    ),
    axis_tap_x="tx",
    axis_tap_y="ty",
)
"""A camera with a known gain for each tap and short names for the tap axes."""

dark = msfc_ccd.fits.open(msfc_ccd.samples.path_dark_esis1, camera)
"""A dark image of the whole sensor, gathered with :data:`camera`."""


def flats(
    a: msfc_ccd.abc.AbstractTapData,
    axis: str,
    signal: na.AbstractScalar,
    gain: u.Quantity,
    readout_noise: u.Quantity,
    num: int = 4,
) -> msfc_ccd.abc.AbstractTapData:
    """Build a sequence of flats with a known gain."""
    rng = np.random.default_rng(seed=42)

    # A pattern of illumination which varies across each tap, which the
    # difference of two flats must cancel.
    i = a.outputs.indices[a.axis_x]
    pattern = 0.5 + i / a.shape[a.axis_x]

    # The light reaches only the light-sensitive pixels
    where_active = ~(a.where_blank() | a.where_overscan())
    mean = signal * pattern * where_active
    mean = na.broadcast_to(mean, mean.shape | a.outputs.shape)

    shape = {axis: num} | mean.shape
    electrons = rng.poisson(
        lam=(gain * mean).to(u.electron).ndarray_aligned(mean.shape).value,
        size=tuple(shape.values()),
    )
    electrons = na.ScalarArray(electrons * u.electron, axes=tuple(shape))
    readout = na.ScalarArray(
        ndarray=rng.normal(size=tuple(shape.values())) * readout_noise,
        axes=tuple(shape),
    )

    return a.replace(outputs=a.outputs + electrons / gain + readout)
