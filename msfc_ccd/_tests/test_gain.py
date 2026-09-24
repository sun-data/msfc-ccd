import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd
from .test_noise import _flats

_camera = msfc_ccd.Camera(
    gain=na.ScalarArray(
        ndarray=[[2.5, 2.6], [2.7, 2.8]] * u.electron / u.DN,
        axes=("tx", "ty"),
    ),
    axis_tap_x="tx",
    axis_tap_y="ty",
)

_taps = [
    msfc_ccd.fits.open(msfc_ccd.samples.path_dark_esis1, _camera).taps,
]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_fe55(a: msfc_ccd.abc.AbstractTapData):
    gain = 2.5 * u.electron / u.DN
    source = msfc_ccd.Fe55()
    rng = np.random.default_rng(seed=42)

    num_blank = a.camera.sensor.num_blank
    num_x, num_y, step = 30, 15, 33

    # Fe 55 events on a grid, spaced far enough apart to stay isolated
    probability_beta = source.probability_k_beta / (
        source.probability_k_alpha + source.probability_k_beta
    )
    where_beta = rng.random(size=(num_x, num_y)) < probability_beta
    charge = np.where(where_beta, source.charge_k_beta, source.charge_k_alpha)
    charge = charge / gain + rng.normal(scale=14, size=charge.shape) * u.DN
    charge = na.ScalarArray(charge, axes=(a.axis_x, a.axis_y))

    index = {
        a.axis_x: slice(num_blank + 8, num_blank + 8 + num_x * step, step),
        a.axis_y: slice(8, 8 + num_y * step, step),
    }
    outputs = a.outputs.copy()
    outputs[index] = outputs[index] + charge

    result = msfc_ccd.gain.fe55(a.replace(outputs=outputs))

    assert isinstance(result, msfc_ccd.TapData)
    assert a.axis_x not in result.outputs.shape
    assert a.axis_y not in result.outputs.shape
    assert na.unit(result.outputs).is_equivalent(u.electron / u.DN)
    assert np.all(np.abs(result.outputs - gain) < 0.02 * gain)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_fe55_without_events(a: msfc_ccd.abc.AbstractTapData):
    """A tap with too few events has no gain to report."""
    result = msfc_ccd.gain.fe55(a)
    assert np.all(np.isnan(result.outputs))


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_photon_transfer(a: msfc_ccd.abc.AbstractTapData):
    axis = "_test_photon_transfer"
    signal = na.ScalarArray([2000, 6000] * u.DN, axes="_test_level")
    gain = 2.5 * u.electron / u.DN
    b = _flats(a, axis, signal, gain, readout_noise=4 * u.DN)

    result = msfc_ccd.gain.photon_transfer(b, axis)

    assert isinstance(result, msfc_ccd.TapData)
    assert set(result.outputs.shape) == {a.axis_tap_x, a.axis_tap_y}
    assert na.unit(result.outputs).is_equivalent(u.electron / u.DN)
    assert np.all(np.abs(result.outputs - gain) < 0.01 * gain)
