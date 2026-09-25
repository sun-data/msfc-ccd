import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd
from . import _shared

_taps = [
    _shared.dark.taps,
]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_readout(a: msfc_ccd.abc.AbstractTapData):
    axis = "_test_readout"
    num = 5
    sigma = 3 * u.DN
    rng = np.random.default_rng(seed=42)
    outputs = a.outputs + na.ScalarArray(
        ndarray=rng.normal(size=(num,) + a.outputs.ndarray.shape) * sigma,
        axes=(axis,) + a.outputs.axes,
    )
    outputs[{axis: 2, a.axis_x: 100, a.axis_y: 100}] = 60000 * u.DN
    b = a.replace(outputs=outputs)
    result = msfc_ccd.noise.readout(b, axis)
    assert isinstance(result, msfc_ccd.TapData)
    assert a.axis_x not in result.shape
    assert a.axis_y not in result.shape
    assert result.outputs.shape[axis] == num - 1
    assert a.axis_x not in result.outputs.shape
    assert a.axis_y not in result.outputs.shape
    assert np.all(np.abs(result.outputs - sigma) < 0.05 * sigma)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_photon_transfer(a: msfc_ccd.abc.AbstractTapData):
    axis = "_test_photon_transfer"
    num = 4
    signal = 4000 * u.DN
    gain = 2.5 * u.electron / u.DN
    readout_noise = 4 * u.DN
    b = _shared.flats(a, axis, signal, gain, readout_noise, num=num)

    result = msfc_ccd.noise.photon_transfer(b, axis)

    assert isinstance(result, na.FunctionArray)
    assert result.outputs.shape[axis] == num - 1
    assert a.axis_x not in result.outputs.shape
    assert a.axis_y not in result.outputs.shape
    assert na.unit(result.inputs).is_equivalent(u.DN)
    assert na.unit(result.outputs).is_equivalent(u.DN**2)

    expected = result.inputs * u.electron / gain + np.square(readout_noise)
    assert np.all(np.abs(result.outputs / expected - 1) < 0.01)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_readout_ignores_masked(a: msfc_ccd.abc.AbstractTapData):
    """The masked rows hold charge from the frame store, not the image."""
    axis = "_test_readout_ignores_masked"
    num = 5
    sigma = 3 * u.DN
    rng = np.random.default_rng(seed=42)
    noise = na.ScalarArray(
        ndarray=rng.normal(size=(num,) + a.outputs.ndarray.shape),
        axes=(axis,) + a.outputs.axes,
    )
    noise = noise * np.where(a.where_masked(), 3 * sigma, sigma)
    result = msfc_ccd.noise.readout(a.replace(outputs=a.outputs + noise), axis)

    # Including the masked rows would raise the result by about 3 percent
    assert np.all(np.abs(result.outputs - sigma) < 0.02 * sigma)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_readout_flats(a: msfc_ccd.abc.AbstractTapData):
    """The shot noise of a pair of flats is not readout noise."""
    axis = "_test_readout_flats"
    gain = 2.5 * u.electron / u.DN
    b = _shared.flats(a, axis, 4000 * u.DN, gain, readout_noise=4 * u.DN)
    with pytest.raises(ValueError, match="signal_max"):
        msfc_ccd.noise.readout(b, axis)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_readout_single(a: msfc_ccd.abc.AbstractTapData):
    """A single image has no pair to difference."""
    with pytest.raises(ValueError, match="at least two"):
        msfc_ccd.noise.readout(a, "_test_readout_single")


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_photon_transfer_single(a: msfc_ccd.abc.AbstractTapData):
    """A single image has no pair to difference."""
    with pytest.raises(ValueError, match="at least two"):
        msfc_ccd.noise.photon_transfer(a, "_test_photon_transfer_single")


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_photon_transfer_saturated(a: msfc_ccd.abc.AbstractTapData):
    """Pixels at the top of the range of the ADC have lost part of their noise."""
    axis = "_test_photon_transfer_saturated"
    gain = 2.5 * u.electron / u.DN
    b = _shared.flats(a, axis, 4000 * u.DN, gain, readout_noise=4 * u.DN)
    ceiling = (2**a.camera.bits_adc - 1) * u.DN
    num_blank = a.camera.sensor.num_blank
    outputs = b.outputs.copy()
    index = {
        a.axis_x: slice(num_blank + 100, num_blank + 200),
        a.axis_y: slice(100, 200),
    }
    outputs[index] = ceiling
    with pytest.raises(ValueError, match="fraction_saturated"):
        msfc_ccd.noise.photon_transfer(b.replace(outputs=outputs), axis)
