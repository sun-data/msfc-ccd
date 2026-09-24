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
