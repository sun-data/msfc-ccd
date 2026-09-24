import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd

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


def _flats(
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
    b = _flats(a, axis, signal, gain, readout_noise, num=num)

    result = msfc_ccd.noise.photon_transfer(b, axis)

    assert isinstance(result, na.FunctionArray)
    assert result.outputs.shape[axis] == num - 1
    assert a.axis_x not in result.outputs.shape
    assert a.axis_y not in result.outputs.shape
    assert na.unit(result.inputs).is_equivalent(u.DN)
    assert na.unit(result.outputs).is_equivalent(u.DN**2)

    expected = result.inputs * u.electron / gain + np.square(readout_noise)
    assert np.all(np.abs(result.outputs / expected - 1) < 0.01)
