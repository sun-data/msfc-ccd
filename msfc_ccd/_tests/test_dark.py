import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd
from . import _shared

_images = [
    _shared.dark,
    msfc_ccd.fits.open(
        path=na.ScalarArray(
            ndarray=np.array(
                [
                    msfc_ccd.samples.path_dark_esis1,
                    msfc_ccd.samples.path_dark_esis3,
                ]
            ),
            axes="channel",
        ),
        camera=_shared.camera,
    ),
    msfc_ccd.fits.open(
        path=na.ScalarArray(
            ndarray=np.array(
                [
                    msfc_ccd.samples.path_dark_esis1,
                    msfc_ccd.samples.path_fe55_esis1,
                ]
            ),
            axes="time",
        ),
        camera=_shared.camera,
    ),
]

_taps = [
    _images[0].taps,
]


def _darks(
    a: msfc_ccd.abc.AbstractTapData,
    axis: str,
    rate: u.Quantity,
) -> msfc_ccd.abc.AbstractTapData:
    """Build a sequence of darks with a known dark current rate."""
    timedelta = na.ScalarArray(
        ndarray=[2, 3, 4, 7, 12] * u.s,
        axes=axis,
    )

    # The dark current accumulates only in the light-sensitive pixels,
    # so the blank columns used for the bias are untouched.
    where_active = ~(a.where_blank() | a.where_overscan())
    outputs = a.outputs + rate * timedelta * where_active

    # Cosmic rays, which arrive in proportion to the exposure time and so
    # would otherwise be counted as dark current.
    # They land only in the active pixels. The blank columns are extra
    # clock cycles of an empty serial register rather than real pixels,
    # so nothing can accumulate there.
    rng = np.random.default_rng(seed=42)
    shape = outputs.shape
    num_blank = a.camera.sensor.num_blank
    num_x = shape[a.axis_x] - num_blank - a.camera.sensor.num_overscan
    num = np.round(10 * timedelta / timedelta[{axis: 0}]).astype(int)
    for i in range(shape[axis]):
        for _ in range(num[{axis: i}].ndarray.item()):
            index = {
                axis: i,
                a.axis_x: num_blank + rng.integers(num_x).item(),
                a.axis_y: rng.integers(shape[a.axis_y]).item(),
            }
            outputs[index] = 60000 * u.DN

    return a.replace(
        inputs=a.inputs.replace(
            timedelta=timedelta,
            timedelta_requested=timedelta,
        ),
        outputs=outputs,
    )


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_images + _taps,
)
def test_master(a: msfc_ccd.abc.AbstractCameraData):
    axis = "_test_master"
    num = 8
    sigma = 3 * u.DN
    rng = np.random.default_rng(seed=42)
    noise = na.ScalarArray(
        ndarray=rng.normal(size=(num,) + a.outputs.ndarray.shape) * sigma,
        axes=(axis,) + a.outputs.axes,
    )
    outputs = a.outputs + noise
    outputs[{axis: 2, a.axis_x: 100, a.axis_y: 100}] = 60000 * u.DN
    b = a.replace(outputs=outputs)
    result = msfc_ccd.dark.master(b, axis)
    assert isinstance(result, type(a))
    assert axis not in result.outputs.shape
    assert result.outputs.shape[a.axis_x] == a.outputs.shape[a.axis_x]
    assert result.outputs.shape[a.axis_y] == a.outputs.shape[a.axis_y]
    residual = result.outputs - a.outputs
    assert np.all(np.abs(residual[{a.axis_x: 100, a.axis_y: 100}]) < 10 * sigma)
    assert np.all(np.abs(residual.mean()) < 0.1 * sigma)
    assert np.all(residual.std() < sigma / np.sqrt(num / 2))


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_current(a: msfc_ccd.abc.AbstractTapData):
    axis = "_test_current"
    rate = 0.05 * u.DN / u.s
    result = msfc_ccd.dark.current(_darks(a, axis, rate), axis)

    assert isinstance(result, msfc_ccd.TapData)
    assert a.axis_x not in result.shape
    assert a.axis_y not in result.shape
    assert axis not in result.outputs.shape
    assert a.axis_x not in result.outputs.shape
    assert a.axis_y not in result.outputs.shape
    assert np.all(np.abs(result.outputs - rate) < 0.01 * rate)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_current_ignores_masked(a: msfc_ccd.abc.AbstractTapData):
    """The masked rows accumulate dark current much faster than the image."""
    axis = "_test_current_ignores_masked"
    rate = 0.05 * u.DN / u.s
    b = _darks(a, axis, rate)

    # Only the light-sensitive part of the masked rows, since the blank
    # columns are clocked out of the serial register.
    where_active = ~(b.where_blank() | b.where_overscan())
    where = b.where_masked() & where_active
    b = b.replace(outputs=b.outputs + 100 * rate * b.inputs.timedelta * where)
    result = msfc_ccd.dark.current(b, axis)

    assert np.all(np.abs(result.outputs - rate) < 0.01 * rate)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_current_linear(a: msfc_ccd.abc.AbstractTapData):
    """The rate must not depend on which exposures were used."""
    axis = "_test_current_linear"
    rate = 0.05 * u.DN / u.s
    b = _darks(a, axis, rate)

    short = msfc_ccd.dark.current(b[{axis: slice(None, 3)}], axis)
    long = msfc_ccd.dark.current(b[{axis: slice(2, None)}], axis)

    assert np.all(np.abs(long.outputs - short.outputs) < 0.01 * rate)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_current_same_exposure(a: msfc_ccd.abc.AbstractTapData):
    """Darks of a single exposure length have no slope to fit."""
    axis = "_test_current_same_exposure"
    b = _darks(a, axis, 0.05 * u.DN / u.s)
    b = b.replace(inputs=b.inputs.replace(timedelta_requested=2 * u.s))
    with pytest.raises(ValueError, match="two"):
        msfc_ccd.dark.current(b, axis)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_current_flats(a: msfc_ccd.abc.AbstractTapData):
    """Images far brighter than a dark are not darks."""
    axis = "_test_current_flats"
    b = _darks(a, axis, 1000 * u.DN / u.s)
    with pytest.raises(ValueError, match="signal_max"):
        msfc_ccd.dark.current(b, axis)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_images + _taps,
)
def test_master_missing_axis(a: msfc_ccd.abc.AbstractCameraData):
    """There is nothing to average along an axis the images do not have."""
    with pytest.raises(ValueError, match="no axis"):
        msfc_ccd.dark.master(a, "_test_master_missing_axis")
