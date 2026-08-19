import pytest
import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd

_paths = [
    msfc_ccd.samples.path_fe55_esis1,
    na.ScalarArray(
        ndarray=np.array(
            [
                msfc_ccd.samples.path_fe55_esis1,
                msfc_ccd.samples.path_fe55_esis3,
            ]
        ),
        axes="time",
    ),
]

_camera = msfc_ccd.Camera(
    gain=na.ScalarArray(
        ndarray=[[2.5, 2.6], [2.7, 2.8]] * u.electron / u.DN,
        axes=("tap_x", "tap_y"),
    ),
)


@pytest.mark.parametrize(
    argnames="path",
    argvalues=_paths,
)
def test_open(path: str | pathlib.Path | na.AbstractScalarArray):
    result = msfc_ccd.fits.open(path)
    assert isinstance(result, msfc_ccd.SensorData)
    assert result.outputs.sum() != 0


@pytest.mark.parametrize(
    argnames="path",
    argvalues=_paths,
)
def test_open_electrons(path: str | pathlib.Path | na.AbstractScalarArray):
    image = msfc_ccd.fits.open(path, camera=_camera)

    taps = image.taps.electrons
    assert isinstance(taps, msfc_ccd.TapData)
    assert na.unit(taps.outputs).is_equivalent(u.electron)
    assert taps.outputs.sum() != 0 * u.electron

    result = image.electrons
    assert isinstance(result, msfc_ccd.SensorData)
    assert na.unit(result.outputs).is_equivalent(u.electron)
    assert result.outputs.sum() != 0 * u.electron


@pytest.mark.parametrize(
    argnames="path",
    argvalues=_paths,
)
def test_open_electrons_default_camera(
    path: str | pathlib.Path | na.AbstractScalarArray,
):
    """The default camera has no gain, so ``electrons`` should say so plainly."""
    image = msfc_ccd.fits.open(path)
    with pytest.raises(ValueError, match="`gain` is `None`"):
        image.taps.electrons
    with pytest.raises(ValueError, match="`gain` is `None`"):
        image.electrons
