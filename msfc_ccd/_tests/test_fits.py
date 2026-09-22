import pytest
import pathlib
import gzip
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
    header = result.inputs
    assert np.all(np.char.startswith(header.camera_id.ndarray, "ESIS"))
    assert np.all(np.isin(header.serial_number.ndarray, ["6", "9"]))
    assert np.all(header.run_mode.ndarray == "Synchronous External Trigger")
    assert np.all(header.status.ndarray == "Complete")
    assert np.all(header.sequence_number >= 0)
    assert np.all(header.count >= 0)


def test_open_uncompressed(tmp_path: pathlib.Path):
    """Every sample is compressed, so decompress one to cover plain FITS."""
    path_compressed = msfc_ccd.samples.path_fe55_esis1
    path = tmp_path / path_compressed.stem

    with gzip.open(path_compressed, "rb") as f:
        path.write_bytes(f.read())

    result = msfc_ccd.fits.open(path)
    assert isinstance(result, msfc_ccd.SensorData)
    assert result.outputs.sum() != 0
    assert np.all(result.outputs == msfc_ccd.fits.open(path_compressed).outputs)


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
