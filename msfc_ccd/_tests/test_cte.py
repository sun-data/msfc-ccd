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
def test_fe55(a: msfc_ccd.abc.AbstractTapData):
    sensor = a.camera.sensor
    gain = 2.5 * u.electron / u.DN
    cti_x = 5e-5
    cti_y = 1e-4
    source = msfc_ccd.Fe55()
    rng = np.random.default_rng(seed=42)

    num_x, num_y, step = 60, 29, 17
    start_x = sensor.num_blank + 8
    start_y = 16

    # Single-pixel Fe 55 events on a grid, spaced to stay isolated
    probability_beta = source.probability_k_beta / (
        source.probability_k_alpha + source.probability_k_beta
    )
    where_beta = rng.random(size=(num_x, num_y)) < probability_beta
    charge = np.where(where_beta, source.charge_k_beta, source.charge_k_alpha)
    charge = charge / gain + rng.normal(scale=5, size=charge.shape) * u.DN

    # The fraction lost to the transfers before each event is read out
    transfers_x = start_x + 1 + step * np.arange(num_x)[:, np.newaxis]
    transfers_y = start_y + 1 + step * np.arange(num_y)[np.newaxis, :]
    charge = charge * (1 - cti_x * transfers_x - cti_y * transfers_y)
    charge = na.ScalarArray(charge, axes=(a.axis_x, a.axis_y))

    index = {
        a.axis_x: slice(start_x, start_x + num_x * step, step),
        a.axis_y: slice(start_y, start_y + num_y * step, step),
    }
    outputs = a.outputs.copy()
    outputs[index] = outputs[index] + charge

    result = msfc_ccd.cte.fe55(a.replace(outputs=outputs))

    assert isinstance(result, msfc_ccd.TapData)
    assert isinstance(result.outputs, na.Cartesian2dVectorArray)
    cti_result = 1 - result.outputs.to(u.dimensionless_unscaled)
    assert np.all(np.abs(cti_result.x - cti_x) < 0.1 * cti_x)
    assert np.all(np.abs(cti_result.y - cti_y) < 0.1 * cti_y)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_fe55_without_events(a: msfc_ccd.abc.AbstractTapData):
    """A tap with too few events has no efficiency to report."""
    result = msfc_ccd.cte.fe55(a)
    assert np.all(np.isnan(result.outputs.x))
    assert np.all(np.isnan(result.outputs.y))


@pytest.mark.parametrize(
    argnames="a",
    argvalues=_taps,
)
def test_eper(a: msfc_ccd.abc.AbstractTapData):
    sensor = a.camera.sensor
    signal = 10000 * u.DN
    cti = 5e-5

    # A flat, and the charge its last active pixel leaves in the overscan
    where_active = ~(a.where_blank() | a.where_overscan())
    num_transfers = a.num_x - sensor.num_overscan
    deferred = signal * cti * num_transfers
    i = a.outputs.indices[a.axis_x]
    overscan = np.where(i == num_transfers, 0.9, 0)
    overscan = overscan + np.where(i == num_transfers + 1, 0.1, 0)
    outputs = a.outputs + signal * where_active + deferred * overscan

    result = msfc_ccd.cte.eper(a.replace(outputs=outputs))

    assert isinstance(result, msfc_ccd.TapData)
    assert a.axis_x not in result.outputs.shape
    assert a.axis_y not in result.outputs.shape
    assert na.unit(result.outputs).is_equivalent(u.percent)
    cti_result = 1 - result.outputs.to(u.dimensionless_unscaled)
    assert np.all(np.abs(cti_result - cti) < 0.02 * cti)
