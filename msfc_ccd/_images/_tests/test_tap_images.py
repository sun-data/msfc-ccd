import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd
from . import test_images

_camera = msfc_ccd.Camera(
    gain=na.ScalarArray(
        ndarray=[[2.5, 2.6], [2.7, 2.8]] * u.electron / u.DN,
        axes=("tx", "ty"),
    ),
    axis_tap_x="tx",
    axis_tap_y="ty",
)


class AbstractTestAbstractTapImage(
    test_images.AbstractTestAbstractCameraData,
):

    def test_axis_tap_x(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.axis_tap_x
        assert isinstance(result, str)
        assert result in a.outputs.shape

    def test_axis_tap_y(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.axis_tap_y
        assert isinstance(result, str)
        assert result in a.outputs.shape

    def test_tap(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.tap
        for ax in result:
            assert isinstance(ax, str)
            assert isinstance(result[ax], na.AbstractScalarArray)

    def test_label(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.label
        for s in result.ndarray.flat:
            assert isinstance(s, str)

    def test_amplifier(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.amplifier
        assert result.shape[a.axis_tap_x] == a.shape[a.axis_tap_x]
        assert result.shape[a.axis_tap_y] == a.shape[a.axis_tap_y]
        assert set(result.ndarray.flat) == {"1/E", "2/H", "3/G", "4/F"}

    def test_where_blank(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.where_blank()
        assert result.sum() == a.camera.sensor.num_blank

    def test_where_overscan(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.where_overscan()
        assert result.sum() == a.camera.sensor.num_overscan

    def test_where_masked(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.where_masked()
        assert result.sum() == a.camera.sensor.num_masked
        assert result[{a.axis_y: 0}]

    def test_bias(self, a: msfc_ccd.abc.AbstractTapData):
        result = a.bias()
        axis_tap_x = a.camera.axis_tap_x
        axis_tap_y = a.camera.axis_tap_y
        assert na.unit(result.outputs) == na.unit(a.outputs)
        assert result.shape[axis_tap_x] == a.shape[axis_tap_x]
        assert result.shape[axis_tap_y] == a.shape[axis_tap_y]

    def test_bias_ignores_overscan(self, a: msfc_ccd.abc.AbstractTapData):
        b = a.replace(outputs=a.outputs + 1000 * u.DN * a.where_overscan())
        assert np.all(a.bias().outputs == b.bias().outputs)
        assert np.all(
            a.bias(num_overscan=None).outputs != b.bias(num_overscan=None).outputs
        )

    def test_unbiased(self, a: msfc_ccd.abc.AbstractTapData):
        super().test_unbiased(a)
        result = a.unbiased
        assert isinstance(result, msfc_ccd.TapData)
        assert na.unit(result.outputs) == na.unit(a.outputs)
        assert np.abs(result.outputs.mean()) < 1 * u.DN

    def test_readout_noise(self, a: msfc_ccd.abc.AbstractTapData):
        axis = "_test_readout_noise"
        num = 5
        sigma = 3 * u.DN
        rng = np.random.default_rng(seed=42)
        outputs = a.outputs + na.ScalarArray(
            ndarray=rng.normal(size=(num,) + a.outputs.ndarray.shape) * sigma,
            axes=(axis,) + a.outputs.axes,
        )
        outputs[{axis: 2, a.axis_x: 100, a.axis_y: 100}] = 60000 * u.DN
        b = a.replace(outputs=outputs)
        result = b.readout_noise(axis)
        assert isinstance(result, msfc_ccd.TapData)
        assert result.outputs.shape[axis] == num - 1
        assert a.axis_x not in result.outputs.shape
        assert a.axis_y not in result.outputs.shape
        assert np.all(np.abs(result.outputs - sigma) < 0.05 * sigma)

    def test_hits(self, a: msfc_ccd.abc.AbstractTapData):
        sensor = a.camera.sensor
        num_blank = sensor.num_blank
        charge = 600 * u.DN

        outputs = a.outputs.copy()

        # Isolated events, spaced far enough apart to stay isolated
        index = [
            {a.axis_x: num_blank + 10 + 71 * k, a.axis_y: 10 + 53 * k} for k in range(8)
        ]
        for i in index:
            outputs[i] = outputs[i] + charge

        # Two adjacent hot pixels, which are not an isolated event
        index_pair = {a.axis_x: num_blank + 800, a.axis_y: 400}
        index_next = {a.axis_x: num_blank + 801, a.axis_y: 400}
        outputs[index_pair] = outputs[index_pair] + charge
        outputs[index_next] = outputs[index_next] + charge

        result = a.replace(outputs=outputs).hits()

        assert isinstance(result, msfc_ccd.TapData)
        assert result.shape[a.axis_x] == a.active.shape[a.axis_x]
        assert result.shape[a.axis_y] == a.shape[a.axis_y]

        # Every isolated event is found, with its charge
        for i in index:
            i = {a.axis_x: i[a.axis_x] - num_blank, a.axis_y: i[a.axis_y]}
            assert np.all(np.isfinite(result.outputs[i]))
            assert np.all(np.abs(result.outputs[i] - charge) < 0.1 * charge)

        # The adjacent pair is not
        i = {a.axis_x: index_pair[a.axis_x] - num_blank, a.axis_y: index_pair[a.axis_y]}
        assert not np.any(np.isfinite(result.outputs[i]))

    def test_hits_single(self, a: msfc_ccd.abc.AbstractTapData):
        num_blank = a.camera.sensor.num_blank
        charge = 600 * u.DN

        # The dark level and the noise of the active pixels, so that pixels
        # can be set to a known height above the dark level
        level = np.median(a.active.outputs, axis=a.axis_xy)
        deviation = np.abs(a.active.outputs - level)
        sigma = 1.482602218505602 * np.median(deviation, axis=a.axis_xy)

        outputs = a.outputs.copy()

        # A single-pixel event
        index = {a.axis_x: num_blank + 100, a.axis_y: 100}
        outputs[index] = level + charge

        # An event with one neighbor between the split and event thresholds
        index_split = {a.axis_x: num_blank + 300, a.axis_y: 300}
        index_neighbor = {a.axis_x: num_blank + 301, a.axis_y: 300}
        outputs[index_split] = level + charge
        outputs[index_neighbor] = level + 4 * sigma

        b = a.replace(outputs=outputs)
        result = b.hits(threshold_split=3)
        result_all = b.hits()

        i = {a.axis_x: index[a.axis_x] - num_blank, a.axis_y: index[a.axis_y]}
        assert np.all(np.abs(result.outputs[i] - charge) < 1 * u.DN)

        i = {
            a.axis_x: index_split[a.axis_x] - num_blank,
            a.axis_y: index_split[a.axis_y],
        }
        assert not np.any(np.isfinite(result.outputs[i]))
        assert np.all(np.isfinite(result_all.outputs[i]))

    def test_gain(self, a: msfc_ccd.abc.AbstractTapData):
        gain = 2.5 * u.electron / u.DN
        fe55 = msfc_ccd.Fe55()
        rng = np.random.default_rng(seed=42)

        num_blank = a.camera.sensor.num_blank
        num_x, num_y, step = 30, 15, 33

        # Fe 55 events on a grid, spaced far enough apart to stay isolated
        probability_beta = fe55.probability_k_beta / (
            fe55.probability_k_alpha + fe55.probability_k_beta
        )
        where_beta = rng.random(size=(num_x, num_y)) < probability_beta
        charge = np.where(where_beta, fe55.charge_k_beta, fe55.charge_k_alpha)
        charge = charge / gain + rng.normal(scale=14, size=charge.shape) * u.DN
        charge = na.ScalarArray(charge, axes=(a.axis_x, a.axis_y))

        index = {
            a.axis_x: slice(num_blank + 8, num_blank + 8 + num_x * step, step),
            a.axis_y: slice(8, 8 + num_y * step, step),
        }
        outputs = a.outputs.copy()
        outputs[index] = outputs[index] + charge

        result = a.replace(outputs=outputs).gain()

        assert isinstance(result, msfc_ccd.TapData)
        assert a.axis_x not in result.outputs.shape
        assert a.axis_y not in result.outputs.shape
        assert na.unit(result.outputs).is_equivalent(u.electron / u.DN)
        assert np.all(np.abs(result.outputs - gain) < 0.02 * gain)

    def test_gain_without_events(self, a: msfc_ccd.abc.AbstractTapData):
        """A tap with too few events has no gain to report."""
        result = a.gain()
        assert np.all(np.isnan(result.outputs))

    def test_cte_eper(self, a: msfc_ccd.abc.AbstractTapData):
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

        result = a.replace(outputs=outputs).cte_eper()

        assert isinstance(result, msfc_ccd.TapData)
        assert a.axis_x not in result.outputs.shape
        assert a.axis_y not in result.outputs.shape
        assert na.unit(result.outputs).is_equivalent(u.percent)
        cti_result = 1 - result.outputs.to(u.dimensionless_unscaled)
        assert np.all(np.abs(cti_result - cti) < 0.02 * cti)

    def test_cte_fe55(self, a: msfc_ccd.abc.AbstractTapData):
        sensor = a.camera.sensor
        gain = 2.5 * u.electron / u.DN
        cti_x = 5e-5
        cti_y = 1e-4
        fe55 = msfc_ccd.Fe55()
        rng = np.random.default_rng(seed=42)

        num_x, num_y, step = 60, 29, 17
        start_x = sensor.num_blank + 8
        start_y = 16

        # Single-pixel Fe 55 events on a grid, spaced to stay isolated
        probability_beta = fe55.probability_k_beta / (
            fe55.probability_k_alpha + fe55.probability_k_beta
        )
        where_beta = rng.random(size=(num_x, num_y)) < probability_beta
        charge = np.where(where_beta, fe55.charge_k_beta, fe55.charge_k_alpha)
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

        result = a.replace(outputs=outputs).cte_fe55()

        assert isinstance(result, msfc_ccd.TapData)
        assert isinstance(result.outputs, na.Cartesian2dVectorArray)
        cti_result = 1 - result.outputs.to(u.dimensionless_unscaled)
        assert np.all(np.abs(cti_result.x - cti_x) < 0.1 * cti_x)
        assert np.all(np.abs(cti_result.y - cti_y) < 0.1 * cti_y)

    def test_cte_fe55_without_events(self, a: msfc_ccd.abc.AbstractTapData):
        """A tap with too few events has no efficiency to report."""
        result = a.cte_fe55()
        assert np.all(np.isnan(result.outputs.x))
        assert np.all(np.isnan(result.outputs.y))

    @classmethod
    def _flats(
        cls,
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

    def test_photon_transfer(self, a: msfc_ccd.abc.AbstractTapData):
        axis = "_test_photon_transfer"
        num = 4
        signal = 4000 * u.DN
        gain = 2.5 * u.electron / u.DN
        readout_noise = 4 * u.DN
        b = self._flats(a, axis, signal, gain, readout_noise, num=num)

        result = b.photon_transfer(axis)

        assert isinstance(result, na.FunctionArray)
        assert result.outputs.shape[axis] == num - 1
        assert a.axis_x not in result.outputs.shape
        assert a.axis_y not in result.outputs.shape
        assert na.unit(result.inputs).is_equivalent(u.DN)
        assert na.unit(result.outputs).is_equivalent(u.DN**2)

        expected = result.inputs * u.electron / gain + np.square(readout_noise)
        assert np.all(np.abs(result.outputs / expected - 1) < 0.01)

    def test_gain_photon_transfer(self, a: msfc_ccd.abc.AbstractTapData):
        axis = "_test_gain_photon_transfer"
        signal = na.ScalarArray([2000, 6000] * u.DN, axes="_test_level")
        gain = 2.5 * u.electron / u.DN
        b = self._flats(a, axis, signal, gain, readout_noise=4 * u.DN)

        result = b.gain_photon_transfer(axis)

        assert isinstance(result, msfc_ccd.TapData)
        assert set(result.outputs.shape) == {a.axis_tap_x, a.axis_tap_y}
        assert na.unit(result.outputs).is_equivalent(u.electron / u.DN)
        assert np.all(np.abs(result.outputs - gain) < 0.01 * gain)

    @classmethod
    def _darks(
        cls,
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
            inputs=a.inputs.replace(timedelta=timedelta),
            outputs=outputs,
        )

    def test_dark_current(self, a: msfc_ccd.abc.AbstractTapData):
        axis = "_test_dark_current"
        rate = 0.05 * u.DN / u.s
        result = self._darks(a, axis, rate).dark_current(axis)

        assert isinstance(result, msfc_ccd.TapData)
        assert axis not in result.outputs.shape
        assert a.axis_x not in result.outputs.shape
        assert a.axis_y not in result.outputs.shape
        assert np.all(np.abs(result.outputs - rate) < 0.01 * rate)

    def test_dark_current_ignores_masked(self, a: msfc_ccd.abc.AbstractTapData):
        """The masked rows accumulate dark current much faster than the image."""
        axis = "_test_dark_current_ignores_masked"
        rate = 0.05 * u.DN / u.s
        b = self._darks(a, axis, rate)

        # Only the light-sensitive part of the masked rows, since the blank
        # columns are clocked out of the serial register.
        where_active = ~(b.where_blank() | b.where_overscan())
        where = b.where_masked() & where_active
        b = b.replace(outputs=b.outputs + 100 * rate * b.inputs.timedelta * where)
        result = b.dark_current(axis)

        assert np.all(np.abs(result.outputs - rate) < 0.01 * rate)

    def test_dark_current_linear(self, a: msfc_ccd.abc.AbstractTapData):
        """The rate must not depend on which exposures were used."""
        axis = "_test_dark_current_linear"
        rate = 0.05 * u.DN / u.s
        b = self._darks(a, axis, rate)

        short = b[{axis: slice(None, 3)}].dark_current(axis)
        long = b[{axis: slice(2, None)}].dark_current(axis)

        assert np.all(np.abs(long.outputs - short.outputs) < 0.01 * rate)

    def test_active(self, a: msfc_ccd.abc.AbstractTapData):
        super().test_active(a)
        sensor = a.camera.sensor
        num_nap = sensor.num_blank + sensor.num_overscan
        result = a.active
        assert isinstance(result, msfc_ccd.TapData)
        assert result.shape[a.axis_x] == a.shape[a.axis_x] - num_nap
        assert result.shape[a.axis_y] == a.shape[a.axis_y]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        msfc_ccd.fits.open(msfc_ccd.samples.path_dark_esis1, _camera).taps,
    ],
)
class TestTapImage(
    AbstractTestAbstractTapImage,
):
    pass
