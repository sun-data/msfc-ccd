import abc
import numpy as np
import astropy.units as u
import named_arrays as na
import msfc_ccd


class AbstractTestAbstractImageData(
    abc.ABC,
):

    def test_axis_x(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.axis_x
        assert isinstance(result, str)
        assert result in a.outputs.shape

    def test_axis_y(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.axis_y
        assert isinstance(result, str)
        assert result in a.outputs.shape

    def test_axis_xy(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.axis_xy
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert a.axis_x in result
        assert a.axis_y in result

    def test_num_x(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.num_x
        assert isinstance(result, int)

    def test_num_y(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.num_y
        assert isinstance(result, int)


class AbstractTestAbstractCameraData(
    AbstractTestAbstractImageData,
):

    def test_camera(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.camera
        if result is not None:
            assert isinstance(result, msfc_ccd.abc.AbstractCamera)

    def test_despiked(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.despiked
        assert isinstance(result, type(a))
        assert (result.outputs - a.outputs).mean() < 1e-6 * u.DN

    def test_dark(self, a: msfc_ccd.abc.AbstractCameraData):
        axis = "_test_dark"
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
        result = b.dark(axis)
        assert isinstance(result, type(a))
        assert axis not in result.outputs.shape
        assert result.outputs.shape[a.axis_x] == a.outputs.shape[a.axis_x]
        assert result.outputs.shape[a.axis_y] == a.outputs.shape[a.axis_y]
        residual = result.outputs - a.outputs
        assert np.all(np.abs(residual[{a.axis_x: 100, a.axis_y: 100}]) < 10 * sigma)
        assert np.all(np.abs(residual.mean()) < 0.1 * sigma)
        assert np.all(residual.std() < sigma / np.sqrt(num / 2))

    @abc.abstractmethod
    def test_unbiased(
        self,
        a: msfc_ccd.abc.AbstractImageData,
    ):
        pass

    @abc.abstractmethod
    def test_active(
        self,
        a: msfc_ccd.abc.AbstractImageData,
    ):
        pass

    def test_electrons(
        self,
        a: msfc_ccd.abc.AbstractImageData,
    ):
        result = a.electrons
        assert result.sum() != 0 * u.electron
