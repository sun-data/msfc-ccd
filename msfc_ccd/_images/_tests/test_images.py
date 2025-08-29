import abc
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

    def test_camera(self, a: msfc_ccd.abc.AbstractImageData):
        result = a.camera
        if result is not None:
            assert isinstance(result, msfc_ccd.abc.AbstractCamera)
