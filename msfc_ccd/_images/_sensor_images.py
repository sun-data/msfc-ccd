from __future__ import annotations
from typing_extensions import Self
import dataclasses
import pathlib
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import named_arrays as na
import msfc_ccd
from .._cameras import AbstractCamera
from ._vectors import ImageHeader
from ._images import AbstractCameraData

__all__ = [
    "SensorData",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSensorData(
    AbstractCameraData,
):
    """An interface for representing data captured by an entire image sensor."""

    @property
    def taps(self) -> msfc_ccd.TapData:
        """Split the data into separate images for each tap."""
        axis_x = self.axis_x
        axis_y = self.axis_y

        axis_tap_x = self.camera.axis_tap_x
        axis_tap_y = self.camera.axis_tap_y

        num_x = self.num_x
        num_y = self.num_y

        num_tap_x = self.camera.sensor.num_tap_x
        num_tap_y = self.camera.sensor.num_tap_y

        num_x_new = num_x // num_tap_x
        num_y_new = num_y // num_tap_y

        slice_left_x = {axis_x: slice(None, num_x_new)}
        slice_left_y = {axis_y: slice(None, num_y_new)}

        slice_right_x = {axis_x: slice(None, num_x_new - 1, -1)}
        slice_right_y = {axis_y: slice(None, num_y_new - 1, -1)}

        x = self.inputs.pixel.x
        y = self.inputs.pixel.y
        outputs = self.outputs

        x = [x[slice_left_x], x[slice_right_x]]
        y = [y[slice_left_y], y[slice_right_y]]

        x = na.stack(x, axis=axis_tap_x)
        y = na.stack(y, axis=axis_tap_y)

        outputs_00 = outputs[slice_left_x][slice_left_y]
        outputs_01 = outputs[slice_left_x][slice_right_y]
        outputs_10 = outputs[slice_right_x][slice_left_y]
        outputs_11 = outputs[slice_right_x][slice_right_y]

        outputs = [
            na.stack([outputs_00, outputs_10], axis=axis_tap_x),
            na.stack([outputs_01, outputs_11], axis=axis_tap_x),
        ]

        outputs = na.stack(outputs, axis=axis_tap_y)

        return msfc_ccd.TapData(
            inputs=dataclasses.replace(
                self.inputs,
                pixel=na.Cartesian2dVectorArray(x, y),
            ),
            outputs=outputs,
            camera=self.camera,
            axis_x=self.axis_x,
            axis_y=self.axis_y,
        )


@dataclasses.dataclass(eq=False, repr=False)
class SensorData(
    AbstractSensorData,
):
    """
    A single image or a sequence of images captured by the MSFC camera.

    Examples
    --------
    Load a sample image and display it.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        import msfc_ccd

        # Load the sample image
        image = msfc_ccd.SensorData.from_fits(
            path=msfc_ccd.samples.path_fe55_esis1,
            camera=msfc_ccd.Camera(),
        )

        # Display the sample image
        fig, ax = plt.subplots(
            constrained_layout=True,
        )
        im = na.plt.imshow(
            image.outputs.value,
            axis_x=image.axis_x,
            axis_y=image.axis_y,
            ax=ax,
        );
    """

    inputs: ImageHeader = dataclasses.MISSING
    """A vector which contains the FITS header for each image."""

    outputs: na.ScalarArray = dataclasses.MISSING
    """The underlying array storing the image data."""

    camera: AbstractCamera = dataclasses.MISSING
    """A model of the sensor used to capture these images."""

    axis_x: str = dataclasses.field(default="detector_x", kw_only=True)
    """The name of the horizontal axis."""

    axis_y: str = dataclasses.field(default="detector_y", kw_only=True)
    """The name of the vertical axis."""

    @classmethod
    def from_fits(
        cls,
        path: str | pathlib.Path | na.AbstractScalarArray,
        camera: AbstractCamera,
        axis_x: str = "detector_x",
        axis_y: str = "detector_y",
    ) -> Self:
        """
        Load an image or an array of images from a FITS file or an array of files.

        Parameters
        ----------
        path
            Either a single path or an array of paths pointing to the FITS files
            to load.
        camera
            A model of the camera used to capture these images.
        axis_x
            The name of the logical axis representing the horizontal dimension of
            the images.
        axis_y
            The name of the logical axis representing the vertical dimension of
            the images.
        """
        path = na.as_named_array(path)

        time_start = na.ScalarArray.zeros(na.shape(path))

        timedelta = np.empty_like(path, dtype=np.int64)
        timedelta_requested = np.empty_like(path, dtype=float) << u.s
        serial_number = np.empty_like(path, dtype=str)
        run_mode = np.empty_like(path, dtype=str)
        status = np.empty_like(path, dtype=str)
        voltage_fpga_vccint = np.empty_like(path, dtype=int)
        voltage_fpga_vccaux = np.empty_like(path, dtype=int)
        voltage_fpga_vccbram = np.empty_like(path, dtype=int)
        temperature_fpga = np.empty_like(path, dtype=int)
        temperature_adc_1 = np.empty_like(path, dtype=int)
        temperature_adc_2 = np.empty_like(path, dtype=int)
        temperature_adc_3 = np.empty_like(path, dtype=int)
        temperature_adc_4 = np.empty_like(path, dtype=int)

        for i, index in enumerate(path.ndindex()):

            hdu = astropy.io.fits.open(path[index].ndarray)[0]

            data_index = na.ScalarArray(
                ndarray=hdu.data,
                axes=(axis_y, axis_x),
            )

            if i == 0:
                data = na.ScalarArray.empty(
                    shape=na.broadcast_shapes(path.shape, data_index.shape),
                    dtype=float,
                )

            data[index] = data_index

            header = hdu.header
            time_start[index] = astropy.time.Time(header["IMG_TS"]).jd
            timedelta[index] = header["MEAS_EXP"]
            timedelta_requested[index] = header["IMG_EXP"] * u.ms
            serial_number[index] = header.get("CAM_SN")
            run_mode[index] = header.get("RUN_MODE")
            status[index] = header.get("IMG_STAT")
            voltage_fpga_vccint[index] = header["FPGAVINT"]
            voltage_fpga_vccaux[index] = header["FPGAVAUX"]
            voltage_fpga_vccbram[index] = header["FPGAVBRM"]
            temperature_fpga[index] = header["FPGATEMP"]
            temperature_adc_1[index] = header["ADCTEMP1"]
            temperature_adc_2[index] = header["ADCTEMP2"]
            temperature_adc_3[index] = header["ADCTEMP3"]
            temperature_adc_4[index] = header["ADCTEMP4"]

        timedelta = camera.calibrate_timedelta_exposure(timedelta)
        voltage_fpga_vccint = camera.calibrate_voltage_fpga(voltage_fpga_vccint)
        voltage_fpga_vccaux = camera.calibrate_voltage_fpga(voltage_fpga_vccaux)
        voltage_fpga_vccbram = camera.calibrate_voltage_fpga(voltage_fpga_vccbram)
        temperature_fpga = camera.calibrate_temperature_fpga(temperature_fpga)
        temperature_adc_1 = camera.calibrate_temperature_adc_1(temperature_adc_1)
        temperature_adc_2 = camera.calibrate_temperature_adc_234(temperature_adc_2)
        temperature_adc_3 = camera.calibrate_temperature_adc_234(temperature_adc_3)
        temperature_adc_4 = camera.calibrate_temperature_adc_234(temperature_adc_4)

        shape = data.shape

        shape_img = {
            axis_x: shape[axis_x],
            axis_y: shape[axis_y],
        }

        pixel = na.indices(shape_img)

        pixel = na.Cartesian2dVectorArray(
            x=pixel[axis_x],
            y=pixel[axis_y],
        )

        t = astropy.time.Time(
            val=time_start.ndarray,
            format="jd",
        )
        t.format = "isot"
        time_start.ndarray = t

        for axis in serial_number.shape:
            sn0 = serial_number[{axis: 0}]
            if np.all(sn0 == serial_number):
                serial_number = sn0

        return cls(
            inputs=ImageHeader(
                pixel=pixel,
                time_start=time_start,
                timedelta=timedelta,
                timedelta_requested=timedelta_requested,
                serial_number=serial_number,
                run_mode=run_mode,
                status=status,
                voltage_fpga_vccint=voltage_fpga_vccint,
                voltage_fpga_vccaux=voltage_fpga_vccaux,
                voltage_fpga_vccbram=voltage_fpga_vccbram,
                temperature_fpga=temperature_fpga,
                temperature_adc_1=temperature_adc_1,
                temperature_adc_2=temperature_adc_2,
                temperature_adc_3=temperature_adc_3,
                temperature_adc_4=temperature_adc_4,
            ),
            outputs=data << u.DN,
            axis_x=axis_x,
            axis_y=axis_y,
            camera=camera,
        )

    def from_taps(
        self,
        taps: msfc_ccd.TapData,
    ) -> Self:
        """
        Create a new copy of this object by concatenating the data from `taps`.

        This method does not concatenate the ``taps.inputs.pixel`` coordinates.
        Instead, this method creates a new set of coordinates from the final shape.

        Parameters
        ----------
        taps
            The data from each tap.
        """
        axis_x = taps.axis_x
        axis_y = taps.axis_y

        axis_tap_x = taps.axis_tap_x
        axis_tap_y = taps.axis_tap_y

        reverse_x = {axis_x: slice(None, None, -1)}
        reverse_y = {axis_y: slice(None, None, -1)}

        a = taps.outputs
        a_00 = a[{axis_tap_x: 0, axis_tap_y: 0}]
        a_01 = a[{axis_tap_x: 0, axis_tap_y: 1}]
        a_10 = a[{axis_tap_x: 1, axis_tap_y: 0}]
        a_11 = a[{axis_tap_x: 1, axis_tap_y: 1}]

        a_01 = a_01[reverse_y]
        a_10 = a_10[reverse_x]
        a_11 = a_11[reverse_x][reverse_y]

        a = na.concatenate(
            arrays=[
                na.concatenate([a_00, a_10], axis=axis_x),
                na.concatenate([a_01, a_11], axis=axis_x),
            ],
            axis=axis_y,
        )

        shape = a.shape
        pixel = na.Cartesian2dVectorArray(
            x=na.ScalarArrayRange(0, shape[axis_x], axis_x),
            y=na.ScalarArrayRange(0, shape[axis_y], axis_y),
        )

        return self.replace(
            inputs=taps.inputs.replace(pixel=pixel),
            outputs=a,
            camera=taps.camera,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    @property
    def unbiased(self) -> Self:
        taps = self.taps.unbiased
        return self.from_taps(taps)

    @property
    def active(self) -> Self:
        taps = self.taps.active
        return self.from_taps(taps)

    @property
    def electrons(self) -> Self:
        taps = self.taps.electrons
        return self.from_taps(taps)
