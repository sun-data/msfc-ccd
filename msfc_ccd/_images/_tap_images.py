from typing_extensions import Self
import dataclasses
import numpy as np
import named_arrays as na
from .._cameras import AbstractCamera
from ._vectors import ImageHeader
from ._images import AbstractCameraData

__all__ = [
    "TapData",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTapData(
    AbstractCameraData,
):
    """An interface for representing data gathered by a single tap."""

    @property
    def axis_tap_x(self) -> str:
        """The name of the horizontal tap axis."""
        return self.camera.axis_tap_x

    @property
    def axis_tap_y(self) -> str:
        """The name of the vertical tap axis."""
        return self.camera.axis_tap_y

    @property
    def tap(self) -> dict[str, na.AbstractScalarArray]:
        """The 2-dimensional index of the tap corresponding to each image."""
        axis_tap_x = self.axis_tap_x
        axis_tap_y = self.axis_tap_y
        shape = self.outputs.shape
        shape_img = {
            axis_tap_x: shape[axis_tap_x],
            axis_tap_y: shape[axis_tap_y],
        }
        return na.indices(shape_img)

    @property
    def label(self) -> na.ScalarArray:
        """Human-readable name of the tap used often for plotting."""
        axis_tap_x = self.axis_tap_x
        axis_tap_y = self.axis_tap_y
        tap_x = self.tap[axis_tap_x].astype(str).astype(object)
        tap_y = self.tap[axis_tap_y].astype(str)
        return "tap (" + tap_x + ", " + tap_y + ")"

    def where_blank(
        self,
        num: None | int = None,
    ) -> na.ScalarArray:
        """
        Create a boolean array which is :obj:`True` for all the blank columns.

        Parameters
        ----------
        num
            The number of blank columns to use starting from those closest
            to the active pixels.
            If :obj:`None` (the default), all the blank pixels are used.
        """
        if num is None:
            num = self.camera.sensor.num_blank

        i = self.outputs.indices[self.axis_x]
        lower = (self.camera.sensor.num_blank - num) <= i
        upper = i < self.camera.sensor.num_blank

        return lower & upper

    def where_overscan(
        self,
        num: None | int = None,
    ) -> na.ScalarArray:
        """
        Create a boolean array which is :obj:`True` for all the overscan columns.

        Parameters
        ----------
        num
            The number of overscan columns to use starting from those closest
            to the active pixels.
            If :obj:`None` (the default), all the overscan pixels are used.
        """
        if num is None:
            num = self.camera.sensor.num_overscan

        i = self.outputs.indices[self.axis_x]
        overscan_start = self.num_x - self.camera.sensor.num_overscan
        lower = overscan_start <= i
        upper = i < (overscan_start + num)

        return lower & upper

    def bias(
        self,
        num_blank: None | int = 25,
        num_overscan: None | int = 0,
    ) -> Self:
        """
        Compute the bias (or pedestal) for each tap.

        Select a number of blank pixels and a number of overscan pixels and
        take the mean to compute the bias.

        By default, only the 25 blank columns closest to the active pixels
        are used.
        The blank columns are read out before any of the active pixels,
        so they are not affected by the signal in the image,
        and the first half of the blank columns is ignored since it contains
        a transient from the start of each row.
        The overscan columns are not used by default since they contain
        charge deferred from the last active pixels in each row,
        which makes the bias depend on the brightness of the image.

        The blank columns are offset from the dark level of the active pixels
        by up to about 1 DN, and this offset is different for each tap.
        This offset is constant in time, so it is removed if a dark image
        prepared using the same bias is subtracted from the result.

        Parameters
        ----------
        num_blank
            The number of blank columns to use starting from those closest
            to the active pixels.
            If :obj:`None`, all the blank pixels are used.
        num_overscan
            The number of overscan columns to use starting from those closest
            to the active pixels.
            If :obj:`None`, all the overscan pixels are used.


        .. nblinkgallery::
            :caption: Relevant Reports
            :name: rst-link-gallery

            ../reports/bias
        """
        where_blank = self.where_blank(num_blank)
        where_overscan = self.where_overscan(num_overscan)

        where = where_blank | where_overscan

        result = dataclasses.replace(
            self,
            outputs=self.outputs.mean(
                axis=(self.axis_x, self.axis_y),
                where=where,
            ),
        )

        return result

    @property
    def unbiased(self) -> Self:
        return self - self.bias()

    @property
    def active(self) -> Self:
        sensor = self.camera.sensor
        slice_active = slice(sensor.num_blank, -sensor.num_overscan)
        slice_active = {self.axis_x: slice_active}

        return dataclasses.replace(
            self,
            inputs=dataclasses.replace(
                self.inputs,
                pixel=dataclasses.replace(
                    self.inputs.pixel,
                    x=self.inputs.pixel.x[slice_active],
                ),
            ),
            outputs=self.outputs[slice_active],
        )

    @property
    def electrons(self) -> Self:
        return self.camera.dn_to_electrons(self)

    def dark(
        self,
        axis: str,
        proportion: float = 0.25,
    ) -> Self:
        """
        Estimate the master dark image of each tap from a sequence of dark images.

        The master dark is the trimmed mean of the sequence along `axis`,
        computed independently for each pixel.
        Trimming the largest and smallest values rejects cosmic rays and
        other spikes which affect a single image, while hot pixels and the
        fixed pattern of the sensor, which are present in every image,
        are preserved.

        The images should have had their bias removed before calling this
        method, so that the result can be subtracted from other images which
        have had their bias removed in the same way.

        Parameters
        ----------
        axis
            The logical axis along which the images are a sequence of darks.
        proportion
            The fraction of the largest and smallest values to remove from each
            pixel before taking the mean.
            The number of images removed is rounded down,
            so with fewer than ``1 / proportion`` images
            nothing is removed.
        """
        outputs = na.mean_trimmed(
            a=self.outputs,
            q=proportion,
            axis=axis,
        )
        return dataclasses.replace(
            self,
            inputs=self.inputs[{axis: 0}],
            outputs=outputs,
        )

    def readout_noise(
        self,
        axis: str,
        threshold: float = 5,
    ) -> Self:
        r"""
        Estimate the readout noise of each tap from a sequence of dark images.

        The difference between each image and the previous image along `axis`
        is computed, which removes the bias, the dark current and the fixed
        pattern noise, and leaves only the readout noise of the two images.
        The readout noise is then the standard deviation of the active pixels
        in each difference image, divided by :math:`\sqrt{2}`,
        after rejecting pixels affected by cosmic rays or other spikes.

        The result has one fewer element along `axis` than the input,
        since it is defined for each pair of adjacent images.
        Take the mean along `axis` to estimate the readout noise of the camera.

        Parameters
        ----------
        axis
            The logical axis along which the images are a sequence of darks.
        threshold
            Pixels in the difference image further than this many standard
            deviations from the median are rejected.
            The standard deviation used for the rejection is estimated from the
            median absolute deviation, which is not affected by the spikes.
        """
        outputs = self.active.outputs
        outputs_1 = outputs[{axis: slice(1, None)}]
        outputs_0 = outputs[{axis: slice(None, -1)}]
        difference = outputs_1 - outputs_0

        axis_xy = (self.axis_x, self.axis_y)
        median = np.median(difference, axis=axis_xy)
        deviation = np.abs(difference - median)
        mad = np.median(deviation, axis=axis_xy)

        # The ratio of the standard deviation to the median absolute deviation
        # for a normal distribution
        factor = 1.482602218505602
        where = deviation < threshold * factor * mad

        std = difference.std(axis=axis_xy, where=where)

        return dataclasses.replace(
            self,
            inputs=self.inputs[{axis: slice(1, None)}],
            outputs=std / np.sqrt(2),
        )


@dataclasses.dataclass(eq=False, repr=False)
class TapData(
    AbstractTapData,
):
    """
    An image or a sequence of images captured from each tap of the sensor.

    Examples
    --------
    Load a sample image and split it into the four tap images.

    .. jupyter-execute::

        import named_arrays as na
        import msfc_ccd

        # Load the sample image
        image = msfc_ccd.fits.open(msfc_ccd.samples.path_fe55_esis1)

        # Split the sample image into four separate images for each tap
        taps = image.taps

        # Display the four images
        fig, axs = na.plt.subplots(
            axis_rows=taps.axis_tap_y,
            nrows=taps.outputs.shape[taps.axis_tap_y],
            axis_cols=taps.axis_tap_x,
            ncols=taps.outputs.shape[taps.axis_tap_x],
            sharex=True,
            sharey=True,
            constrained_layout=True,
        );
        na.plt.pcolormesh(
            taps.inputs.pixel,
            C=taps.outputs.value,
            ax=axs,
        );

    """

    inputs: ImageHeader = dataclasses.MISSING
    """A vector which contains the FITS header for each image."""

    outputs: na.ScalarArray = dataclasses.MISSING
    """The underlying array storing the image data."""

    camera: AbstractCamera = dataclasses.MISSING
    """A model of the camera used to capture these images."""

    axis_x: str = dataclasses.field(default="detector_x", kw_only=True)
    """The name of the horizontal axis."""

    axis_y: str = dataclasses.field(default="detector_y", kw_only=True)
    """The name of the vertical axis."""
