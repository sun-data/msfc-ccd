from typing_extensions import Self
import dataclasses
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
        num_blank: None | int = 0,
        num_overscan: None | int = None,
    ) -> Self:
        """
        Compute the bias (or pedastal) for each tap.

        Select a number of blank pixels and a number of overscan pixels and
        take the mean to compute the bias.

        Parameters
        ----------
        num_blank
            The number of blank columns to use starting from those closest
            to the active pixels.
            If :obj:`None`, all the blank pixels are used.
        num_overscan
            The number of overscan columns to use starting from those closest
            to the active pixels.
            If :obj:`None` (the default), all the overscan pixels are used.


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
