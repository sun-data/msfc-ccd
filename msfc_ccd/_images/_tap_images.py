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

    @property
    def amplifier(self) -> na.ScalarArray:
        """
        The name each tap is known by.

        ``E``, ``F``, ``G`` and ``H`` are the four output amplifiers of the
        sensor, named as on page 16 of the datasheet, and MSFC numbers the
        same four quadrants 1 to 4.
        Their layout across the readout frame is

        .. code-block:: text

            +-------+-------+
            |  2/H  |  3/G  |
            +-------+-------+
            |  1/E  |  4/F  |
            +-------+-------+

        with the first row read out at the bottom,
        so ``1/E`` is ``(tap_y=0, tap_x=0)`` and ``3/G`` is
        ``(tap_y=1, tap_x=1)``.
        This is the mapping needed to compare a per-tap measurement against
        the values MSFC published by quadrant number.

        Examples
        --------
        .. jupyter-execute::

            import msfc_ccd

            image = msfc_ccd.fits.open(msfc_ccd.samples.path_fe55_esis3)

            image.taps.amplifier.ndarray
        """
        return na.ScalarArray(
            ndarray=np.array([["1/E", "4/F"], ["2/H", "3/G"]]),
            axes=(self.axis_tap_y, self.axis_tap_x),
        )

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

    def where_masked(self) -> na.ScalarArray:
        """
        Create a boolean array which is :obj:`True` for all the masked rows.

        See :attr:`msfc_ccd.TeledyneCCD230.num_masked` for how these rows
        differ from the rest of the image.
        """
        i = self.outputs.indices[self.axis_y]
        return i < self.camera.sensor.num_masked

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

    def hits(
        self,
        threshold: float = 5,
        threshold_split: None | float = None,
    ) -> Self:
        r"""
        Find the isolated single-pixel events in each image.

        An :math:`^{55}\text{Fe}` X-ray, or a cosmic ray arriving close to
        normal incidence, deposits its charge in one pixel and the pixels
        immediately around it.
        This method finds every pixel more than `threshold` readout noises
        above the dark level whose eight neighbors are all below that level,
        and measures the charge of the event as the sum of the
        :math:`3 \times 3` region centered on it.

        The result has the shape of :attr:`active`, with the charge of each
        event in the pixel where it landed and :obj:`numpy.nan` everywhere
        else, so a sequence of images can be pooled by taking a histogram
        along the sequence axis and the two detector axes.

        The bias is removed, and then the median of the active pixels,
        which removes the dark current and the fixed pattern of the sensor
        to the accuracy needed to place the threshold.

        Parameters
        ----------
        threshold
            How many readout noises above the dark level a pixel must be to
            start an event.
            The readout noise is estimated from the median absolute deviation
            of the active pixels, which is not moved by the events themselves.
        threshold_split
            If given, keep only the single-pixel events, whose eight neighbors
            are all less than this many readout noises above the dark level,
            and measure the charge of each as that of the center pixel alone.
            This is what :func:`msfc_ccd.cte.fe55` needs, since the sum over the
            :math:`3 \times 3` region would recover the charge that an
            imperfect transfer leaves in a neighboring pixel.

        Examples
        --------
        Plot the distribution of event charges for one tap of the ESIS
        channel 3 camera.
        The peak is the :math:`^{55}\text{Fe}` K-:math:`\alpha` line,
        and the tail below it is the events which lost part of their charge.

        .. jupyter-execute::

            import matplotlib.pyplot as plt
            import astropy.units as u
            import named_arrays as na
            import msfc_ccd

            # Load a sample Fe 55 image and split it into taps
            image = msfc_ccd.fits.open(msfc_ccd.samples.path_fe55_esis3)
            taps = image.taps

            # Find the isolated events
            hits = taps.hits()

            # Pool them into a histogram for each tap
            hist = na.histogram(
                a=hits.outputs,
                bins={"charge": 26},
                axis=hits.axis_xy,
                min=0 * u.DN,
                max=800 * u.DN,
            )

            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.stairs(
                hist.inputs[{"tap_x": 0, "tap_y": 0}],
                hist.outputs[{"tap_x": 0, "tap_y": 0}],
                axis="charge",
                ax=ax,
                baseline=None,
            )
            ax.set_xlabel("charge in the 3x3 region (DN)")
            ax.set_ylabel("number of events");
        """
        result = self.unbiased.active

        outputs = result.outputs
        outputs = outputs - np.median(outputs, axis=self.axis_xy)

        # The readout noise, which the events themselves do not move
        mad = np.median(np.abs(outputs), axis=self.axis_xy)
        factor = 1.482602218505602
        where_hot = outputs > (threshold * factor * mad)

        # The charge of the event, and the number of hot pixels around it
        size = {self.axis_x: 3, self.axis_y: 3}
        charge = 9 * na.ndfilters.mean_filter(outputs, size=size)
        num_hot = 9 * na.ndfilters.mean_filter(where_hot.astype(float), size=size)

        # Events on the border have an incomplete 3x3 region
        index_x = outputs.indices[self.axis_x]
        index_y = outputs.indices[self.axis_y]
        interior = (0 < index_x) & (index_x < (result.num_x - 1))
        interior = interior & (0 < index_y) & (index_y < (result.num_y - 1))

        where = where_hot & (num_hot < 1.5) & interior

        if threshold_split is not None:
            where_split = outputs > (threshold_split * factor * mad)
            num_split = 9 * na.ndfilters.mean_filter(
                where_split.astype(float),
                size=size,
            )
            where = where & (num_split < 1.5)
            charge = outputs

        return dataclasses.replace(
            result,
            outputs=charge * np.where(where, 1, np.nan),
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
