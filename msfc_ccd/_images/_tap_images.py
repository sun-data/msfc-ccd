from typing_extensions import Self
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
from .._cameras import AbstractCamera
from .._gain import Fe55, _fit_gain
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

    def hits(
        self,
        threshold: float = 5,
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

        return dataclasses.replace(
            result,
            outputs=charge * np.where(where, 1, np.nan),
        )

    def gain(
        self,
        threshold: float = 5,
        gain_min: u.Quantity = 2 * u.electron / u.DN,
        gain_max: u.Quantity = 5 * u.electron / u.DN,
        fe55: None | Fe55 = None,
    ) -> Self:
        r"""
        Measure the gain of each tap from one or more Fe 55 exposures.

        The isolated events found by :meth:`hits` are pooled over every axis
        except the two tap axes, and the gain of each tap is fit to the
        charges of those events.

        The model is a pair of Gaussians, one for each
        :math:`^{55}\text{Fe}` line, whose separation and relative height are
        fixed by the line energies and the emission probabilities, on a flat
        background of events which lost part of their charge to a neighboring
        pixel or to the surface.
        The only free parameters are the gain, the width of the lines and the
        size of that background.
        The fit is an unbinned maximum likelihood, so there is no bin width to
        choose, which matters because a single image yields only a few tens of
        events per tap.

        Parameters
        ----------
        threshold
            Passed to :meth:`hits`.
        gain_min
            The smallest gain to consider.
            This and `gain_max` bracket where the K-:math:`\alpha` peak can
            lie, which is what lets the peak be found without a starting
            guess.
        gain_max
            The largest gain to consider.
        fe55
            The properties of the :math:`^{55}\text{Fe}` source.
            If :obj:`None`, :class:`msfc_ccd.Fe55` is used.

        Returns
        -------
        A copy of these images where the outputs are the gain of each tap,
        or :obj:`numpy.nan` for a tap with too few events to fit.

        Examples
        --------
        Measure the gain of the ESIS channel 3 camera.

        .. jupyter-execute::

            import msfc_ccd

            image = msfc_ccd.fits.open(msfc_ccd.samples.path_fe55_esis3)

            image.taps.gain().outputs.ndarray
        """
        if fe55 is None:
            fe55 = Fe55()

        hits = self.hits(threshold)

        unit = u.electron / u.DN
        gain_min = gain_min.to_value(unit)
        gain_max = gain_max.to_value(unit)

        charge = hits.outputs
        axes = tuple(charge.shape)
        axes_tap = tuple(a for a in (self.axis_tap_y, self.axis_tap_x) if a in axes)
        axes_pooled = tuple(a for a in axes if a not in axes_tap)

        shape_tap = tuple(charge.shape[a] for a in axes_tap)
        ndarray = charge.ndarray_aligned(axes_tap + axes_pooled)
        ndarray = ndarray.to_value(u.DN).reshape(shape_tap + (-1,))

        result = np.empty(shape_tap)
        for index in np.ndindex(*shape_tap):
            result[index] = _fit_gain(
                charge=ndarray[index],
                gain_min=gain_min,
                gain_max=gain_max,
                fe55=fe55,
            )

        return dataclasses.replace(
            self,
            inputs=self.inputs[{a: 0 for a in axes_pooled}],
            outputs=na.ScalarArray(result * unit, axes=axes_tap),
        )

    def dark_current(
        self,
        axis: str,
        proportion: float = 0.01,
    ) -> Self:
        """
        Estimate the dark current rate of each tap.

        The images must be a sequence of darks gathered using a range of
        exposure lengths.
        The signal in each image is the trimmed mean of the bias-subtracted
        active pixels, and the rate is the slope of a linear fit of that signal
        against the measured exposure time of each image,
        :attr:`msfc_ccd.ImageHeader.timedelta`.

        The intercept of the fit is discarded.
        The blank columns used to compute the bias are offset from the dark
        level of the active pixels by about 1 DN,
        and the fixed pattern of the sensor is not removed,
        so the intercept is not the dark current at zero exposure.
        Only the slope is independent of both.

        The dark current accumulated during a single image is much smaller
        than the readout noise, about 0.07 DN in a two-second image,
        so it is only measurable after averaging over all the active pixels,
        and a long exposure is needed to separate it from the bias.

        Parameters
        ----------
        axis
            The logical axis along which the images are a sequence of darks.
        proportion
            The fraction of the brightest and darkest active pixels to remove
            from each image before taking the mean.

            The default removes the cosmic rays and leaves everything else.
            Cosmic rays accumulate in proportion to the exposure time,
            so without any trimming they enter the slope directly and the
            signal grows faster than the dark current does.

            Trimming much harder than the default removes the hot pixels too,
            which changes what is being measured.
            Each pixel has its own dark current rate,
            so as the exposure time grows the brightest pixels are
            increasingly the ones with the highest rates,
            and a heavy trim follows a progressively cooler subset of the
            sensor.
            The result is then the rate of a typical pixel rather than the
            mean over the sensor, and it no longer grows linearly with the
            exposure time, which makes the slope of the fit depend on which
            exposures were used.
            Measured on the dark tests of 2017-07-12, the slope between 4 and
            12 seconds is 1.44 times the slope between 2 and 4 seconds with no
            trimming, 1.01 times at the default, and 0.66 times at a
            proportion of 0.25.

        Examples
        --------
        Estimate the dark current rate of the ESIS channel 1 camera using
        a pair of dark images with different exposure lengths.

        .. jupyter-execute::

            import numpy as np
            import named_arrays as na
            import msfc_ccd

            # Load a two-second dark image and a twelve-second dark image
            path = na.ScalarArray(
                ndarray=np.array([
                    msfc_ccd.samples.path_dark_2s_esis1,
                    msfc_ccd.samples.path_dark_12s_esis1,
                ]),
                axes="time",
            )
            images = msfc_ccd.fits.open(path)

            # Estimate the dark current rate of each tap
            images.taps.dark_current("time").outputs.to("DN / s").ndarray

        Two images give only a rough estimate, since the uncertainty in the
        bias of each image is about a tenth of the signal accumulated between
        them.
        Averaging many images at each of five exposure lengths gives
        0.038 to 0.042 DN / s across the four taps of this camera.
        """
        signal = na.mean_trimmed(
            a=self.unbiased.active.outputs,
            q=proportion,
            axis=self.axis_xy,
        )
        timedelta = self.inputs.timedelta

        signal = signal - signal.mean(axis)
        timedelta = timedelta - timedelta.mean(axis)

        rate = (timedelta * signal).sum(axis) / np.square(timedelta).sum(axis)

        return dataclasses.replace(
            self,
            inputs=self.inputs[{axis: 0}],
            outputs=rate,
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
