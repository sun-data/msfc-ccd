from typing_extensions import Self
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
from .._cameras import AbstractCamera
from .._gain import Fe55, _fit_gain, _fit_cte
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

    def photon_transfer(
        self,
        axis: str,
        threshold: float = 5,
    ) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:
        r"""
        Compute the photon transfer curve of each tap from a sequence of flats.

        The images must be a sequence of uniformly illuminated images, or
        flats, gathered using the same illumination and exposure length.
        For each pair of adjacent images along `axis`,
        the signal is the mean of the two images and the variance is half the
        variance of their difference,
        computed over the active pixels outside the masked rows,
        :meth:`where_masked`.

        Differencing the two images removes everything the two images have in
        common, including the pattern of the illumination and the variation in
        response from pixel to pixel,
        and leaves only the shot noise and the readout noise.
        The shot noise has a variance, in electrons, equal to the signal,
        so the variance in data numbers is

        .. math::

            \sigma^2 = \frac{S}{g} + \sigma_r^2,

        where :math:`S` is the signal, :math:`g` is the gain,
        and :math:`\sigma_r` is the readout noise.
        Gathering pairs at several levels of illumination traces out the
        photon transfer curve, and :meth:`gain_photon_transfer` uses it to
        measure the gain.

        The result has one fewer element along `axis` than the input,
        since it is defined for each pair of adjacent images.

        Parameters
        ----------
        axis
            The logical axis along which the images are a sequence of flats.
        threshold
            Pixels in the difference image further than this many standard
            deviations from the median are rejected, as in
            :meth:`readout_noise`, to remove cosmic rays.

        Examples
        --------
        Compute the signal and the variance of a pair of images of a diffuse
        LED source.

        .. jupyter-execute::

            import numpy as np
            import named_arrays as na
            import msfc_ccd

            # Load two consecutive images with the same illumination
            path = na.ScalarArray(
                ndarray=np.array([
                    msfc_ccd.samples.path_led_esis1,
                    msfc_ccd.samples.path_led_esis1_next,
                ]),
                axes="time",
            )
            images = msfc_ccd.fits.open(path)

            # Compute the signal and the variance of each tap
            ptc = images.taps.photon_transfer("time")

            ptc.inputs.ndarray

        .. jupyter-execute::

            ptc.outputs.ndarray
        """
        num_masked = self.camera.sensor.num_masked
        outputs = self.unbiased.active.outputs
        outputs = outputs[{self.axis_y: slice(num_masked, None)}]
        outputs_1 = outputs[{axis: slice(1, None)}]
        outputs_0 = outputs[{axis: slice(None, -1)}]

        signal = ((outputs_1 + outputs_0) / 2).mean(self.axis_xy)

        variance = self._variance_difference(
            difference=outputs_1 - outputs_0,
            threshold=threshold,
        )

        return na.FunctionArray(
            inputs=signal,
            outputs=variance,
        )

    def gain_photon_transfer(
        self,
        axis: str,
        threshold: float = 5,
    ) -> Self:
        r"""
        Measure the gain of each tap from a sequence of flat images.

        The gain is the ratio of the signal to the shot noise variance
        computed by :meth:`photon_transfer`,

        .. math::

            g = \frac{S}{\sigma^2 - \sigma_r^2},

        pooled over every axis except the two tap axes,
        so pairs of flats at several levels of illumination can be combined.
        The readout noise, :math:`\sigma_r`, is measured from the blank columns
        of the same pairs of images, where there is no shot noise,
        so it follows any change in the noise of the camera during the test.

        This method is independent of :meth:`gain`, which measures the gain
        from the charge released by :math:`^{55}\text{Fe}` X-rays.
        On the tests of 2017-07-12 the two agree to within 3 percent,
        with the flats giving the lower gain on every tap.

        The flats should be well below full well.
        Above about 5,000 DN, the variance of the ESIS cameras grows more
        slowly than the signal, so the apparent gain rises,
        by 1 to 2 percent at 15,000 DN and about 5 percent at 30,000 DN.
        The sample images in the example below are near 15,000 DN.

        Parameters
        ----------
        axis
            The logical axis along which the images are a sequence of flats.
        threshold
            Pixels in the difference images further than this many standard
            deviations from the median are rejected, to remove cosmic rays.

        Examples
        --------
        Measure the gain of the ESIS channel 1 camera from a pair of images of
        a diffuse LED source.

        .. jupyter-execute::

            import numpy as np
            import named_arrays as na
            import msfc_ccd

            # Load two consecutive images with the same illumination
            path = na.ScalarArray(
                ndarray=np.array([
                    msfc_ccd.samples.path_led_esis1,
                    msfc_ccd.samples.path_led_esis1_next,
                ]),
                axes="time",
            )
            images = msfc_ccd.fits.open(path)

            # Measure the gain of each tap
            images.taps.gain_photon_transfer("time").outputs.ndarray
        """
        ptc = self.photon_transfer(axis, threshold)

        outputs = self.outputs
        outputs = outputs[{self.axis_x: self.where_blank(num=25)}]
        outputs_1 = outputs[{axis: slice(1, None)}]
        outputs_0 = outputs[{axis: slice(None, -1)}]
        variance_readout = self._variance_difference(
            difference=outputs_1 - outputs_0,
            threshold=threshold,
        )

        axes_tap = (self.axis_tap_x, self.axis_tap_y)
        axes_pooled = tuple(a for a in ptc.outputs.shape if a not in axes_tap)

        signal = ptc.inputs.sum(axes_pooled)
        variance = (ptc.outputs - variance_readout).sum(axes_pooled)

        return dataclasses.replace(
            self,
            inputs=self.inputs[{a: 0 for a in axes_pooled}],
            outputs=(u.electron * signal / variance).to(u.electron / u.DN),
        )

    def _variance_difference(
        self,
        difference: na.AbstractScalarArray,
        threshold: float,
    ) -> na.AbstractScalarArray:
        """Half the variance of a difference image, rejecting spikes."""
        axis_xy = (self.axis_x, self.axis_y)
        median = np.median(difference, axis=axis_xy)
        deviation = np.abs(difference - median)
        mad = np.median(deviation, axis=axis_xy)

        # The ratio of the standard deviation to the median absolute deviation
        # for a normal distribution
        factor = 1.482602218505602
        where = deviation < threshold * factor * mad

        return np.square(difference.std(axis=axis_xy, where=where)) / 2

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
            This is what :meth:`cte_fe55` needs, since the sum over the
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

        The masked rows at the start of each tap, :meth:`where_masked`,
        are left out of the mean.
        They hold charge from the frame store rather than from the image area,
        and their dark current rate is about 80 times higher,
        so although they are less than two percent of the active pixels,
        including them would roughly double the result.

        The intercept of the fit is discarded.
        The blank columns used to compute the bias are offset from the dark
        level of the active pixels by about 1 DN,
        and the fixed pattern of the sensor is not removed,
        so the intercept is not the dark current at zero exposure.
        Only the slope is independent of both.

        The dark current accumulated during a single image is much smaller
        than the readout noise, about 0.05 DN in a two-second image,
        so it is only measurable after averaging over all the active pixels,
        and a long exposure is needed to separate it from the bias.

        Parameters
        ----------
        axis
            The logical axis along which the images are a sequence of darks.
        proportion
            The fraction of the brightest and darkest active pixels to remove
            from each image before taking the mean,
            which protects the mean from cosmic rays.

            On the dark tests of 2017-07-12, cosmic rays are too rare to
            matter, and the result is the same to within one percent for any
            proportion from 0 to 0.05.
            Trimming much harder removes the hot pixels too,
            which are part of the dark current.

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
        bias of each image is about a fifth of the signal accumulated between
        them.
        Averaging many images at each of five exposure lengths gives
        0.020 to 0.025 DN / s across the four taps of this camera,
        within six percent of the independent analysis of the same test by
        MSFC.
        """
        num_masked = self.camera.sensor.num_masked
        signal = self.unbiased.active.outputs[{self.axis_y: slice(num_masked, None)}]
        signal = na.mean_trimmed(
            a=signal,
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

    def cte_fe55(
        self,
        threshold: float = 5,
        threshold_split: float = 3,
        gain_min: u.Quantity = 2 * u.electron / u.DN,
        gain_max: u.Quantity = 5 * u.electron / u.DN,
        fe55: None | Fe55 = None,
    ) -> Self:
        r"""
        Estimate the charge transfer efficiency of each tap from Fe 55 images.

        Every :math:`^{55}\text{Fe}` K-:math:`\alpha` X-ray releases the same
        charge, but an event far from the amplifier is transferred more times
        before it is read out, and loses a little of its charge to each
        imperfect transfer.
        The single-pixel events found by :meth:`hits` are pooled over every
        axis except the two tap axes, and fit with the model used by
        :meth:`gain`, except that the center of each line falls linearly with
        the number of serial and parallel transfers of the event,

        .. math::

            q = q_0 \left(1 - \text{CTI}_x n_x - \text{CTI}_y n_y \right),

        where :math:`n_x` counts the blank columns as well as the active
        columns between the event and the amplifier,
        and :math:`n_y` counts the rows.

        Only the center pixel of each single-pixel event is used,
        which is the standard form of the method.
        MSFC summed the :math:`3 \times 3` region around each event, which
        recovers the charge left in the neighboring pixel and so hides the
        loss; their fits gave a charge transfer efficiency above 100 percent.

        The result is a vector for each tap,
        whose :math:`x` component is the serial charge transfer efficiency and
        whose :math:`y` component is the parallel charge transfer efficiency.

        The loss is tiny, about a data number across the whole tap,
        so this method needs thousands of events per tap,
        or several hundred images from the ESIS cameras.

        Subtract a master dark from the images first, using
        :meth:`~msfc_ccd.abc.AbstractCameraData.dark`.
        The dark level of the ESIS cameras rises by 0.3 to 0.4 DN from the
        first row read out to the last, which the median removed by
        :meth:`hits` cannot follow, and which would otherwise look like a
        negative parallel inefficiency of about :math:`10^{-6}`.

        The fit also averages over any columns with charge traps.
        On a sensor with many of them, the traps dominate the result,
        and the serial component should be taken from :meth:`cte_eper` instead.

        Parameters
        ----------
        threshold
            How many readout noises above the dark level a pixel must be to
            start an event.
        threshold_split
            How many readout noises above the dark level each of the eight
            neighbors of an event must stay below for it to count as a
            single-pixel event.
        gain_min
            The smallest gain allowed, which sets where the K-:math:`\alpha`
            peak can be.
        gain_max
            The largest gain allowed.
        fe55
            The properties of the :math:`^{55}\text{Fe}` source.
            If :obj:`None`, the default :class:`msfc_ccd.Fe55` is used.

        Examples
        --------
        Estimate the charge transfer efficiency of the ESIS channel 3 camera
        from four Fe 55 exposures.
        Four images are far too few for a precise result,
        so this only shows how the method is called.

        .. jupyter-execute::

            import numpy as np
            import named_arrays as na
            import msfc_ccd

            path = na.ScalarArray(
                ndarray=np.array(msfc_ccd.samples.paths_fe55_esis3),
                axes="time",
            )
            images = msfc_ccd.fits.open(path)

            cte = images.taps.cte_fe55().outputs

            # The serial charge transfer efficiency
            cte.x.ndarray

        .. jupyter-execute::

            # The parallel charge transfer efficiency
            cte.y.ndarray
        """
        if fe55 is None:
            fe55 = Fe55()

        hits = self.hits(threshold, threshold_split)

        unit = u.electron / u.DN
        gain_min = gain_min.to_value(unit)
        gain_max = gain_max.to_value(unit)

        # The masked rows are not part of the image
        charge = hits.outputs
        index_x = charge.indices[self.axis_x]
        index_y = charge.indices[self.axis_y]
        where_image = index_y >= self.camera.sensor.num_masked
        charge = charge * np.where(where_image, 1, np.nan)

        # The number of transfers before each pixel reaches the amplifier
        transfers_x = index_x + self.camera.sensor.num_blank + 1
        transfers_y = index_y + 1

        shape = charge.shape
        axes_tap = tuple(a for a in (self.axis_tap_y, self.axis_tap_x) if a in shape)
        axes_pooled = tuple(a for a in shape if a not in axes_tap)
        shape_tap = tuple(shape[a] for a in axes_tap)

        def flatten(a: na.AbstractScalar) -> np.ndarray:
            a = na.broadcast_to(a, shape)
            a = a.ndarray_aligned(axes_tap + axes_pooled)
            return np.asarray(a).reshape(shape_tap + (-1,))

        charge = flatten(charge.to(u.DN).value)
        transfers_x = flatten(transfers_x)
        transfers_y = flatten(transfers_y)

        cti_x = np.empty(shape_tap)
        cti_y = np.empty(shape_tap)
        for index in np.ndindex(*shape_tap):
            cti_x[index], cti_y[index] = _fit_cte(
                charge=charge[index],
                transfers_x=transfers_x[index],
                transfers_y=transfers_y[index],
                gain_min=gain_min,
                gain_max=gain_max,
                fe55=fe55,
            )

        return dataclasses.replace(
            self,
            inputs=self.inputs[{a: 0 for a in axes_pooled}],
            outputs=na.Cartesian2dVectorArray(
                x=na.ScalarArray((1 - cti_x) * 100 * u.percent, axes=axes_tap),
                y=na.ScalarArray((1 - cti_y) * 100 * u.percent, axes=axes_tap),
            ),
        )

    def cte_eper(
        self,
        num_active: int = 10,
    ) -> Self:
        r"""
        Estimate the serial charge transfer efficiency of each tap from a flat.

        This is the extended pixel edge response (EPER) method.
        Each serial transfer leaves a small fraction of the charge behind,
        which is released into the pixels read out afterwards.
        In a flat, the charge left behind by the last active pixel of each row
        spills into the overscan columns, so the charge transfer inefficiency
        is

        .. math::

            \text{CTI} = \frac{Q_o}{S N},

        where :math:`Q_o` is the total charge in the overscan columns of a
        row, :math:`S` is the signal in the last active pixels of that row,
        and :math:`N` is the number of serial transfers of the last active
        pixel.
        The result is the charge transfer efficiency,
        :math:`\text{CTE} = 1 - \text{CTI}`, for each image.

        :math:`N` counts the blank columns as well as the active columns,
        since the blank columns are elements of the serial register between the
        active pixels and the amplifier.
        MSFC counted only the active columns, which makes their inefficiency
        about 5 percent larger for the same data.

        The masked rows, :meth:`where_masked`, are ignored.
        Only the serial transfers are measured,
        since the sensor is read out without any overscan rows.

        The charge left behind is not a fixed fraction of the signal.
        On the tests of 2017-07-12, the fraction is nearly constant on some
        taps and falls with signal on others, and on every tap it rises below
        about a thousand data numbers,
        so the result describes the camera at the signal level of the flat.
        The overscan columns of a dark image sit within about 0.4 DN of the
        bias, which matters only for faint flats;
        subtract a dark image first to remove it.

        Parameters
        ----------
        num_active
            The number of active columns at the end of each row used to
            measure the signal.

        Examples
        --------
        Estimate the serial charge transfer efficiency of the ESIS channel 1
        camera from an image of a diffuse LED source.

        .. jupyter-execute::

            import msfc_ccd

            image = msfc_ccd.fits.open(msfc_ccd.samples.path_led_esis1)

            image.taps.cte_eper().outputs.ndarray
        """
        sensor = self.camera.sensor

        outputs = self.unbiased.outputs
        outputs = outputs[{self.axis_y: slice(sensor.num_masked, None)}]

        num_transfers = self.num_x - sensor.num_overscan
        slice_active = slice(num_transfers - num_active, num_transfers)
        slice_overscan = slice(num_transfers, None)

        signal = outputs[{self.axis_x: slice_active}].mean(self.axis_xy)
        deferred = outputs[{self.axis_x: slice_overscan}].sum(self.axis_x)
        deferred = deferred.mean(self.axis_y)

        cti = deferred / (signal * num_transfers)

        return dataclasses.replace(
            self,
            outputs=(1 - cti).to(u.percent),
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
