msfc-ccd
========

The sounding rocket team at Marshall Space Flight Center builds CCD cameras for
solar instruments flown on sounding rockets.
This library reads the FITS files those cameras produce, and models the camera
and sensor well enough to turn a raw readout into a calibrated image.

Every array in this package is a :mod:`named_arrays` array, so the axes carry
names such as ``detector_x`` rather than integer positions, and an image loaded
from disk broadcasts against the other arrays in the stack without reshaping.

Installation
============

:mod:`msfc_ccd` is published on PyPI and can be installed using::

    pip install msfc-ccd


Features
========

*   :func:`msfc_ccd.fits.open`, which loads one FITS file, or a sequence of
    them, into a single object.
*   :class:`msfc_ccd.SensorData`, an image or a sequence of images as read from
    the sensor, along with the operations that calibrate it:
    :attr:`~msfc_ccd.abc.AbstractSensorData.taps`,
    :attr:`~msfc_ccd.abc.AbstractSensorData.active`,
    :attr:`~msfc_ccd.abc.AbstractSensorData.unbiased`, and
    :attr:`~msfc_ccd.abc.AbstractSensorData.electrons`.
*   :class:`msfc_ccd.TapData`, the same image split into the four quadrants
    read out by the four taps of the sensor.
*   :class:`msfc_ccd.ImageHeader`, the FITS metadata for each image, including
    the exposure time, the sensor and FPGA temperatures, and the timestamps.
*   :class:`msfc_ccd.Camera` and :class:`msfc_ccd.TeledyneCCD230`, models of
    the camera and its sensor, which carry the parameters needed to calibrate
    an image: gain, dark current, readout noise, charge transfer efficiency,
    and the exposure timing.
*   :mod:`msfc_ccd.samples`, a handful of real FITS files gathered from the
    cameras, used by the examples throughout this documentation.


Key concepts
============

**An image pairs pixel values with the header that describes them.**
:class:`msfc_ccd.SensorData` stores the pixel values in
:attr:`~msfc_ccd.abc.AbstractImageData.outputs`, in data numbers
(the ``DN`` unit from :mod:`astropy.units`), on axes named by
:attr:`~msfc_ccd.abc.AbstractImageData.axis_x` and
:attr:`~msfc_ccd.abc.AbstractImageData.axis_y`, and the FITS metadata in
:attr:`~msfc_ccd.abc.AbstractImageData.inputs` as an
:class:`msfc_ccd.ImageHeader`.
Loading many files at once adds another named axis to both, so a sequence of
images is the same type as a single image.

**The sensor is read out through four taps.**
Each quadrant of the CCD has its own amplifier, so a raw frame is really four
images side by side, each with its own bias and gain.
:attr:`~msfc_ccd.abc.AbstractSensorData.taps` splits a frame into a
:class:`msfc_ccd.TapData` with ``tap_x`` and ``tap_y`` axes, and
:meth:`~msfc_ccd.SensorData.from_taps` reassembles one, flipping each quadrant
back into the orientation of the sensor.

**Blank columns measure the bias.**
Each tap reads out 50 blank columns before the light-sensitive pixels and 2
overscan columns after them.
The blank columns are read before any charge from the image, so the mean of
the ones closest to the image is an estimate of the bias for that tap.
The overscan columns are read after the image, and pick up charge deferred
from its last column, so they are not used.
:attr:`~msfc_ccd.abc.AbstractTapData.active` trims them away,
:attr:`~msfc_ccd.abc.AbstractTapData.unbiased` subtracts the bias they measure,
and the two compose: ``image.taps.unbiased.active``.

**Readout noise comes from a sequence of darks.**
Differencing two adjacent dark images cancels the bias, the dark current and
the fixed pattern of the sensor, leaving only the readout noise of the two
frames, so :meth:`~msfc_ccd.abc.AbstractTapData.readout_noise` estimates it
from the active pixels of each difference rather than from the columns at the
edge of a single frame.

**Dark current needs a range of exposure lengths.**
A two-second dark accumulates less than a tenth of a data number of dark
current, far below the readout noise, so
:meth:`~msfc_ccd.abc.AbstractTapData.dark_current` averages over all the
active pixels of each image and fits the result against the measured exposure
time of a set of darks taken at several exposure lengths.
Only the slope of that fit is the dark current; the intercept also contains
the offset between the blank columns and the active pixels, and the fixed
pattern of the sensor.

**Fe 55 events are isolated single-pixel hits.**
A 5.9 keV X-ray from an :math:`^{55}\text{Fe}` source releases a known number
of electrons in one pixel, so the charge it leaves is a ruler for the gain.
:meth:`~msfc_ccd.abc.AbstractTapData.hits` finds every pixel far enough above
the dark level whose eight neighbors are not, and measures the charge of the
event as the sum of the surrounding three by three region, which recovers the
charge that spilled into the neighbors.
It returns an image of the same shape with the charge of each event in place
and :obj:`numpy.nan` elsewhere, so a sequence of images is pooled by taking a
histogram along the sequence axis and the two detector axes.

**Converting to electrons needs a gain.**
:attr:`~msfc_ccd.abc.AbstractSensorData.electrons` multiplies by
:attr:`msfc_ccd.Camera.gain`, which is usually different for each tap and has
to be measured.
A :class:`msfc_ccd.Camera` constructed without one, which is what
:func:`msfc_ccd.fits.open` uses by default, has no gain to apply, and raises a
:class:`ValueError` naming the missing parameter rather than guessing.
:meth:`~msfc_ccd.abc.AbstractTapData.gain` measures it from an
:math:`^{55}\text{Fe}` exposure, and the result goes straight into
``msfc_ccd.Camera(gain=...)``.


Examples
========
Load and display a single FITS file.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import named_arrays as na
    import msfc_ccd

    # Load the sample image
    image = msfc_ccd.fits.open(msfc_ccd.samples.path_fe55_esis1)

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

|

Measure the bias of each tap, and remove it.

.. jupyter-execute::

    # Split the frame into the four tap images
    taps = image.taps

    # Each tap has its own amplifier, and so its own bias
    taps.bias().outputs.ndarray

.. jupyter-execute::

    # Subtract the bias and trim the blank and overscan columns
    corrected = image.from_taps(taps.unbiased.active)

    # Display the corrected image
    fig, ax = plt.subplots(
        constrained_layout=True,
    )
    im = na.plt.imshow(
        corrected.outputs.value,
        axis_x=corrected.axis_x,
        axis_y=corrected.axis_y,
        ax=ax,
        vmin=-20,
        vmax=100,
    );

|

Measure the readout noise of each tap from a pair of adjacent dark images.

.. jupyter-execute::

    import numpy as np

    # Define the name of the time axis
    axis_time = "time"

    # Load two adjacent dark images as a sequence
    darks = msfc_ccd.fits.open(
        path=na.ScalarArray(
            ndarray=np.array([
                msfc_ccd.samples.path_led_dark_esis1,
                msfc_ccd.samples.path_led_dark_esis1_next,
            ]),
            axes=axis_time,
        ),
    )

    # The difference of adjacent frames leaves only the readout noise
    darks.taps.readout_noise(axis_time).outputs.ndarray

|

Measure the dark current rate of each tap from a pair of dark images with
different exposure lengths.

.. jupyter-execute::

    # Load a two-second dark image and a twelve-second dark image
    darks = msfc_ccd.fits.open(
        path=na.ScalarArray(
            ndarray=np.array([
                msfc_ccd.samples.path_dark_2s_esis1,
                msfc_ccd.samples.path_dark_12s_esis1,
            ]),
            axes=axis_time,
        ),
    )

    # The slope of the signal against the exposure time is the dark current
    darks.taps.dark_current(axis_time).outputs.to("DN / s").ndarray

|

Measure the gain of each tap from an Fe 55 exposure, and use it to convert an
image into electrons.

.. jupyter-execute::

    # Load an Fe 55 exposure and measure the gain of each tap
    fe55 = msfc_ccd.fits.open(msfc_ccd.samples.path_fe55_esis3)
    gain = fe55.taps.gain().outputs

    gain.ndarray

.. jupyter-execute::

    # Build a camera with that gain, and reload the image through it
    camera = msfc_ccd.Camera(gain=gain)
    calibrated = msfc_ccd.fits.open(
        path=msfc_ccd.samples.path_fe55_esis3,
        camera=camera,
    )

    calibrated.taps.unbiased.active.electrons.outputs.sum().ndarray

|

Inspect the header of an image.

.. jupyter-execute::

    header = image.inputs

    print(f"serial number:  {header.serial_number.ndarray}")
    print(f"exposure:       {header.timedelta.ndarray}")
    print(f"start:          {header.time_start.ndarray}")
    print(f"FPGA temp:      {header.temperature_fpga.ndarray}")

|

Inspect the sensor model that the calibration steps rely on.

.. jupyter-execute::

    import astropy.units as u

    sensor = image.camera.sensor

    print(f"sensor:          {sensor.manufacturer} {sensor.family}")
    print(f"active pixels:   {sensor.num_pixel_x} x {sensor.num_pixel_y}")
    print(f"blank/overscan:  {sensor.num_blank} / {sensor.num_overscan}")
    print(f"readout noise:   {sensor.readout_noise}")
    print(f"charge transfer: {sensor.cte}")
    print(f"dark current:    {sensor.dark_current(263 * u.K):0.3f} at 263 K")

|


API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    msfc_ccd

|


Reports
=======

Jupyter notebook investigations which help to characterize the CCDs and
justify the decisions made in this package.

.. toctree::
    :maxdepth: 1

    reports/bias
    reports/dark-current
    reports/gain

|


References
==========

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
