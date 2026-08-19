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

**Blank and overscan columns measure the bias.**
Each tap reads out 50 blank columns before the light-sensitive pixels and 2
overscan columns after them.
Those columns see no light, so their mean is an estimate of the bias for that
tap.
:attr:`~msfc_ccd.abc.AbstractTapData.active` trims them away,
:attr:`~msfc_ccd.abc.AbstractTapData.unbiased` subtracts the bias they measure,
and the two compose: ``image.taps.unbiased.active``.

**Converting to electrons needs a gain.**
:attr:`~msfc_ccd.abc.AbstractSensorData.electrons` multiplies by
:attr:`msfc_ccd.Camera.gain`, which is usually different for each tap and has
to be measured.
A :class:`msfc_ccd.Camera` constructed without one, which is what
:func:`msfc_ccd.fits.open` uses by default, has no gain to apply, and raises a
:class:`ValueError` naming the missing parameter rather than guessing.
Supply your measured value with ``msfc_ccd.Camera(gain=...)`` before
converting, remembering that the gain differs from tap to tap.


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
    taps.bias().outputs

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
