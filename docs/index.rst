Introduction
============

This library aims to provide a set of simple utilities to determine the gain and
bias of CCD cameras developed by Marshall Space Flight Center for use in
spaceflight.

|

Examples
--------
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
