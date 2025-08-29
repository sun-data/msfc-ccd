"""Utilities for working with FITS files."""

import pathlib
import named_arrays as na
import msfc_ccd


__all__ = [
    "open",
]


def open(
    path: str | pathlib.Path | na.AbstractScalarArray,
    camera: None | msfc_ccd.abc.AbstractCamera = None,
    axis_x: str = "detector_x",
    axis_y: str = "detector_y",
) -> msfc_ccd.SensorData:
    """
    Load the given FITS images into memory.

    This is a convenience function for :meth:`msfc_ccd.SensorData.from_fits`.

    Parameters
    ----------
    path
        Either a single path or an array of paths pointing to the FITS files
        to open.
    camera
        A model of the camera used to capture the images being loaded.
        If :obj:`None` (the default), the :class:`msfc_ccd.Camera`
        will be used.
    axis_x
        The name of the logical axis representing the horizontal dimension of
        the images.
    axis_y
        The name of the logical axis representing the vertical dimension of
        the images.

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

    Load and display an array of two FITS files.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na
        import msfc_ccd

        # Define the name of the time axis.
        axis_time = "time"

        # Define the array of two sample FITS files
        path = na.ScalarArray(
            ndarray=np.array([
                msfc_ccd.samples.path_fe55_esis1,
                msfc_ccd.samples.path_fe55_esis3,
            ]),
            axes=axis_time,
        )

        # Load the sample images
        image = msfc_ccd.fits.open(path)

        # Display the sample images
        fig, axs = na.plt.subplots(
            axis_rows=axis_time,
            nrows=image.outputs.shape[axis_time],
            sharex=True,
            constrained_layout=True,
        )
        im = na.plt.imshow(
            image.outputs.value,
            axis_x=image.axis_x,
            axis_y=image.axis_y,
            ax=axs,
        );
    """
    if camera is None:
        camera = msfc_ccd.Camera()

    return msfc_ccd.SensorData.from_fits(
        path=path,
        camera=camera,
        axis_x=axis_x,
        axis_y=axis_y,
    )
