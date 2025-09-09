from typing_extensions import Self
import abc
import dataclasses
import named_arrays as na
from .._cameras import AbstractCamera
from ._vectors import ImageHeader

__all__ = [
    "AbstractImageData",
    "AbstractCameraData",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImageData(
    na.FunctionArray[
        ImageHeader,
        na.AbstractScalarArray,
    ],
):
    """An interface for image-like data."""

    @property
    @abc.abstractmethod
    def axis_x(self) -> str:
        """The name of the horizontal axis."""

    @property
    @abc.abstractmethod
    def axis_y(self) -> str:
        """The name of vertical axis."""

    @property
    def axis_xy(self):
        """A :class:`tuple` containing :attr:`axis_x` and :attr:`axis_y`."""
        return (self.axis_x, self.axis_y)

    @property
    def num_x(self) -> int:
        """The number of pixels along the horizontal axis."""
        return self.outputs.shape[self.axis_x]

    @property
    def num_y(self) -> int:
        """The number of pixels along the vertical axis."""
        return self.outputs.shape[self.axis_y]


class AbstractCameraData(
    AbstractImageData,
):
    """
    An interface for image data taken with a particular camera.

    This class is useful for a single interface for data from the entire image
    sensor and for data from only a single tap on the sensor.
    """

    @property
    @abc.abstractmethod
    def camera(self) -> AbstractCamera:
        """A model of the camera used to capture these images."""

    @property
    @abc.abstractmethod
    def unbiased(self) -> Self:
        """A new copy of these images where the bias has been removed."""

    @property
    @abc.abstractmethod
    def active(self) -> Self:
        """A new copy of these images without the bias and overscan columns."""

    @property
    @abc.abstractmethod
    def electrons(self) -> Self:
        """A new copy of these images in units of electrons."""

    @property
    def despiked(self) -> Self:
        """Remove cosmic rays using :func:`astroscrappy.detect_cosmics`."""
        return na.despike(
            array=self,
            axis=(self.axis_x, self.axis_y),
        )
