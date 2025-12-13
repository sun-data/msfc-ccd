from typing import ClassVar, Literal
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "TeledyneCCD230",
]


@dataclasses.dataclass(repr=False)
class AbstractSensor(
    optika.mixins.Printable,
):
    """An interface for an imaging sensor or an ensemble of imaging sensors."""

    num_tap_x: ClassVar[int] = 2
    """The number of taps along the long axis of the CCD sensor."""

    num_tap_y: ClassVar[int] = 2
    """The number of taps along the short axis of the CCD sensor."""

    @property
    @abc.abstractmethod
    def manufacturer(self) -> str:
        """The company which produced the sensor."""

    @property
    @abc.abstractmethod
    def family(self) -> str:
        """The model number or product family of this sensor."""

    @property
    @abc.abstractmethod
    def serial_number(self) -> None | str:
        """A unique number which identifies this sensor."""

    @property
    @abc.abstractmethod
    def material(self) -> optika.sensors.materials.AbstractSiliconSensorMaterial:
        """The light-sensitive material used by this sensor."""

    @property
    @abc.abstractmethod
    def num_pixel(self) -> na.Cartesian2dVectorArray[int, int]:
        """The number of pixels along the horizontal and vertical axes."""

    @property
    @abc.abstractmethod
    def num_pixel_active(self):
        """The number of pixels that are used to detect light."""

    @property
    @abc.abstractmethod
    def width_pixel(self) -> u.Quantity:
        """The physical size of a single pixel on the imaging sensor."""

    @property
    def width_active(self):
        """The physical size of the light sensitive area of the sensor."""
        result = self.width_pixel * self.num_pixel_active
        return result.to(u.mm)

    @property
    @abc.abstractmethod
    def num_blank(self) -> int:
        """The number of blank columns at the start of each row."""

    @property
    @abc.abstractmethod
    def num_overscan(self) -> int:
        """The number of overscan columns at the end of each row."""

    @property
    @abc.abstractmethod
    def cte(self) -> u.Quantity:
        """The charge transfer efficiency of the sensor."""

    @property
    @abc.abstractmethod
    def readout_noise(self) -> u.Quantity:
        """The standard deviation of the error on each pixel value."""

    @property
    def temperature(self):
        """The operating temperature of this sensor."""
        return self.material.temperature

    def dark_current(
        self,
        temperature: None | u.Quantity | na.AbstractScalar = None,
    ):
        """
        Calculate the rate of charge accumulation when the sensor is not illuminated.

        Parameters
        ----------
        temperature
            The temperature of the sensor.
            If :obj:`None`, the value of :attr:`temperature` is used.
        """


@dataclasses.dataclass(repr=False)
class TeledyneCCD230(
    AbstractSensor,
):
    """The standard sensor used by the MSFC cameras."""

    manufacturer: str = "Teledyne/e2v"
    """The company which produced the sensor."""

    family: str = "CCD230-42"
    """The model number or product family of this sensor."""

    serial_number: None | str = None
    """A unique number which identifies this sensor."""

    grade: None | str = None
    """
    The quality of the device.

    Grade 0 is the best possible and Grade 5 is the worst possible.
    """

    material: None | optika.sensors.materials.AbstractSiliconSensorMaterial = None
    """
    The light-sensitive material used by this sensor.
    
    If :obj:`None`, :func:`optika.sensors.materials.e2v_ccd97` will be used.
    """

    width_pixel: u.Quantity = 15 * u.um
    """The physical size of a single pixel on the imaging sensor."""

    num_pixel_x: int = 2048
    """The number of pixels along the horizontal axis of the CCD sensor."""

    num_pixel_y: int = 2064
    """The number of pixels along the vertical axis of the CCD sensor."""

    num_blank: int = 50
    """The number of blank columns at the start of each row."""

    num_overscan: int = 2
    """The number of overscan columns at the end of each row."""

    cte: u.Quantity = 99.9995 * u.percent
    """The charge transfer efficiency of the sensor."""

    readout_noise: u.Quantity = 4 * u.electron
    """The standard deviation of the error on each pixel value."""

    readout_mode: Literal["full-frame", "transfer"] = "transfer"
    """
    The frame readout mode of the sensor.
    
    Either the entire sensor is read at the same time (``"full-frame"``),
    or half of the sensor is used for storage (``"transfer"``).
    """

    width_package_x: u.Quantity = 42 * u.mm
    """The horizontal size of the physical sensor package."""

    width_package_y: u.Quantity = 61 * u.mm
    """The vertical size of the physical sensor package."""

    def __post_init__(self):
        if self.material is None:
            self.material = optika.sensors.materials.e2v_ccd97()

    @property
    def num_pixel(self) -> na.Cartesian2dVectorArray:
        return na.Cartesian2dVectorArray(
            x=self.num_pixel_x,
            y=self.num_pixel_y,
        )

    @property
    def num_pixel_active(self):
        """
        The number of pixels that are used to detect light.

        If :attr:`readout_mode` is ``"full-frame"``, then this is the same
        as :attr:`num_pixel`.
        If :attr:`readout_mode` is ``"transfer"``, then the vertical component
        of :attr:`num_pixel` is divided by 2 since half of the sensor is now used
        for charge storage.
        """
        result = self.num_pixel
        if self.readout_mode == "transfer":
            return result.replace(y=result.y // 2)

    @property
    def width_package(self) -> na.Cartesian2dVectorArray:
        """The vertical and horizontal width of the physical sensor package."""
        return na.Cartesian2dVectorArray(
            x=self.width_package_x,
            y=self.width_package_y,
        )

    @classmethod
    def _frac_Qd_Qdo(cls, temperature: u.Quantity | na.AbstractScalar):
        T = temperature
        return 122 * T * np.square(T) * np.exp(-6400 * u.K / T) / u.K**3

    def dark_current(
        self,
        temperature: None | u.Quantity | na.AbstractScalar = None,
    ):
        """
        Calculate the rate of charge accumulation when the sensor is not illuminated.

        Parameters
        ----------
        temperature
            The temperature of the sensor.
            If :obj:`None`, the value of :attr:`temperature` is used.


        .. nblinkgallery::
            :caption: Examples
            :name: rst-link-gallery

            ../reports/dark-current
        """
        if temperature is None:
            temperature = self.temperature
        Q_248K = 0.2 * u.electron / u.s
        f = self._frac_Qd_Qdo(248 * u.K)
        Q_do = Q_248K / f
        return Q_do * self._frac_Qd_Qdo(temperature)
