import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
from ._sensors import AbstractSensor
import optika
import msfc_ccd

__all__ = [
    "AbstractCamera",
    "Camera",
]


@dataclasses.dataclass(repr=False)
class AbstractCamera(
    optika.mixins.Printable,
):
    """An interface describing a generalized camera."""

    @property
    @abc.abstractmethod
    def sensor(self) -> AbstractSensor:
        """A model of the sensor used by this camera to capture light."""

    @property
    @abc.abstractmethod
    def gain(self) -> u.Quantity | na.AbstractScalar:
        """The conversion factor between electrons and ADC counts."""

    @property
    @abc.abstractmethod
    def axis_tap_x(self) -> str:
        """The name of the logical axis corresponding to changing horizontal tap."""

    @property
    @abc.abstractmethod
    def axis_tap_y(self) -> str:
        """The name of the logical axis corresponding to changing vertical tap."""

    @property
    @abc.abstractmethod
    def timedelta_exposure(self) -> u.Quantity | na.AbstractScalar:
        """The current exposure length."""

    @classmethod
    def calibrate_timedelta_exposure(cls, value: int) -> u.Quantity:
        """
        Convert the exposure time from counts to physical units.

        Parameters
        ----------
        value
            The exposure time in counts.
        """
        return value * 0.000000025 * u.s

    @classmethod
    def calibrate_voltage_fpga(cls, value: int) -> u.Quantity:
        """
        Convert the FPGA voltage from counts to physical units.

        Parameters
        ----------
        value
            The FPGA voltage in counts.
        """
        return value * 3 / 4096 * u.V

    @classmethod
    def calibrate_temperature_fpga(cls, value: int) -> u.Quantity:
        """
        Convert the FPGA temperature from counts to physical units.

        Parameters
        ----------
        value
            The FPGA temperature in counts.
        """
        result = value * 503.975 / 4096 * u.K
        result = result.to(u.deg_C, equivalencies=u.temperature())
        return result

    @classmethod
    def calibrate_temperature_adc_1(cls, value: int) -> u.Quantity:
        """
        Convert the ADC 1 temperature from counts to physical units.

        Parameters
        ----------
        value
            The ADC 1 temperature in counts.
        """
        r = (9.814453125 * value) / (1 - (value / 4096.0))
        result = 3455.0 / np.log(r / 0.0927557) * u.K
        result = result.to(u.deg_C, equivalencies=u.temperature())
        return result

    @classmethod
    def calibrate_temperature_adc_234(cls, value: int) -> u.Quantity:
        """
        Convert the ADC 2, 3, or 4 temperature from counts to physical units.

        Parameters
        ----------
        value
            The ADC 2, 3, or 4 temperature in counts.
        """
        r = (9.814453125 * value) / (1 - (value / 4096.0))
        a = 0.0011275
        b = 0.00023441
        c = 0.000000086482
        result = 1 / (a + (b * np.log(r)) + (c * np.log(r)) ** 3) * u.K
        result = result.to(u.deg_C, equivalencies=u.temperature())
        return result

    def dn_to_electrons(
        self,
        a: u.Quantity | na.AbstractArray,
    ) -> na.AbstractArray:
        """Convert an array from DN to electrons by multiplying by :attr:`gain`."""
        return self.gain * a


@dataclasses.dataclass(repr=False)
class Camera(
    AbstractCamera,
):
    """
    A model of the cameras developed by the MSFC sounding rocket team.

    This is a composition of a :class:`msfc_ccd.abc.AbstractSensor` object
    and various parameters such as the exposure time etc.
    Also provided are some conversion equations between counts and
    physical units for various parameters such as the FPGA temperature, etc.
    """

    sensor: None | AbstractSensor = None
    """
    A model of the sensor used by this camera to capture light.

    If :obj:`None` (the default), :class:`TeledyneCCD230` will be used.
    """

    gain: None | u.Quantity | na.AbstractScalar = None
    """
    The conversion factor between electrons and DN.
    
    This is usually tap-dependent and contains :attr:`axis_tap_x` and
    :attr:`axis_tap_y` dimensions.
    """

    bits_adc: int = 16
    """The number of bits supported by the analog-to-digital converter"""

    timedelta_exposure: u.Quantity = 10 * u.s
    """The current exposure length."""

    timedelta_exposure_min: u.Quantity = 2 * u.s
    """The minimum exposure length supported by this camera."""

    timedelta_exposure_max: u.Quantity = 600 * u.s
    """The maximum exposure length supported by this camera"""

    timedelta_exposure_step: u.Quantity = 100 * u.ms
    """The smallest possible change in exposure length supported by this camera."""

    timedelta_transfer: u.Quantity = 50 * u.ms
    """The time required to transfer the exposed pixels into the storage region."""

    timedelta_readout: u.Quantity = 1.1 * u.s
    """The time required to perform a readout operation."""

    axis_tap_x: str = dataclasses.field(default="tap_x", kw_only=True)
    """The name of the logical axis corresponding to changing horizontal tap."""

    axis_tap_y: str = dataclasses.field(default="tap_y", kw_only=True)
    """The name of the logical axis corresponding to changing vertical tap."""

    def __post_init__(self):
        if self.sensor is None:
            self.sensor = msfc_ccd.TeledyneCCD230()
