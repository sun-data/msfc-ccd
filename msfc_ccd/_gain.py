"""Support for measuring the gain of a camera from an Fe 55 exposure."""

import dataclasses
import warnings
import numpy as np
import scipy.optimize
import scipy.stats
import astropy.units as u
import optika

__all__ = [
    "Fe55",
]


@dataclasses.dataclass(frozen=True)
class Fe55:
    r"""
    The X-rays emitted by an :math:`^{55}\text{Fe}` source.

    An :math:`^{55}\text{Fe}` nucleus captures one of its own electrons and
    becomes :math:`^{55}\text{Mn}`, which emits an X-ray at one of two
    energies as the resulting hole is filled.
    Both energies are known to much better than the resolution of these
    sensors, so the charge that an X-ray releases is a ruler which can be
    used to measure the gain.

    How much charge that is depends on the temperature of the sensor,
    through the energy needed to create one electron-hole pair, so
    :attr:`temperature` has to be set to the temperature the sensor was at.

    Examples
    --------
    The number of electrons released by each line.

    .. jupyter-execute::

        import msfc_ccd

        fe55 = msfc_ccd.Fe55()

        print(f"K-alpha: {fe55.energy_k_alpha:0.4f} -> {fe55.charge_k_alpha:0.1f}")
        print(f"K-beta:  {fe55.energy_k_beta:0.4f} -> {fe55.charge_k_beta:0.1f}")

    Warming the sensor to room temperature would change the answer by
    almost a percent.

    .. jupyter-execute::

        import astropy.units as u

        warm = msfc_ccd.Fe55(temperature=300 * u.K)

        print(f"{warm.charge_k_alpha:0.1f} at {warm.temperature}")
    """

    probability_k_alpha_1: float = 0.162
    r"""The probability of emitting a K-:math:`\alpha_1` X-ray."""

    probability_k_alpha_2: float = 0.082
    r"""The probability of emitting a K-:math:`\alpha_2` X-ray."""

    probability_k_beta: float = 0.0285
    r"""The probability of emitting a K-:math:`\beta` X-ray."""

    energy_k_alpha_1: u.Quantity = 5.89875 * u.keV
    r"""The energy of the K-:math:`\alpha_1` X-rays."""

    energy_k_alpha_2: u.Quantity = 5.88765 * u.keV
    r"""The energy of the K-:math:`\alpha_2` X-rays."""

    energy_k_beta: u.Quantity = 6.49045 * u.keV
    r"""The energy of the K-:math:`\beta` X-rays."""

    temperature: u.Quantity = -55 * u.deg_C
    """
    The temperature of the sensor when the exposure was gathered.

    The default is the temperature the ESIS cameras are designed to run at.
    The energy needed to create one electron-hole pair falls as silicon
    warms, so a sensor at room temperature would release almost a percent
    more charge for the same X-ray.
    """

    @property
    def probability_k_alpha(self) -> float:
        r"""
        The probability of emitting a K-:math:`\alpha` X-ray of either energy.

        The two are 11 eV apart, far closer than these sensors can resolve,
        so they are treated as one line.
        """
        return self.probability_k_alpha_1 + self.probability_k_alpha_2

    @property
    def energy_k_alpha(self) -> u.Quantity:
        r"""The mean energy of the K-:math:`\alpha` X-rays."""
        p_1 = self.probability_k_alpha_1
        p_2 = self.probability_k_alpha_2
        e_1 = self.energy_k_alpha_1
        e_2 = self.energy_k_alpha_2
        return (p_1 * e_1 + p_2 * e_2) / (p_1 + p_2)

    def _charge(self, energy: u.Quantity) -> u.Quantity:
        # `quantum_yield_ideal` divides by zero on its way to the answer,
        # which is harmless but would otherwise warn on every call and show
        # up as a stderr cell in the documentation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = optika.sensors.quantum_yield_ideal(
                wavelength=energy,
                temperature=self.temperature,
            )
        return (result.ndarray * u.ph) << u.electron

    @property
    def charge_k_alpha(self) -> u.Quantity:
        r"""The electrons released in silicon by a K-:math:`\alpha` X-ray."""
        return self._charge(self.energy_k_alpha)

    @property
    def charge_k_beta(self) -> u.Quantity:
        r"""The electrons released in silicon by a K-:math:`\beta` X-ray."""
        return self._charge(self.energy_k_beta)


def _fit_gain(
    charge: np.ndarray,
    gain_min: float,
    gain_max: float,
    fe55: Fe55,
) -> float:
    """
    Fit the gain to an unbinned list of event charges, in DN.

    The likelihood is a pair of Gaussians whose separation and relative
    height are fixed by the Fe 55 line energies and emission probabilities,
    on a flat background of events which lost part of their charge.
    Only the gain, the width of the lines and the size of the background
    are free.
    """
    charge = charge[np.isfinite(charge)]

    q_alpha = fe55.charge_k_alpha.to_value(u.electron)
    q_beta = fe55.charge_k_beta.to_value(u.electron)

    p_beta = fe55.probability_k_beta / (
        fe55.probability_k_alpha + fe55.probability_k_beta
    )
    p_alpha = 1 - p_beta

    # The K-alpha peak can only lie where the gain is physically possible
    allowed = q_alpha / gain_max, q_alpha / gain_min
    charge = charge[(allowed[0] < charge) & (charge < allowed[1])]
    if charge.size < 8:
        return np.nan

    # Locate the peak. The events which lost part of their charge form a
    # background which rises toward low charge, and with only a few tens of
    # events a single bin of it can tie with the peak, so the histogram is
    # smoothed before the largest bin is taken.
    counts, edges = np.histogram(charge, bins=16, range=allowed)
    counts = np.convolve(counts, np.ones(3) / 3, mode="same")
    center = (edges[:-1] + edges[1:])[np.argmax(counts)] / 2

    # Fit only near the peak, which leaves the background nearly flat
    lower, upper = 0.80 * center, 1.30 * center
    charge = charge[(lower < charge) & (charge < upper)]
    if charge.size < 5:  # pragma: nocover
        return np.nan

    # Confining the gain to the window keeps the K-alpha peak inside it,
    # so the normalizations below can never collapse.
    bounds = [
        (max(gain_min, q_alpha / upper), min(gain_max, q_alpha / lower)),
        (1, (upper - lower) / 2),
        (0, 1),
    ]

    def neglog(p: np.ndarray) -> float:
        gain, width, background = p
        center_alpha = q_alpha / gain
        center_beta = q_beta / gain
        cdf = scipy.stats.norm.cdf
        norm_alpha = float(
            cdf(upper, center_alpha, width) - cdf(lower, center_alpha, width)
        )
        norm_beta = float(
            cdf(upper, center_beta, width) - cdf(lower, center_beta, width)
        )
        density = p_alpha * scipy.stats.norm.pdf(charge, center_alpha, width)
        density = density / norm_alpha
        density = (
            density
            + p_beta * scipy.stats.norm.pdf(charge, center_beta, width) / norm_beta
        )
        density = (1 - background) * density + background / (upper - lower)
        return -np.sum(np.log(np.maximum(density, 1e-300)))

    result = scipy.optimize.minimize(
        fun=neglog,
        x0=np.array([q_alpha / center, 15.0, 0.2]),
        method="Nelder-Mead",
        bounds=bounds,
        options=dict(maxiter=4000, xatol=1e-5, fatol=1e-5),
    )
    if not result.success:  # pragma: nocover
        return np.nan

    return result.x[0]
