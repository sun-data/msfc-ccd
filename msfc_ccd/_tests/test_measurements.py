from typing import Callable
import pytest
import msfc_ccd
from . import _shared


@pytest.mark.parametrize(
    argnames="function,args",
    argvalues=[
        (msfc_ccd.noise.readout, ("time",)),
        (msfc_ccd.noise.photon_transfer, ("time",)),
        (msfc_ccd.dark.current, ("time",)),
        (msfc_ccd.gain.fe55, ()),
        (msfc_ccd.gain.photon_transfer, ("time",)),
        (msfc_ccd.cte.fe55, ()),
        (msfc_ccd.cte.eper, ()),
    ],
)
def test_requires_taps(function: Callable, args: tuple):
    """A measurement of each tap refuses an image of the whole sensor."""
    with pytest.raises(TypeError, match="images.taps"):
        function(_shared.dark, *args)
