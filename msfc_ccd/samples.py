"""Sample FITS files gathered from the MSFC cameras."""

import pathlib

__all__ = [
    "path_fe55_esis1",
    "path_fe55_esis3",
    "paths_fe55_esis3",
    "path_dark_esis1",
    "path_dark_esis3",
    "path_led_esis1",
    "path_led_esis1_next",
    "path_led_dark_esis1",
    "path_led_dark_esis1_next",
    "path_dark_2s_esis1",
    "path_dark_12s_esis1",
]

path_fe55_esis1 = pathlib.Path(__file__).parent / "_data/fe55/ESIS1_00002.fit.gz"
"""
An Fe 55 sample image from the ESIS channel 1 camera. 

Gathered by the MSFC sounding rocket team during camera testing and validation 
on 2017-07-06. 
"""

path_fe55_esis3 = pathlib.Path(__file__).parent / "_data/fe55/ESIS3_05400.fit.gz"
"""
An Fe 55 sample image from the ESIS channel 3 camera.

Gathered by the MSFC sounding rocket team during camera testing and validation
on 2017-07-12.
"""

paths_fe55_esis3 = tuple(
    pathlib.Path(__file__).parent / f"_data/fe55/ESIS3_{i:05d}.fit.gz"
    for i in (5400, 5408, 5416, 5424)
)
"""
A sequence of four Fe 55 images from the ESIS channel 3 camera,
beginning with :obj:`path_fe55_esis3`.

A single image holds only a few tens of Fe 55 events in each tap.
Four is enough for the gain of every tap to settle to within about half a
percent of the value measured from forty.
"""

path_dark_esis1 = pathlib.Path(__file__).parent / "_data/darks/ESIS1_00099.fit.gz"
"""
A sample dark image from the ESIS channel 1 camera.

Captured during the ESIS launch on 2019-09-30.
"""

path_dark_esis3 = pathlib.Path(__file__).parent / "_data/darks/ESIS3_00099.fit.gz"
"""
A sample dark image from the ESIS channel 3 camera.

Captured during the ESIS launch on 2019-09-30.
"""

path_led_esis1 = pathlib.Path(__file__).parent / "_data/led/ESIS1_04803.fit.gz"
"""
A sample image of a diffuse LED source from the ESIS channel 1 camera.

Gathered by the MSFC sounding rocket team during the linearity test
on 2017-07-12.
"""

path_led_esis1_next = pathlib.Path(__file__).parent / "_data/led/ESIS1_04804.fit.gz"
"""
The next image in the sequence after :obj:`path_led_esis1`.

Captured two seconds later with the same illumination, so the pair can be
used to measure the shot noise and the gain of the camera.
"""

path_led_dark_esis1 = pathlib.Path(__file__).parent / "_data/led/ESIS1_04860.fit.gz"
"""
A sample dark image from the ESIS channel 1 camera.

Gathered two minutes after :obj:`path_led_esis1`, with the LED turned off.
"""

path_led_dark_esis1_next = (
    pathlib.Path(__file__).parent / "_data/led/ESIS1_04861.fit.gz"
)
"""
The next dark image in the sequence after :obj:`path_led_dark_esis1`.

Captured two seconds later, so the pair can be used to measure the
readout noise of the camera.
"""

path_dark_2s_esis1 = (
    pathlib.Path(__file__).parent / "_data/dark_current/ESIS1_01772.fit.gz"
)
"""
A two-second dark image from the ESIS channel 1 camera.

Gathered by the MSFC sounding rocket team during the first dark test
on 2017-07-12.
"""

path_dark_12s_esis1 = (
    pathlib.Path(__file__).parent / "_data/dark_current/ESIS1_01829.fit.gz"
)
"""
A twelve-second dark image from the ESIS channel 1 camera.

Gathered eight minutes after :obj:`path_dark_2s_esis1`,
so the pair can be used to measure the dark current rate of the camera.
"""
