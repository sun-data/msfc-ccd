# msfc-ccd

[![tests](https://github.com/sun-data/msfc-ccd/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/msfc-ccd/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/sun-data/msfc-ccd/graph/badge.svg?token=sbIjziJUHL)](https://codecov.io/github/sun-data/msfc-ccd)[![Black](https://github.com/sun-data/msfc-ccd/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/msfc-ccd/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/msfc-ccd/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/msfc-ccd/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/msfc-ccd/badge/?version=latest)](https://msfc-ccd.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/msfc-ccd.svg)](https://badge.fury.io/py/msfc-ccd)

A Python library for characterizing and using the CCD cameras developed by Marshall Space Flight Center.

More information is available in the [documentation](https://msfc-ccd.readthedocs.io/).

## Installation

`msfc-ccd` is available on PyPI and can be installed using pip:

```shell
pip install msfc-ccd
```

## Examples

Load and display a single FITS file.

```python3
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
```
![A sample FITS image](https://msfc-ccd.readthedocs.io/en/latest/_images/index_0_0.png)
