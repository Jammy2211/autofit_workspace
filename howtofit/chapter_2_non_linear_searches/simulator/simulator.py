import os
from astropy.io import fits
from howtofit.chapter_1_introduction.tutorial_6_complex_models.src.model import profiles

import numpy as np

# %%
#%matplotlib inline

# %%
"""
___Simulator___
This script simulates the 1D Gaussians line profile datasets used throughout chapter 1.
"""


def numpy_array_1d_to_fits(array_1d, file_path, overwrite=False):
    """Write a 1D NumPy array to a .fits file.

    Parameters
    ----------
    array_1d : ndarray
        The 1D array that is written to fits.
    file_path : str
        The full path of the file that is output, including the file name and '.fits' extension.
    overwrite : bool
        If True and a file already exists with the input file_path the .fits file is overwritten. If False, an error \
        will be raised.

    Returns
    -------
    None

    Examples
    --------
    array_1d = np.ones(shape=(,5))
    numpy_array_1d_to_fits(array_1d=array_1d, file_path='/path/to/file/filename.fits', overwrite=True)
    """

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array_1d, new_hdr)
    hdu.writeto(file_path)


# %%
"""
The path to the chapter and dataset folder on your computer. The data should be distributed with PyAutoFit, however
if you wish to reuse this script to generate it again (or new datasets) you must update the paths appropriately.
"""

# %%
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_2_non_linear_searches"
dataset_path = f"{chapter_path}/dataset"

# %%
"""
__Gaussian x2 split__

Setup the path and filename the .fits file of the Gaussian is written to.
"""

# %%
data_path = f"{dataset_path}/gaussian_x2_split"

# %%
"""
Create the model instances of the two Gaussians.
"""

# %%
gaussian_0 = profiles.Gaussian(centre=25.0, intensity=30.0, sigma=3.0)
gaussian_1 = profiles.Gaussian(centre=75.0, intensity=30.0, sigma=3.0)

# %%
"""
Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
thus defining the number of data-points in our data.
"""

# %%
pixels = 100
xvalues = np.arange(pixels)

# %%
"""
Evaluate all three Gaussian model instances at every xvalues to create their model line profiles and sum them
together to create the overall model line profile.
"""

# %%
model_line = gaussian_0.line_from_xvalues(
    xvalues=xvalues
) + gaussian_1.line_from_xvalues(xvalues=xvalues)

# %%
"""
Determine the noise (at a specified signal to noise level) in every pixel of our model line profile.
"""

# %%
signal_to_noise_ratio = 25.0
noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

# %%
"""
Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
"""

# %%
data = model_line + noise
noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

# %%
"""
Output this data to fits file, so it can be loaded and fitted in the HowToFit tutorials.
"""

# %%
numpy_array_1d_to_fits(
    array_1d=data, file_path=f"{data_path}/data.fits", overwrite=True
)
numpy_array_1d_to_fits(
    array_1d=noise_map, file_path=f"{data_path}/noise_map.fits", overwrite=True
)
