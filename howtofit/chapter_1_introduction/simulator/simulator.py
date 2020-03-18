import os
from astropy.io import fits
from howtofit.chapter_1_introduction.tutorial_5_complex_models.src.model import profiles

import numpy as np

# This script simulates the Gaussians for chapter 1. I'm being lazy and not documenting it atm.


def numpy_array_1d_to_fits(array_1d, file_path, overwrite=False):
    """Write a 1D NumPy array to a .fits file.

    Before outputting a NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures \
    appear the same orientation as .fits files loaded in DS9.

    Parameters
    ----------
    array_2d : ndarray
        The 2D array that is written to fits.
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
    array_2d = np.ones((5,5))
    numpy_array_to_fits(array_2d=array_2d, file_path='/path/to/file/filename.fits', overwrite=True)
    """

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array_1d, new_hdr)
    hdu.writeto(file_path)


chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"
dataset_path = chapter_path + "dataset/"

### Gaussian X1 ###

data_path = dataset_path + "gaussian_x1/"
if not os.path.exists(data_path):
    os.mkdir(data_path)

signal_to_noise_ratio = 25.0
pixels = 100
xvalues = np.arange(pixels)

gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
line = gaussian.line_from_xvalues(xvalues=xvalues)

noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

data = line + noise

numpy_array_1d_to_fits(array_1d=data, file_path=data_path + "data.fits", overwrite=True)
numpy_array_1d_to_fits(
    array_1d=data, file_path=data_path + "noise_map.fits", overwrite=True
)

### Gaussian X1 + Exponential x1 ###

data_path = dataset_path + "gaussian_x1_exponential_x1/"
if not os.path.exists(data_path):
    os.mkdir(data_path)

signal_to_noise_ratio = 25.0
pixels = 100
xvalues = np.arange(pixels)
noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
exponential = profiles.Exponential(centre=70.0, intensity=40.0, rate=0.005)
line = gaussian.line_from_xvalues(xvalues=xvalues) + exponential.line_from_xvalues(
    xvalues=xvalues
)

noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

data = line + noise

numpy_array_1d_to_fits(array_1d=data, file_path=data_path + "data.fits", overwrite=True)
numpy_array_1d_to_fits(
    array_1d=noise_map, file_path=data_path + "noise_map.fits", overwrite=True
)

### Gaussian X2 + Exponential x1 ###

data_path = dataset_path + "gaussian_x2_exponential_x1/"
if not os.path.exists(data_path):
    os.mkdir(data_path)

signal_to_noise_ratio = 25.0
pixels = 100
xvalues = np.arange(pixels)
noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

gaussian_0 = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
gaussian_1 = profiles.Gaussian(centre=20.0, intensity=30.0, sigma=5.0)
exponential = profiles.Exponential(centre=70.0, intensity=40.0, rate=0.005)
line = (
    gaussian_0.line_from_xvalues(xvalues=xvalues)
    + gaussian_1.line_from_xvalues(xvalues=xvalues)
    + exponential.line_from_xvalues(xvalues=xvalues)
)

noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

data = line + noise

numpy_array_1d_to_fits(array_1d=data, file_path=data_path + "data.fits", overwrite=True)
numpy_array_1d_to_fits(
    array_1d=noise_map, file_path=data_path + "noise_map.fits", overwrite=True
)


### Gaussian x3 ###

data_path = dataset_path + "gaussian_x3/"
if not os.path.exists(data_path):
    os.mkdir(data_path)

signal_to_noise_ratio = 25.0
pixels = 100
xvalues = np.arange(pixels)
noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

gaussian_0 = profiles.Gaussian(centre=50.0, intensity=20.0, sigma=1.0)
gaussian_1 = profiles.Gaussian(centre=50.0, intensity=40.0, sigma=5.0)
gaussian_2 = profiles.Gaussian(centre=50.0, intensity=60.0, sigma=10.0)
line = (
    gaussian_0.line_from_xvalues(xvalues=xvalues)
    + gaussian_1.line_from_xvalues(xvalues=xvalues)
    + gaussian_2.line_from_xvalues(xvalues=xvalues)
)

noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

data = line + noise

numpy_array_1d_to_fits(array_1d=data, file_path=data_path + "data.fits", overwrite=True)
numpy_array_1d_to_fits(
    array_1d=noise_map, file_path=data_path + "noise_map.fits", overwrite=True
)

### Gaussian X1 (0) ###

data_path = dataset_path + "gaussian_x1_0/"
if not os.path.exists(data_path):
    os.mkdir(data_path)

signal_to_noise_ratio = 25.0
pixels = 100
xvalues = np.arange(pixels)
noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=1.0)
line = gaussian.line_from_xvalues(xvalues=xvalues)

noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

data = line + noise

numpy_array_1d_to_fits(array_1d=data, file_path=data_path + "data.fits", overwrite=True)
numpy_array_1d_to_fits(
    array_1d=noise_map, file_path=data_path + "noise_map.fits", overwrite=True
)

### Gaussian X1 (1) ###

data_path = dataset_path + "gaussian_x1_1/"
if not os.path.exists(data_path):
    os.mkdir(data_path)

signal_to_noise_ratio = 25.0
pixels = 100
xvalues = np.arange(pixels)
noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=5.0)
line = gaussian.line_from_xvalues(xvalues=xvalues)

noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

data = line + noise

numpy_array_1d_to_fits(array_1d=data, file_path=data_path + "data.fits", overwrite=True)
numpy_array_1d_to_fits(
    array_1d=noise_map, file_path=data_path + "noise_map.fits", overwrite=True
)

### Gaussian X1 (2) ###

data_path = dataset_path + "gaussian_x1_2/"
if not os.path.exists(data_path):
    os.mkdir(data_path)

signal_to_noise_ratio = 25.0
pixels = 100
xvalues = np.arange(pixels)
noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
line = gaussian.line_from_xvalues(xvalues=xvalues)

noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

data = line + noise

numpy_array_1d_to_fits(array_1d=data, file_path=data_path + "data.fits", overwrite=True)
numpy_array_1d_to_fits(
    array_1d=noise_map, file_path=data_path + "noise_map.fits", overwrite=True
)
