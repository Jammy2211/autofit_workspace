import autofit as af

from astropy.io import fits
import numpy as np


class Dataset:
    def __init__(self, data, noise_map, psf, pixel_scale):
        """A class containing the data, noise-map, psf and pixel-scale of a 2D imaging dataset.

        Parameters
        ----------
        image : np.ndarray
            The array of the image data, in units of electrons per second.
        noise_map : np.ndarray
            An array describing the RMS standard deviation error in each pixel in units of electrons per second.
        psf : np.ndarray
            An array describing the Point Spread Function kernel of the image.
        pixel_scales: float
            The arc-second to pixel conversion factor of each pixel.
        """

        self.data = data
        self.noise_map = noise_map
        self.psf = psf
        self.pixel_scale = pixel_scale

    @property
    def grid(self):
    #    return np.arange(self.data.shape[0])

    @classmethod
    def from_fits(cls, data_path, noise_map_path, psf_path, pixel_scale):
        """Load the data, noise-map and psf of a 2D galaxy imaging dataset from '.fits' files.

        Parameters
        ----------
        data_path : np.ndarray
            The path on your hard-disk to the '.fits' file of the data.
        noise_map_path : np.ndarray
            The path on your hard-disk to the '.fits' file of the noise-map.
        psf_path : np.ndarray
            The path on your hard-disk to the '.fits' file of the psf.
        pixel_scales: float
            The arc-second to pixel conversion factor of each pixel.
        """
        data_hdu_list = fits.open(data_path)
        noise_map_hdu_list = fits.open(noise_map_path)
        psf_hdu_list = fits.open(psf_path)

        data = np.array(data_hdu_list[0].data)
        noise_map = np.array(noise_map_hdu_list[0].data)
        psf = np.array(psf_hdu_list[0].data)

        return Dataset(data=data, noise_map=noise_map, psf=psf, pixel_scale=pixel_scale)
