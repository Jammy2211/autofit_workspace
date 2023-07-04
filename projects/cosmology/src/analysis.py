import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import List

import autofit as af


class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray, psf: np.ndarray, grid: np.ndarray):
        """
        The analysis class for the **PyAutoFit** example Astronomy project on gravitational lensing.

        This class contains imaging data of a strong gravitational lens, and it fits it with a lens model which can
        include:

        1) The light and mass of the foreground galaxy responsible for strong lensing.
        2) The light of the background source galaxy which is observed multiple times.
        3) The light and mass of additional galaxies nearby, that may be included in the strong lens model.

        The imaging data contains:

        1) An image of the strong lens.
        2) The noise in every pixel of that image.
        3) The Point Spread Function (PSF) describing how the optics of the telescope blur the image.
        4) A (y,x) grid of coordinates describing the locations of the image pixels in a unit system, which are used
           to ray-trace the strong lens model.

        This project is a scaled down version of the Astronomy project **PyAutoLens**, which was the original project
        from which PyAutoFit is an offshoot!

        https://github.com/Jammy2211/PyAutoLens

        Parameters
        ----------
        data
            The image containing the observation of the strong gravitational lens that is fitted.
        noise_map
            The RMS noise values of the image data, which is folded into the log likelihood calculation.
        grid
            The (y, x) coordinates of the image from which the lensing calculation is performed and model image is
            computed using.
        """

        self.data = data
        self.noise_map = noise_map
        self.psf = psf
        self.grid = grid

        # The circular masking introduces zeros at the edge of the noise-map,
        # which can lead to divide-by-zero errors.
        # We set these values to 1.0e8, to ensure they do not contribute to the log likelihood.
        self.noise_map_fit = noise_map
        self.noise_map_fit[noise_map == 0] = 1.0e8

    def log_likelihood_function(self, instance) -> float:
        """
        The `log_likelihood_function` of the strong lensing example, which performs the following step:

        1) Using the lens model passed into the function (whose parameters are set via the non-linear search) create
        an image of this strong lens. This uses gravitational lensing ray-tracing calculations.

        2) Subtract this model image from the data and compute its residuals, chi-squared and likelihood via the
        noise map.

        Parameters
        ----------
        instance
            An instance of the lens model set via the non-linear search.

        Returns
        -------
        float
            The log likelihood value of this particular lens model.
        """

        """
        The 'instance' that comes into this method contains the `Galaxy`'s we setup in the `Model` and `Collection`,
        which can be seen by uncommenting the code below.
        """

        # print("Lens Model Instance:")
        # print("Lens Galaxy = ", instance.galaxies.lens)
        # print("Lens Galaxy Bulge = ", instance.galaxies.lens.light_profile_list)
        # print("Lens Galaxy Bulge Centre = ", instance.galaxies.lens.light_profile_list[0].centre)
        # print("Lens Galaxy Mass Centre = ", instance.galaxies.lens.mass_profile_list[0].centre)
        # print("Source Galaxy = ", instance.galaxies.source)

        """
        The methods of the `Galaxy` class are available, making it easy to fit
        the lens model.
        """

        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data

        normalized_residual_map = residual_map / self.noise_map_fit

        chi_squared_map = (normalized_residual_map) ** 2.0

        chi_squared = np.sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map**2.0))

        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):
        """
        Visualizes the maximum log likelihood model-fit to the strong lens data so far in the non-linear search, as
        well as at the end of the search.

        Parameters
        ----------
        paths
            The **PyAutoFit** `Paths` object which includes the path to the output image folder.
        instance
            An instance of the lens model set via the non-linear search.
        during_analysis
            If the visualization is being performed during the non-linear search or one it is complete.
        """

        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        normalized_residual_map = residual_map / self.noise_map_fit
        chi_squared_map = (normalized_residual_map) ** 2.0

        def plot_array(array, name, title=None, norm=None):

            import os

            os.makedirs(paths.image_path, exist_ok=True)

            plt.imshow(array, norm=norm)
            plt.colorbar()
            plt.title(title)
            plt.savefig(f"{paths.image_path}/{name}.png")
            plt.close()

        plot_array(array=self.data, title="Data", name="data")
        plot_array(array=self.noise_map, title="Noise Map", name="noise_map")
        plot_array(array=model_data, title="Model Data" , name="model_data")
        plot_array(array=residual_map, title="Residual Map", name="residual_map")
        plot_array(array=normalized_residual_map, title="Normalized Residual Map",  name="normalized_residual_map")
        plot_array(array=chi_squared_map, title="Chi-Squared Map", name="chi_squared_map")

    def traced_grid_from(self, instance) -> List[np.ndarray]:
        """
        This function performs ray-tracing calculations describing how light is gravitationally lensed
        on its path through the Universe by a foreground galaxy.

        For the purpose of illustrating **PyAutoFit** you do not need to understand what this function is
        doing, the main thing to note is that it allows us to create a model lensed image of a strong
        lens model.

        Nevertheless, if you are curious, a simple description of the steps performed is as follows:

        1) Go to the foreground `lens` galaxy in the model `instance` (the lower redshift galaxy) and use
           its mass profiles to compute the deflection angles of light-rays, describing how the light
           of the galaxies behind it are deflected.

        2) Subtract these deflection angles from the image-plane coordinates to determine where these
           light-rays end up after bending, meaning that they tell us how the deflected image of the
           background galaxies appear.

        3) Repeat this galaxy by galaxy, working out how they deflect light and how this changes the
           appearance of the images of the background galaxies.

        There are strong lens systems in nature which consist of multiple galaxies at different redshifts
        (e.g. distances from Earth). Such systems require their own bespoke ray-tracing calculations,
        called "multi-plane" ray-tracing. This function does not perform these calculations, as they are
        quite complicated and would make the code below less clear.

        Parameters
        ----------
        instance
            An instance of the lens model set via the non-linear search.

        Returns
        -------
        grids_at_planes
            The deflected (y,x) coordinates of the original image at every plane of the strong lens system, which can
            be used for computing its image.
        """

        lens = instance.galaxies[0]

        deflections = lens.deflections_from_grid(grid=self.grid)

        lensed_grid = self.grid - deflections

        return lensed_grid

    def model_data_from_instance(self, instance):
        """
        Create the image of a strong lens system, accounting for the complicated ray-tracing calculations that
        describe how light is gravitationally lensed on its path through the Universe by nearby galaxies.

        For the purpose of illustrating **PyAutoFit** you do not need to understand what this function is doing, the
        main thing to note is that it allows us to create a model lensed image of a strong lens model.

        Nevertheless, if you are curious, it simply computes deflected (y,x) Cartesian grids by performing ray-tracing
        using the `grids_via_ray_tracing_from` function and uses each galaxy's light profile, based on where these
        deflected grids fall on the image, to evaluate the appearance of each galaxy. This is summed to create the
        overall image of the strong lens system.

        NOTE: To avoid an AstroPy dependency the calculation below is not strictly correct, as it omits a Cosmology
        calculation that rescales the deflection of light. This does not impact the project's
        illustration of **PyAutoFit**.

        Parameters
        ----------
        instance
            An instance of the lens model set via the non-linear search.

        Returns
        -------
        image
            The image of this strong lens model.
        """

        lens = instance.galaxies[0]
        lens_image = lens.image_from_grid(grid=self.grid)

        source = instance.galaxies[1]
        traced_grid = self.traced_grid_from(instance=instance)
        source_image = source.image_from_grid(grid=traced_grid)

        overall_image = lens_image + source_image

        # The grid has zeros at its edges, which produce nans in the model image.
        # These lead to an ill-defined log likelihood, so we set them to zero.
        overall_image = np.nan_to_num(overall_image)

        model_data = signal.convolve2d(
            overall_image, self.psf, mode="same"
        )

        return model_data
