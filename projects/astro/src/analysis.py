import numpy as np
import matplotlib.pyplot as plt
from typing import List

import autofit as af


class Analysis(af.Analysis):
    def __init__(self, image: np.ndarray, noise_map: np.ndarray, grid: np.ndarray):
        """
        The analysis class for the **PyAutoFit** example Astronomy project on gravitational lensing.

        This class contains imaging data of a strong gravitational lens, and it fits it with a lens model which can
        include:

        1) The light and mass of the foreground galaxy responsible for strong lensing.
        2) The light of the background source galaxy which is observed multiple times.
        3) The light and mass of additional galaxies nearby, that may be included in the strong lens model.

        This project is a scaled down version of the Astronomy project **PyAutoLens**, which was the original project
        from which PyAutoFit is an offshoot!

        https://github.com/Jammy2211/PyAutoLens

        Parameters
        ----------
        image
            The image containing the observation of the strong gravitational lens that is fitted.
        noise_map
            The RMS noise values of the image data, which is folded into the log likelihood calculation.
        grid
            The (y, x) coordinates of the image from which the lensing calculation is performed and model image is
            computed using.
        """

        self.image = image
        self.noise_map = noise_map
        self.grid = grid

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

        model_image = self.model_image_from_instance(instance=instance)

        residual_map = self.image - model_image
        normalized_residual_map = residual_map / self.noise_map
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

        model_image = self.model_image_from_instance(instance=instance)

        residual_map = self.image - model_image
        normalized_residual_map = residual_map / self.noise_map
        chi_squared_map = (normalized_residual_map) ** 2.0

        def plot_array(array, title=None, norm=None):
            plt.imshow(array, norm=norm)
            plt.colorbar()
            plt.title(title)
            plt.show()
            plt.close()

        plot_array(array=self.image, title="Data")
        plot_array(array=self.noise_map, title="Noise Map")
        plot_array(array=model_image, title="Noise Map")
        plot_array(array=residual_map, title="Residual Map")
        plot_array(array=normalized_residual_map, title="Normalized Residual Map")
        plot_array(array=chi_squared_map, title="Chi-Squared Map")

    def grids_at_planes_from_instance(self, instance) -> List[np.ndarray]:
        """
        This function performs complicated ray-tracing calculations that describe how light is gravitationally lensed
        on its path through the Universe by nearby galaxies.

        For the purpose of illustrating **PyAutoFit** you do not need to understand what this function is doing, the
        main thing to note is that it allows us to create a model lensed image of a strong lens model.

        For the enthusiastic amongst you, the following paper gives the complete mathematical formalism for performing
        these calculations:

        https://arxiv.org/abs/1409.0015

        NOTE: To avoid an AstroPy dependency the calculation below is not strictly correct, as it omits a Cosmology
        calculation that rescales the deflection of light. This does not impact the project's
        illustration of **PyAutoFit**.

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

        grids_at_planes = []
        deflections_at_planes = []

        for galaxy_index, galaxy in enumerate(instance.galaxies):
            scaled_grid = self.grid

            if galaxy_index > 0:
                for previous_galaxy_index in range(galaxy_index):
                    scaled_deflections = deflections_at_planes[previous_galaxy_index]

                    scaled_grid -= scaled_deflections

            grids_at_planes.append(scaled_grid)
            deflections_at_planes.append(galaxy.deflections_from_grid(grid=scaled_grid))

        return grids_at_planes

    def model_image_from_instance(self, instance):
        """
        Create the image of a strong lens system, accounting for the complicated ray-tracing calculations that
        describe how light is gravitationally lensed on its path through the Universe by nearby galaxies.

        For the purpose of illustrating **PyAutoFit** you do not need to understand what this function is doing, the
        main thing to note is that it allows us to create a model lensed image of a strong lens model.

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

        model_image = np.zeros(shape=self.image.shape)

        grids_at_planes = self.grids_at_planes_from_instance(instance=instance)

        for galaxy_index, galaxy in enumerate(instance.galaxies):
            model_image += galaxy.image_from_grid(grid=grids_at_planes[galaxy_index])

        return model_image
