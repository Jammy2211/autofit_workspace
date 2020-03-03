import autoarray as aa
import numpy as np


class Gaussian:
    def __init__(
        self,
        centre=(0.0, 0.0),  # <- PyAutoFit recognises these constructor arguments
        intensity=0.1,  # <- are the Gaussian's model parameters.
        sigma=0.01,
    ):
        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma

    def image_from_grid(self, grid):
        """
        Calculate the intensity of the light profile on a grid of Cartesian (y,x) coordinates.

        The input grid of (y,x) coordinates is translated to a coordinate system centred on the Gaussian, using its
        centre.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        """
        transformed_grid = np.subtract(grid, self.centre)
        grid_radii = np.sqrt(
            np.add(np.square(transformed_grid[:, 0]), np.square(transformed_grid[:, 1]))
        )
        image = self.image_from_grid_radii(grid_radii=grid_radii)
        return aa.masked.array.manual_1d(
            array=image, mask=grid.mask
        )

    def image_from_grid_radii(self, grid_radii):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(grid_radii, self.sigma))),
        )
