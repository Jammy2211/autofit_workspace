from src import geometry_profiles as gp

import typing
import numpy as np


class MassProfile(gp.GeometryProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
        mass: float = 1.0,
    ):
        """
        Abstract base class for a mass profile, which describes the mass of a galaxy as a function of radius.

        Parameters
        ----------
        centre
            The (y,x) coordinates of the profile centre.
        axis_ratio
            The axis-ratio of the ellipse (minor axis / major axis).
        angle
            The rotation angle in degrees counter-clockwise from the positive x-axis.
        mass
            The mass intensity of the profile, which is the Einstein effective_radius in arc-seconds.
        """

        super().__init__(centre=centre, axis_ratio=axis_ratio, angle=angle)

        self.mass = mass


class MassIsothermal(MassProfile):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
        mass: float = 1.0,
    ):
        """
        The elliptical isothermal mass distribution often used in Astronomy to represent the combined mass of stars
        and dark matter in galaxies.

        Parameters
        ----------
        centre
            The (y,x) coordinates of the profile centre.
        axis_ratio
            The axis-ratio of the ellipse (minor axis / major axis).
        angle
            The rotation angle in degrees counter-clockwise from the positive x-axis.
        mass
            The mass intensity of the profile, which is the Einstein effective_radius in arc-seconds.
        """
        super().__init__(centre=centre, axis_ratio=axis_ratio, angle=angle, mass=mass)

    def psi_from(self, grid: np.ndarray) -> np.ndarray:
        """
        Returns `psi`, a value required when computing the deflections of the elliptical isothermal mass distribution,
        where:

        psi = sqrt((q^2 * x^2) + y^2)

        Parameters
        ----------
        grid
            The (y,x) coordinates of the grid where the `psi`'s are computed.
        """

        return (
            (self.axis_ratio**2.0 * grid[:, :, 0] ** 2.0) + grid[:, :, 1] ** 2.0
        ) ** 0.5

    def deflections_from_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Returns the deflection angles on a grid of (y,x) coordinates, which describe how
        the isothermal mass profile bends light via the effect of gravitational lensing.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        grid_transformed = self.transformed_to_reference_frame_grid_from(grid=grid)

        factor = 2.0 * self.mass * self.axis_ratio / np.sqrt(1 - self.axis_ratio**2)

        psi = self.psi_from(grid=grid_transformed)

        deflections = np.zeros(grid.shape)

        deflections[:, :, 0] = factor * np.arctan(
            np.divide(
                np.multiply(np.sqrt(1 - self.axis_ratio**2), grid_transformed[:, :, 0]),
                psi,
            )
        )
        deflections[:, :, 1] = factor * np.arctanh(
            np.divide(
                np.multiply(np.sqrt(1 - self.axis_ratio**2), grid_transformed[:, :, 1]),
                psi,
            )
        )

        return self.rotated_grid_from_reference_frame_from(grid=deflections)
