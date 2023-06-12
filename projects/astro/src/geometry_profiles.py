import typing
import numpy as np


class GeometryProfile:
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
    ):
        """
        Abstract base class for the geometry of a profile representing the light or mass of a galaxy.

        Using the centre, axis-ratio and position angle of the profile this class describes how to convert a
        (y,x) grid of Cartesian coordinates to the elliptical geometry of the profile.

        Parameters
        ----------
        centre
            The (y,x) coordinates of the profile centre.
        axis_ratio
            The axis-ratio of the ellipse (minor axis / major axis).
        angle
            The rotation angle in degrees counter-clockwise from the positive x-axis.
        """

        self.centre = centre
        self.axis_ratio = axis_ratio
        self.angle = angle

    def transformed_to_reference_frame_grid_from(self, grid: np.ndarray):
        """
        Transform a grid of (y,x) coordinates to the geometric reference frame of the profile via a translation using
        its `centre` and a rotation using its `angle`.

        Parameters
        ----------
        grid
            The (y, x) coordinate grid in its original reference frame.
        """

        shifted_grid = grid - self.centre
        radius = ((shifted_grid[:, :, 0] ** 2.0 + shifted_grid[:, :, 1] ** 2.0)) ** 0.5

        theta_coordinate_to_profile = np.arctan2(
            shifted_grid[:, :, 1], shifted_grid[:, :, 0]
        ) - np.radians(self.angle)

        transformed_grid = np.zeros(grid.shape)

        transformed_grid[:, :, 0] = radius * np.cos(theta_coordinate_to_profile)
        transformed_grid[:, :, 1] = radius * np.sin(theta_coordinate_to_profile)

        return transformed_grid

    def rotated_grid_from_reference_frame_from(self, grid: np.ndarray) -> np.ndarray:
        """
        Rotate a grid of (y,x) coordinates which have been transformed to the elliptical reference frame of a profile
        back to the original unrotated coordinate frame.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of an elliptical profile.
        """
        cos_angle = np.cos(np.radians(self.angle))
        sin_angle = np.sin(np.radians(self.angle))

        transformed_grid = np.zeros(grid.shape)

        transformed_grid[:, :, 0] = np.add(
            np.multiply(grid[:, :, 0], cos_angle),
            -np.multiply(grid[:, :, 1], sin_angle),
        )
        transformed_grid[:, :, 1] = np.add(
            np.multiply(grid[:, :, 0], sin_angle), np.multiply(grid[:, :, 1], cos_angle)
        )

        return transformed_grid

    def elliptical_radii_grid_from(self, grid: np.ndarray) -> np.ndarray:
        """
        Convert a grid of (y,x) coordinates to a grid of elliptical radii.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """

        return (
            (grid[:, :, 1] ** 2.0) + (grid[:, :, 0] / self.axis_ratio) ** 2.0
        ) ** 0.5
