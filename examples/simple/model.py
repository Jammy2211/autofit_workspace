import numpy as np

# The Gaussian class in this module is the model components that is fitted to data using a non-linear search. The
# inputs of its __init__ constructor are the parameters which can be fitted for.

# The log_likelihood_function in the Analysis class receives an instance of this classes where the values of its
# parameters have been set up according to the non-linear search. Because instances of the classes are used, this means
# their methods (e.g. line_from_xvalues) can be used in the log likelihood function.


class Gaussian:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        intensity=0.1,  # <- are the Gaussian's model parameters.
        sigma=0.01,
    ):
        """Represents a 1D Gaussian line profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the Gaussian line profile.
        sigma : float
            The sigma value controlling the size of the Gaussian.
        """

        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma

    def line_from_xvalues(self, xvalues):
        """
        Calculate the intensity of the line profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues : ndarray
            The x coordinates in the original reference frame of the grid.
        """

        transformed_xvalues = xvalues - self.centre

        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )
