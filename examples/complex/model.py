import numpy as np

# The Gaussian and Exponential classes in this module are the model components that can be fitted to data using a
# non-linear search. The inputs of their __init__ constructors are their parameters, which can be fitted for.

# The log_likelihood_function in the Analysis class receives instances of these classes where the values of their
# parameters have been set up according to the non-linear search. Because instances of the classes are used, this means
# their methods (e.g. line_from_xvalues) can be used in the log likelihood function.


class Profile:
    def __init__(self, centre=0.0, intensity=0.01):
        """Represents an Abstract 1D line profile.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the line profile.
        """

        # Every profile class we add below (e.g. Gaussian, Exponential) will call this __init__ method of the Profile
        # base class. Given that every profile will have a centre and intensity, this means we can set these parameters
        # in the Profile class's init method instead of repeating the two lines of code for every individual profile.

        self.centre = centre
        self.intensity = intensity


class Gaussian(Profile):
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

        # Writing (Profile) above does not mean the Gaussian class will call the Profile class's __init__ method. To
        # achieve this we have the call the 'super' method following the format below.
        super(Gaussian, self).__init__(centre=centre, intensity=intensity)

        # This super method calls the __init__ method of the Profile class above, which means we do not need
        # to write the two lines of code below (which are commented out given they are not necessary).

        # self.centre = centre
        # self.intensity = intensity

        self.sigma = sigma  # We still need to set sigma for the Gaussian, of course.

    def line_from_xvalues(self, xvalues):
        """
        Calculate the intensity of the line profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        values : ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


class Exponential(Profile):
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments are the model
        intensity=0.1,  # <- parameters of the Exponential.
        rate=0.01,
    ):
        """Represents a 1D Exponential line profile symmetric about a centre, which may be treated as a model-component
        of PyAutoFit the parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the Gaussian line profile.
        ratw : float
            The decay rate controlling has fast the Exponential declines.
        """

        super(Exponential, self).__init__(centre=centre, intensity=intensity)

        self.rate = rate

    def line_from_xvalues(self, xvalues):
        """
        Calculate the intensity of the line profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Exponential, using its centre.

        Parameters
        ----------
        values : ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.intensity * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )
