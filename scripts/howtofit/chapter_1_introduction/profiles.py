import numpy as np

"""
In tutorial 5, we perform modeling using multiple profiles, in particular the `Gaussian` profile from the previous
tutorials and an Exponential profile. In analysis.py, we will edit how model-data is generated from profiles such
that it is the sum of all profiles in our model.

In this module, we thus now have two classes following the PyAutoFit model component format. We have renamed the
module from `gaussian.py` to `profiles.py` to reflect this. We have created an abstract base class `Profile` from
which all profiles inherit.

If you are not familiar with Python classes, in particular inheritance and the `super` method below, you may
be unsure what the classes are doing below. I have included comments describing what these command do.

The Profile class is a base class from which all profiles we add (e.g Gaussian, Exponential, additional profiles
added down the line) will inherit. This is useful, as it signifinies which aspects of our model are different ways of
representing the same thing.
"""


class Profile:
    def __init__(self, centre=0.0, normalization=0.01):
        """
        Represents an Abstract 1D profile.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the profile.

        Every profile class we add below (e.g. Gaussian, Exponential) will call this __init__ method of the Profile
        base class. Given that every profile will have a centre and normalization, this means we can set these parameters
        in the Profile class`s init method instead of repeating the two lines of code for every individual profile.
        """
        self.centre = centre
        self.normalization = normalization


"""
The inclusion of (Profile) in the `Gaussian` below instructs Python that the `Gaussian` class is going to inherit from
the Profile class.
"""


class Gaussian(Profile):
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization=0.1,  # <- are the Gaussian`s model parameters.
        sigma=0.01,
    ):
        """Represents a 1D `Gaussian` profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a `NonLinearSearch`.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        sigma
            The sigma value controlling the size of the Gaussian.

        Writing (Profile) above does not mean the `Gaussian` class will call the Profile class`s __init__ method. To
        achieve this we have the call the `super` method following the format below.
        """

        super(Gaussian, self).__init__(centre=centre, normalization=normalization)

        """
        This super method calls the __init__ method of the Profile class above, which means we do not need
        to write the two lines of code below (which are commented out given they are not necessary).
        """

        # self.centre = centre
        # self.normalization = normalization

        self.sigma = sigma  # We still need to set sigma for the Gaussian, of course.

    def model_data_1d_via_xvalues_from(self, xvalues):
        """
        Calculate the 1D Gaussian profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        values
            The x coordinates in the original reference frame of the grid.
        """

        transformed_xvalues = np.subtract(xvalues, self.centre)

        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


class Exponential(Profile):
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments are the model
        normalization=0.1,  # <- parameters of the Gaussian.
        rate=0.01,
    ):
        """Represents a 1D Exponential profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a `NonLinearSearch`.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        ratw
            The decay rate controlling has fast the Exponential declines.
        """

        super(Exponential, self).__init__(centre=centre, normalization=normalization)

        self.rate = rate

    def model_data_1d_via_xvalues_from(self, xvalues):
        """
        Calculate the 1D Gaussian profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Exponential, using its centre.

        Parameters
        ----------
        values
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.normalization * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )
