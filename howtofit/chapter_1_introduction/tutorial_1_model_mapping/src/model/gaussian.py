import autofit as af

# PyAutoFit interfaces with Python classes to define the model-components of a model, where the __init__ constructor
# arguments form the model parameters

# For your model-fitting problem, you will have classes corresponding to the different models you might it. For example,
# in later tutorials when we extend our model, we'll have more 1D profiles such as Exponential and an AsymetricGaussian.


class Gaussian(af.ModelObject):
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        intensity=0.1,  # <- are the Gaussian's model parameters.
        sigma=0.01,
    ):
        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma
