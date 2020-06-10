import autofit as af

import numpy as np

# The 'analysis.py' module contains the dataset and log likelihood function which given a model instance (set up by
# the non-linear search) fits the dataset and returns the log likelihood of that model.


class Analysis(af.Analysis):

    # In this example the Analysis only contains the data and noise-map. It can be easily extended however, for more
    # complex data-sets and model fitting problems.

    def __init__(self, data, noise_map):

        super().__init__()

        self.data = data
        self.noise_map = noise_map

    # In the log_likelihood_function function below, 'instance' is an instance of our model, which in this example is
    # an instance of the Gaussian class in 'model.py'. The parameters of the Gaussian are set via the non-linear
    # search. This gives us the instance of our model we need to fit our data!

    def log_likelihood_function(self, instance):
        """
        Determine the log likelihood of a fit of a Gaussian to the dataset, using a model instance of the Gaussian.

        Parameters
        ----------
        instance : model.Gaussian
            The Gaussian model instance.

        Returnsn
        -------
        fit : Fit.log_likelihood
            The log likelihood value indicating how well this model fit the dataset.
        """

        # The 'instance' that comes into this method is an instance of the Gaussian class. To convince yourself of this,
        # go ahead and uncomment the lines below and run the non-linear search.

        # print("Gaussian Instance:")
        # print("Centre = ", instance.centre)
        # print("Intensity = ", instance.intensity)
        # print("Sigma = ", instance.sigma)

        # Get the range of x-values the data is defined on, to evaluate the model of the Gaussian.
        xvalues = np.arange(self.data.shape[0])

        # Use these xvalues to create model data of our Gaussian.
        model_data = instance.line_from_xvalues(xvalues=xvalues)

        # Fit the model gaussian line data to the observed data, computing the residuals and chi-squareds.
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)

        return log_likelihood
