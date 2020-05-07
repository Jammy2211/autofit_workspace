import autofit as af

# The 'analysis.py' module contains the dataset and log likelihood function which given a model instance (set up by
# the non-linear search) fits the dataset and returns the log likelihood of that model.


class Analysis(af.Analysis):

    # In this example the Analysis only contains the data and noise-map. It can be easily extended however, for more
    # complex data-sets and model fitting problems.

    def __init__(self, data, noise_map):

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

        Returns
        -------
        fit : Fit.log_likelihood
            The log likelihood value indicating how well this model fit the dataset.
        """
        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)
        return fit.log_likelihood

    def model_data_from_instance(self, instance):
        return instance.gaussian.line_from_xvalues(xvalues=self.dataset.xvalues)

    def fit_from_model_data(self, model_data):
        return f.DatasetFit(dataset=self.dataset, model_data=model_data)

    def visualize(self, instance, during_analysis):

        # Visualization will be covered in tutorial 4.

        # Do not delete this, as PyAutoFit will crash otherwise!

        pass
