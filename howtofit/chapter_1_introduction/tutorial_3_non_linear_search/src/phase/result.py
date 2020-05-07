import autofit as af

# The Result class stores the results of a non-linear search's, in particular:

# - instance: The 'best-fit' model instance, corresponding to the model and combination of parameters that produced the highest
#             log_likelihood in the non-linear search.

# - log_likelihood: The highest log likelihood value of the best-fit model.

# - analysis: The Analysis class in 'analysis.py' used by the non-linear search. This allows the Result class to be
#             extended with methods that use the dataset and analysis, for examplee the 'max_log_likelihood_model_image'
#             and 'max_log_likelihood_fit' methods below.

# - output: A class that provides an interface between the non-linear search's output on your hard-disk and Python.
#           This object is described in the main tutorial script and expanded upon in tutorial 6.


class Result(af.Result):
    def __init__(self, instance, log_likelihood, analysis, samples):
        """
        The result of a non-linear search.

        Parameters
        ----------
        instance: autofit.mapper.model.ModelInstance
            A model instance comprising the model instances that gave the highest log likelihood fit.
        log_likelihood: float
            A value indicating the figure of merit (e.g. the log likelihood) given by the highest log likelihood fit.
        """
        self.instance = instance
        self.log_likelihood = log_likelihood
        self.analysis = analysis
        self.output = samples

    @property
    def max_log_likelihood_model_data(self):
        return self.analysis.model_data_from_instance(instance=self.instance)

    @property
    def max_log_likelihood_fit(self):
        return self.analysis.fit_from_model_data(
            model_data=self.max_log_likelihood_model_data
        )
