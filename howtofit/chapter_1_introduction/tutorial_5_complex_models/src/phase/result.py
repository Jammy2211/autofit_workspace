import autofit as af

# The 'result.py' module is unchanged from the previous tutorial, although there is a short comment below worth reading.


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
        self.figure_of_merit = log_likelihood
        self.analysis = analysis
        self.output = samples

    @property
    def max_log_likelihood_model_data(self):

        # It is worth noting why we store the 'Analysis' class in the Result class. In this tutorial, we changed our
        # model and how it created the model-data (e.g. as a sum of line profiles). However, we did not need to change
        # the result module in any way, because it uses the 'analysis.py' module.

        # Had this function explicitly written out how the most likely model-data is created it would of needed to be
        # updated, creating more work for ourselves!

        return self.analysis.model_data_from_instance(instance=self.instance)

    @property
    def max_log_likelihood_fit(self):
        return self.analysis.fit_from_model_data(
            model_data=self.max_log_likelihood_model_data
        )
