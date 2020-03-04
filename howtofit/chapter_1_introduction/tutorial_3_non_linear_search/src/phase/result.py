import autofit as af


class Result(af.Result):
    def __init__(self, instance, likelihood, analysis, output):
        """
        The result of a non-linear search.

        Parameters
        ----------
        instance: autofit.mapper.model.ModelInstance
            A model instance comprising the model instances that gave the highest likelihood fit.
        likelihood: float
            A value indicating the figure of merit (e.g. the likelihood) given by the highest likelihood fit.
        """
        self.instance = instance
        self.likelihood = likelihood
        self.analysis = analysis
        self.output = output

    @property
    def most_likely_model_image(self):
        return self.analysis.model_image_from_instance(instance=self.instance)

    @property
    def most_likely_fit(self):
        return self.analysis.fit_from_model_image(
            model_image=self.most_likely_model_image
        )
