import autofit as af
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.fit import (
    fit as f,
)

# The 'analysis.py' module contains the dataset and given a model instance (set up by mapping a unit vector of a
# non-linear search to a model via the priors) fits the dataset. This returns a log likelihood for every model sampled
# by the non-linear search.


class Analysis(af.Analysis):

    # In this tutorial the Analysis only contains the dataset. More attributes will be included in later tutorials.

    def __init__(self, dataset):

        self.dataset = dataset

    # In the log_likelihood_function below, 'instance' is an instance of our model, which in this tutorial we have
    # defined as Gaussian class in 'model.py'. This instance is a unit vector mapper via each parameters prior using
    # unit values determined by the non-linear. Crucially, this gives us the instance of our model we need to fit our
    # data!

    # The reason a Gaussian is mapped to an instance in this way is because of the following line in 'phase.py':

    # gaussian = af.PhaseProperty("gaussian")

    # Thus, PhaseProperties define our model components and thus tell the non-linear search what it is fitting! For
    # your model-fitting problem, you'll need to update the PhaseProperty(s) to your model-components!

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
