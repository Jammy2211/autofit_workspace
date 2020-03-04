import autofit as af
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.fit import (
    fit as f,
)

# The 'analysis.py' module contains the dataset, and given a model instance (set up via a non-linear search in
# 'phase.py' fits the model to the dataset so as to return a likelihood.


class Analysis(af.Analysis):
    def __init__(self, dataset):

        self.dataset = dataset

    # Below, 'instance' is a model-component, e.g. an instance of the Gaussian class with a set of parameters. The
    # parameters are mapped from the priors using values determined from the non-linear. Crucially, this is how we
    # have access to an instance of our model so as to fit our data!

    # The reason a Gaussian is mapped to an instance in this way is because of the line:

    # gaussian = af.PhaseProperty("gaussian")

    # In phase.py. Thus, PhaseProperties define our model components and thus tell the non-linear search what it is
    # fitting! For your model-fitting problem, you'll need to update the PhaseProperty(s) to your model-components!

    def fit(self, instance):
        """
        Determine the fit of a Gaussian to the dataset, using a model-instance of the Gaussian.

        Parameters
        ----------
        instance
            The Gaussian model instance.

        Returns
        -------
        fit : Fit.likelihood
            The likelihood value indicating how well this model fit the dataset.
        """
        model_image = self.model_image_from_instance(instance=instance)
        fit = self.fit_from_model_image(model_image=model_image)
        return fit.likelihood

    def model_image_from_instance(self, instance):
        return instance.gaussian.image_from_grid(grid=self.dataset.grid)

    def fit_from_model_image(self, model_image):
        return f.DatasetFit(dataset=self.dataset, model_data=model_image)

    def visualize(self, instance, during_analysis):

        # Visualization will be covered in tutorial 4.

        # Do not delete this, as PyAutoFit will crash otherwise!

        pass
