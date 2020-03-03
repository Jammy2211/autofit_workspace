import autofit as af
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.fit.fit import DatasetFit


class Analysis(af.Analysis):
    def __init__(self, dataset):

        self.dataset = dataset

    # Below, 'instance' is a model-component, e.g. an instance of the Gaussian class with a set of parameters. The
    # parameters are mapped from the priors using values determined from the non-linear. Crucially, this is how we
    # have access to an instance of our model so as to fit our data!

    # The reason a Gaussian is mapped to an instance in this way is because of the line:

    # gaussian = af.PhaseProperty("gaussian")

    # In phase.py. Thus, PhaseProperties define our model components and thus tell the non-linear search what it is
    # fitting!

    def fit(self, instance):
        """
        Determine the fit of a Gaussian to the imaging, using the model-instance of a Gaussian.

        Parameters
        ----------
        instance
            The Gaussian model instance.

        Returns
        -------
        fit : Fit.likelihood
            The likelihood value indicating how well this model fit the masked imaging dataset.
        """

        model_image = instance.gaussian.image_from_grid(grid=self.dataset.grid)
        fit = self.fit_from_model_image(model_image=model_image)
        return fit.likelihood

    def fit_from_model_image(self, model_image):
        return DatasetFit(data=self.dataset.image, noise_map=self.dataset.noise_map, model_data=model_image)

    def visualize(self, instance, during_analysis):

        # Visualization will be covered in tutorial 4.

        # Do not delete this, as PyAutoFit will crash otherwise!

        pass
