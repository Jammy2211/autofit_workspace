import autofit as af
import autoarray as aa

from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.fit import fit as f
from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.phase import (
    visualizer,
)


class Analysis(af.Analysis):
    def __init__(self, masked_dataset, image_path=None):

        self.masked_dataset = masked_dataset

        # The visualizer is the tool that we'll use the visualize a phase's unmasked dataset (before the model-fitting
        # begins) and the best-fit solution found by the model-fit (during and after the model-fitting).

        # Check out 'visualizer.py' for more details.

        self.visualizer = visualizer.Visualizer(
            masked_dataset=masked_dataset, image_path=image_path
        )

    def fit(self, instance):
        """Determine the fit of a Gaussian to the dataset, using the model-instance of a Gaussian.

        Parameters
        ----------
        instance
            The Gaussian model instance.

        Returns
        -------
        fit : Fit.likelihood
            The likelihood value indicating how well this model fit the masked dataset.
        """
        model_image = self.model_image_from_instance(instance=instance)
        fit = self.fit_from_model_image(model_image=model_image)
        return fit.likelihood

    def model_image_from_instance(self, instance):
        return instance.gaussian.image_from_grid(grid=self.masked_dataset.grid).in_2d

    def fit_from_model_image(self, model_image):
        return f.DatasetFit(masked_dataset=self.masked_dataset, model_data=model_image)

    def visualize(self, instance, during_analysis):

        # During a phase, the 'visualize' method is called throughout the model-fitting. The 'instance' passed into
        # the visualize method is highest likelihood solution obtained by the model-fit so far.

        # In the analysis we use this instance to create the best-fit fit of our model-fit.

        model_image = self.model_image_from_instance(instance=instance)
        fit = self.fit_from_model_image(model_image=model_image)

        # The visualizer now outputs images of the best-fit results to hard-disk (checkout 'visualizer.py'.)

        self.visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
