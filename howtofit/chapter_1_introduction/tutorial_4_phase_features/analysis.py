import autofit as af
import autoarray as aa

from howtofit.chapter_1_introduction.tutorial_4_phase_features import visualizer

class Analysis(af.Analysis):
    def __init__(self, masked_dataset, image_path=None):

        self.visualizer = visualizer.DatasetVisualizer(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.masked_dataset = masked_dataset

    @property
    def masked_imaging(self):
        return self.masked_dataset

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
        model_image = instance.gaussian.image_from_grid(grid=self.masked_imaging.grid)
        fit = self.fit_from_model_image(model_image=model_image)
        return fit.likelihood

    def fit_from_model_image(self, model_image):
        return aa.fit(masked_dataset=self.masked_imaging, model_data=model_image)

    def visualize(self, instance, during_analysis):

        model_image = instance.gaussian.image_from_grid(grid=self.masked_imaging.grid)
        fit = self.fit_from_model_image(model_image=model_image)

        self.visualizer.visualize_fit(
            fit=fit, during_analysis=during_analysis
        )