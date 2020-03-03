import autofit as af
from autofit.tools.phase import Dataset
from howtofit.chapter_1_introduction.tutorial_4_phase_features import tagging
from howtofit.chapter_1_introduction.tutorial_4_phase_features.meta_dataset import (
    MetaDataset,
)
from howtofit.chapter_1_introduction.tutorial_4_phase_features.result import Result
from howtofit.chapter_1_introduction.tutorial_4_phase_features.analysis import Analysis


class Phase(af.AbstractPhase):

    gaussian = af.PhaseProperty("gaussian")

    Result = Result

    @af.convert_paths
    def __init__(
        self, paths, gaussian, signal_to_noise_limit=None, optimizer_class=af.MultiNest
    ):
        """
        A phase which fits a Gaussian model using a non-linear search.

        Parameters
        ----------
        paths : af.Paths
            Handles the output directory structure.
        gaussian : gaussians.Gaussian
            The model component Gaussian class fitted by this phase.
        optimizer_class: class
            The class of a non_linear optimizer
        """

        phase_tag = tagging.phase_tag_from_phase_settings(
            signal_to_noise_limit=signal_to_noise_limit
        )
        paths.phase_tag = phase_tag

        super().__init__(paths=paths, optimizer_class=optimizer_class)

        self.gaussian = gaussian

        self.meta_dataset = MetaDataset(signal_to_noise_limit=signal_to_noise_limit)

    def run(self, dataset: Dataset, mask):
        """
        Pass a dataset to the phase, running the phase and non-linear search.

        Parameters
        ----------
        dataset: aa.Imaging
            The dataset fitted by the phase, which in this case is a PyAutoArray imaging object.
        mask: Mask
            The mask used for the analysis.

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model.
        """

        analysis = self.make_analysis(dataset=dataset, mask=mask)

        result = self.run_analysis(analysis=analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, mask):
        """
        Create an Analysis object, which creates the dataset and contains the functions which perform the fit.

        Parameters
        ----------
        dataset: aa.Imaging
            The dataset fitted by the phase, which in this case is a PyAutoArray imaging object.

        Returns
        -------
        analysis : Analysis
            An analysis object that the non-linear search calls to determine the fit likelihood for a given model
            instance.
        """

        masked_dataset = self.meta_dataset.masked_dataset_from_dataset_and_mask(
            dataset=dataset, mask=mask
        )

        return Analysis(
            masked_dataset=masked_dataset, image_path=self.optimizer.paths.image_path
        )

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            figure_of_merit=result.figure_of_merit,
            analysis=analysis,
        )
