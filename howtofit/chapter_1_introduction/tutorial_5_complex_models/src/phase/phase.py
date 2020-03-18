import autofit as af
from howtofit.chapter_1_introduction.tutorial_5_complex_models.src.dataset import (
    masked_dataset as md,
)
from howtofit.chapter_1_introduction.tutorial_5_complex_models.src.dataset.dataset import (
    Dataset,
)
from howtofit.chapter_1_introduction.tutorial_5_complex_models.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_5_complex_models.src.phase.analysis import (
    Analysis,
)

# This module has some minor changes from tutorial 4 which are described in comments below.


class Phase(af.AbstractPhase):

    # Because we now have multiple profiles in our model, we have renamed 'gaussian' to 'profiles'. As before,
    # PyAutoFit uses this information to map the input Profile classes to a model instance when performing a fit.

    profiles = af.PhaseProperty("profiles")

    Result = Result

    @af.convert_paths
    def __init__(self, paths, profiles, optimizer_class=af.MultiNest):
        """
        A phase which fits a model composed of multiple line profiles (Gaussian, Exponential) using a non-linear search.

        Parameters
        ----------
        paths : af.Paths
            Handles the output directory structure.
        profiles : [profiles.Profile]
            The model components (e.g. Gaussian, Exponenial) fitted by this phase.
        optimizer_class: class
            The class of a non_linear optimizer
        """

        super().__init__(paths=paths, optimizer_class=optimizer_class)

        self.profiles = profiles

    def run(self, dataset: Dataset, mask):
        """
        Pass a dataset to the phase, running the phase and non-linear search.

        Parameters
        ----------
        dataset: aa.Dataset
            The dataset fitted by the phase, as defined in the 'dataset.py' module.
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
        dataset: aa.Dataset
            The dataset fitted by the phase, as defined in the 'dataset.py' module.

        Returns
        -------
        analysis : Analysis
            An analysis object that the non-linear search calls to determine the fit likelihood for a given model
            instance.
        """

        masked_dataset = md.MaskedDataset(dataset=dataset, mask=mask)

        return Analysis(
            masked_dataset=masked_dataset, image_path=self.optimizer.paths.image_path
        )

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
            output=result.output,
        )
