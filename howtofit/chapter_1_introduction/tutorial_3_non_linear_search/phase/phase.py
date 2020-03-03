import autofit as af
from autofit.tools.phase import Dataset
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.phase.result import Result
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.phase.analysis import (
    Analysis,
)


class Phase(af.AbstractPhase):

    # This tells the phase that the input parameter 'gaussian' is a model component that is fitted for by the phase's
    # non-linear search.

    # In analysis.py, in the function 'fit' the input parameter 'instance' is a gaussian mapped from this model.

    gaussian = af.PhaseProperty("gaussian")

    Result = Result

    @af.convert_paths  # <- This handles setting up output paths.
    def __init__(
        self,
        paths,
        gaussian,
        optimizer_class=af.MultiNest,  # <- This specifies the default non-linear search used by the phase.
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
        super().__init__(paths=paths, optimizer_class=optimizer_class)
        self.gaussian = gaussian

    def run(self, dataset: Dataset):
        """
        Pass a dataset to the phase, running the phase and non-linear search.

        Parameters
        ----------
        dataset: aa.Imaging
            The dataset fitted by the phase, which in this case is a PyAutoArray imaging object.

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model.
        """

        analysis = self.make_analysis(dataset=dataset)

        result = self.run_analysis(analysis=analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset):
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
        return Analysis(dataset=dataset)

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            analysis=analysis,
        )
