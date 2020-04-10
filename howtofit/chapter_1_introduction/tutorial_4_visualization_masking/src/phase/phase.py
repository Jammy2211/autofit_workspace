import autofit as af
from howtofit.chapter_1_introduction.tutorial_4_visualization_masking.src.dataset.dataset import (
    Dataset, MaskedDataset
)
from howtofit.chapter_1_introduction.tutorial_4_visualization_masking.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_4_visualization_masking.src.phase.analysis import (
    Analysis,
)


# The Phase class __init__ method is unchanged from the previous tutorial, however other methods are.


class Phase(af.AbstractPhase):

    gaussian = af.PhaseProperty("gaussian")

    Result = Result

    @af.convert_paths
    def __init__(self, paths, gaussian, non_linear_class=af.MultiNest):
        """
        A phase which fits a Gaussian model using a non-linear search.

        Parameters
        ----------
        paths : af.Paths
            Handles the output directory structure.
        gaussian : gaussians.Gaussian
            The model component Gaussian class fitted by this phase.
        non_linear_class: class
            The class of a non_linear optimizer
        """

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.gaussian = gaussian

    # The run method is slightly different, as it now passed a mask in addition to the dataset. These are used to set up
    # the masked-dataset in the 'analysis.py' module.

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

        # To mask the dataset we simply pass both to the MaskedDataset class.

        masked_dataset = MaskedDataset(dataset=dataset, mask=mask)

        # The 'image_path' is where visualizatiion of the model fit is output. Below, we direct it to the same path as
        # the non-linear search output, but with an additional folder 'image' at the end. This path should be used
        # for pretty much any project.

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
