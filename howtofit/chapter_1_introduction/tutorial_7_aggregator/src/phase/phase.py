import autofit as af
from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.dataset.dataset import (
    Dataset,
)
from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.phase.analysis import (
    Analysis,
)
from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.phase.meta_dataset import (
    MetaDataset,
)
from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.phase import (
    tagging,
)


# The 'phase.py' module is mostly unchanged from the previous tutorial, however the 'run' function has been updated.


class Phase(af.AbstractPhase):

    profiles = af.PhaseProperty("profiles")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        profiles,
        data_trim_left=None,
        data_trim_right=None,
        non_linear_class=af.MultiNest,
    ):
        """
        A phase which fits a model composed of multiple line profiles (Gaussian, Exponential) using a non-linear search.

        Parameters
        ----------
        paths : af.Paths
            Handles the output directory structure.
        profiles : [profiles.Profile]
            The model components (e.g. Gaussian, Exponenial) fitted by this phase.
        non_linear_class: class
            The class of a non_linear optimizer
        data_trim_left : int or None
            The number of pixels by which the data is trimmed from the left-hand side.
        data_trim_right : int or None
            The number of pixels by which the data is trimmed from the right-hand side.
        """

        phase_tag = tagging.phase_tag_from_phase_settings(
            data_trim_left=data_trim_left, data_trim_right=data_trim_right
        )
        paths.phase_tag = (
            phase_tag
        )  # The phase_tag must be manually added to the phase.

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.profiles = profiles

        self.meta_dataset = MetaDataset(
            data_trim_left=data_trim_left, data_trim_right=data_trim_right
        )

    @property
    def phase_folders(self):
        return self.optimizer.phase_folders

    def run(self, dataset: Dataset, mask, info=None):
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

        # These functions save the objects we will later access using the aggregator. They are saved via the 'pickle'
        # module in Python, which serializes the data on to the hard-disk.

        # See the 'dataset.py' module for a description of what the metadata is.

        self.save_metadata(dataset=dataset)
        self.save_dataset(dataset=dataset)
        self.save_mask(mask=mask)
        self.save_meta_dataset(meta_dataset=self.meta_dataset)
        self.save_info(info=info)

        # This saves the optimizer information of the phase, meaning that we can use the non_linear_class instance
        # (e.g. MultiNest) to interpret our results in the aggregator.

        self.assert_and_save_pickle()

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
            An analysis object that the non-linear search calls to determine the fit log_likelihood for a given model
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
            log_likelihood=result.log_likelihood,
            analysis=analysis,
            output=result.output,
        )
