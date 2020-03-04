import autofit as af
from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.dataset.dataset import (
    Dataset,
)
from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.phase import tagging
from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.phase.meta_dataset import (
    MetaDataset,
)
from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.phase.analysis import (
    Analysis,
)

# The main addition to the phase in this tutorial is that we have added a 'phase setting' to the __init__ constructor
# called the 'signal_to_noise_limit'. This phase setting alters the dataset that is fitted by the phase, as well as
# the directory results are output.


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

        # Here, we create a 'tag' for our phase. Basically, if we use an optional phase setting to alter the dataset we
        # fit (here, a signal_to_noise_limit), we want to 'tag' the phase such that results are output to a unique
        # directory whose names makes it explicit how the dataset was changed.

        # If this setting is off, the tag is an empty string and thus the directory structure is not changed.

        phase_tag = tagging.phase_tag_from_phase_settings(
            signal_to_noise_limit=signal_to_noise_limit
        )
        paths.phase_tag = (
            phase_tag
        )  # The phase_tag must be manually added to the phase.

        super().__init__(paths=paths, optimizer_class=optimizer_class)

        self.gaussian = gaussian

        # Phase settings alter the dataset that is fitted, however a phase does not have access to the dataset until it
        # is run (e.g. the run method below is passed the dataset). In order for a phase to use its input phase
        # settings to create the dataset it fits, these settings are stored in the 'meta_dataset' attribute and used
        # when the 'run' and 'make_analysis' methods are called.

        self.meta_dataset = MetaDataset(signal_to_noise_limit=signal_to_noise_limit)

    def run(self, dataset: Dataset, mask):
        """
        Pass a dataset to the phase, running the phase and non-linear search.

        Parameters
        ----------
        dataset: aa.Dataset
            The dataset fitted by the phase, which in this case is a PyAutoArray dataset object.
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
            The dataset fitted by the phase, which in this case is a PyAutoArray dataset object.

        Returns
        -------
        analysis : Analysis
            An analysis object that the non-linear search calls to determine the fit likelihood for a given model
            instance.
        """

        # Here, the meta_dataset is used to create the masked dataset that is fitted. If the signal_to_noise_limit
        # setting is passed into the phase, the function below uses it to alter the masked dataset.

        # Checkout 'meta_dataset.py' for more details.

        masked_dataset = self.meta_dataset.masked_dataset_from_dataset_and_mask(
            dataset=dataset, mask=mask
        )

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
