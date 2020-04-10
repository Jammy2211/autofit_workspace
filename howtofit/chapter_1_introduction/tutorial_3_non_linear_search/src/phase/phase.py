import autofit as af
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.dataset.dataset import (
    Dataset,
)
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_3_non_linear_search.src.phase.analysis import (
    Analysis,
)

# The phase package combines a data-set, model and non-linear search, allowing us to fit the dataset with the model. It
# essentially acts as the 'meeting point' between the modules we used in previous tutorials (dataset, fit, plot)
# and modules in the phase package (phase.py, analysis.py, result.py).


class Phase(af.AbstractPhase):

    # This tells the phase that the input parameter 'gaussian' is a model component that is fitted for by the phase's
    # non-linear search.

    # In 'analysis.py', the function 'fit' has an input parameter called 'instance' which is the gaussian mapped from
    # this model via a unit vector and the model priors (as described in tutorial 1).

    # For your model-fitting problem, this will be replaced by the classes in your 'model' package.

    gaussian = af.PhaseProperty("gaussian")

    Result = Result  # Set the result to the Result class in 'result.py'

    @af.convert_paths  # <- This handles setting up output paths.
    def __init__(
        self,
        paths,
        gaussian,  # <- The user inputs a model -> gaussian.py -> Gaussian class here.
        non_linear_class=af.MultiNest,  # <- This specifies the default non-linear search used by the phase.
    ):
        """
        A phase which fits a Gaussian model using a non-linear search.

        Parameters
        ----------
        paths : af.Paths
            Handles the output directory structure.
        gaussian : model.gaussians.Gaussian
            The model component Gaussian class fitted by this phase.
        non_linear_class: class
            The class of a non_linear optimizer
        """
        super().__init__(paths=paths, non_linear_class=non_linear_class)
        self.gaussian = gaussian

    def run(self, dataset: Dataset):
        """ Pass a dataset to the phase, running the phase and non-linear search.

        Parameters
        ----------
        dataset : dataset.Dataset
            The dataset fitted by the phase, which is specified in the module 'dataset/dataset.py'

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model.
        """

        # These functions create instances of the Analysis class (in 'analysis.py'), runs the analysis (which performs
        # the non-linear search ) and returns an instance of the Result class (in 'result.py').

        # Once you've looked through this module, check those modules out to see exactly what these classes do!

        analysis = self.make_analysis(dataset=dataset)

        # 'run_analysis' is not located in analysis.py, instead it is an inherited method from the parent class
        # 'af.AbstractPhase'. Essentially, all this function does is begin the non-linear search, using the analysis
        # created above.

        result = self.run_analysis(analysis=analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset):
        """
        Create an Analysis object, which uses the dataset with functions to perform a fit.

        Parameters
        ----------
        dataset : dataset.Dataset
            The dataset fitted by the phase, which is specified in the module 'dataset/dataset.py'

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
            output=result.output,
        )
