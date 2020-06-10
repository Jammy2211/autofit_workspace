import autofit as af


class Pipeline(af.Pipeline):
    def run(self, dataset, mask, info=None):
        def runner(phase, results):
            return phase.run(dataset=dataset, results=results, mask=mask, info=info)

        return self.run_function(runner)


from howtofit.chapter_2_non_linear_searches.src.phase import phase as ph
from howtofit.chapter_2_non_linear_searches.src.model import profiles


def make_pipeline():

    pipeline_name = "pipeline__x2_gaussians"

    """
    Phase 1:
     
    Fit the Gaussian on the left.
    """

    phase1 = ph.Phase(
        phase_name="phase_1__left_gaussian",
        profiles=af.CollectionPriorModel(gaussian_0=profiles.Gaussian),
        non_linear_class=af.PySwarmsGlobal,
    )

    """
    Phase 2: 
    
    Fit the Gaussian on the right, where the best-fit Gaussian resulting from phase 1 above fits the left-hand Gaussian.
    """

    phase2 = ph.Phase(
        phase_name="phase_2__right_gaussian",
        profiles=af.CollectionPriorModel(
            gaussian_0=phase1.result.instance.profiles.gaussian_0,  # <- Use the Gaussian fitted in phase 1
            gaussian_1=profiles.Gaussian,
        ),
        non_linear_class=af.PySwarmsGlobal,
    )

    """
    Phase 3:
     
    Fit both Gaussians, using the results of phases 1 and 2 to initialize their model parameters.
    """

    phase3 = ph.Phase(
        phase_name="phase_3__both_gaussian",
        profiles=af.CollectionPriorModel(
            gaussian_0=phase1.result.model.profiles.gaussian_0,  # <- use phase 1 Gaussian results.
            gaussian_1=phase2.result.model.profiles.gaussian_1,  # <- use phase 2 Gaussian results.
        ),
        non_linear_class=af.DynestyStatic,
    )

    return Pipeline(pipeline_name, phase1, phase2, phase3)
