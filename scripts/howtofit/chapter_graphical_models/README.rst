In this chapter, we take you through how to compose and fit graphical models in **PyAutoFit**. Graphical models
simultaneously fit many datasets with a model that has 'local' parameters specific to each individual dataset
and 'global' parameters that fit for global trends across the whole dataset.

**Binder** links to every tutorial are included.

Files
-----

The chapter contains the following tutorials:

`Tutorial 1: Individual Models <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_1_individual_models.ipynb>`_
- An example of inferring global parameters from a dataset by fitting the model to each individual dataset one-by-one.

`Tutorial 2: Graphical Model <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_2_graphical_model.ipynb>`_
- Fitting the dataset with a graphical model that fits all datasets simultaneously to infer the global parameters.

`Tutorial 3: Graphical Benefits <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_3_graphical_benefits.ipynb>`_
- Illustrating the benefits of graphical modeling over simpler approaches using a more complex model.

`Tutorial 4: Hierarchical Models <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_4_hierarchical_models.ipynb>`_
- Fitting hierarchical models using the graphical modeling framework.

`Tutorial 5: Expectation Propagation <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_5_expectation_propagation.ipynb>`_
- Scaling graphical models up to fit extremely large datasets using Expectation Propagation (EP).