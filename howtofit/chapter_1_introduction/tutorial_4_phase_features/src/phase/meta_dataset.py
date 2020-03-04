from howtofit.chapter_1_introduction.tutorial_4_phase_features.src.dataset import (
    masked_dataset as md,
)

# In this tutorial, we'll run phases where the dataset we input into the phase is altered before the model-fitting
# procedure is run. Details of how the dataset is changed are given in the tutorial, but essentially the noise-map
# is reweighted to limit the signal-to-noise of pixels in the dataset to a maximum value.

# The 'meta_dataset.py' module is the module in PyAutoFit which handles the creation of these new datasets, as well as
# handling the masking of a dataset during its creation in the 'make_analysis' method. Essenetially, if we want to
# have the option of a phase editing the data-set, the parameters which control this (e.g. 'signal_to_noise_limit')
# are stored here and then used when the 'phase.run' method is called.

# Your model-fitting problem may not require the meta_dataset.py module and you may feel that masking can be handled
# entirely within the 'analysis.py' module. If so that is fine - it really depends on the nature of your problem.
# However, I've included the 'meta_dataset' module in all tutorials from here on, as those with a more complicated
# model fitting problem will find it useful.


class MetaDataset:

    # The signal_to_noise_limit is passed to the phase when it is set up and then subsequent passed and stored in an
    # instance of the 'MetaDataset' class.

    def __init__(self, signal_to_noise_limit=None):

        self.signal_to_noise_limit = signal_to_noise_limit

    # The masked dataset that is fitted by an analysis is created by the MetaDataset class, as shown below, by being
    # passed a dataset and mask. if the MetaDataset has a signal_to_noise_limit it is then used to apply a S/N limit
    # to the masked dataset that is fitted.

    def masked_dataset_from_dataset_and_mask(self, dataset, mask):

        masked_dataset = md.MaskedDataset(dataset=dataset, mask=mask)

        if self.signal_to_noise_limit is not None:
            masked_dataset = masked_dataset.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        return masked_dataset
