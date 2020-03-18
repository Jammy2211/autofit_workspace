from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.dataset import (
    masked_dataset as md,
)

# In this tutorial, we run phases where the dataset we input into the phase is altered before the model-fitting
# procedure is run. The dataset is trimmed by an input number of pixeles to the left and / or right.

# The 'meta_dataset.py' module is the module in PyAutoFit which handles the creation of these new datasets, If we want
# to have the option of a phase editing the data-set, the parameters which control this (e.g. 'data_trim_left')
# are stored here and then used when the 'phase.run' method is called.

# Your model-fitting problem may not require the meta_dataset.py module. If so that is fine, and you can revert to the
# templates of the previous tutorials which do not use one. It really depends on the nature of your problem.


class MetaDataset:

    # The data_trim_left and data_trim_right are passed to the phase when it is set up and stored in an
    # instance of the 'MetaDataset' class.

    def __init__(self, data_trim_left, data_trim_right):

        self.data_trim_left = data_trim_left
        self.data_trim_right = data_trim_right

    # The masked dataset that is fitted by an analysis is created by the MetaDataset class using the method below

    # If the MetaDataset's data trimm attributes are not Nonoe, they are used to trim the masked-dataset before it is
    # fitted.

    def masked_dataset_from_dataset_and_mask(self, dataset, mask):

        masked_dataset = md.MaskedDataset(dataset=dataset, mask=mask)

        if self.data_trim_left is not None:
            masked_dataset = masked_dataset.with_left_trimmed(
                data_trim_left=self.data_trim_left
            )

        if self.data_trim_right is not None:
            masked_dataset = masked_dataset.with_right_trimmed(
                data_trim_right=self.data_trim_right
            )

        return masked_dataset
