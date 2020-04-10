from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.dataset import (
    dataset as ds
)

# The 'meta_dataset.py' module is unchanged from the previous tutorial.


class MetaDataset:
    def __init__(self, data_trim_left, data_trim_right):

        self.data_trim_left = data_trim_left
        self.data_trim_right = data_trim_right

    def masked_dataset_from_dataset_and_mask(self, dataset, mask):

        masked_dataset = ds.MaskedDataset(dataset=dataset, mask=mask)

        if self.data_trim_left is not None:
            masked_dataset = masked_dataset.with_left_trimmed(
                data_trim_left=self.data_trim_left
            )

        if self.data_trim_right is not None:
            masked_dataset = masked_dataset.with_right_trimmed(
                data_trim_right=self.data_trim_right
            )

        return masked_dataset
