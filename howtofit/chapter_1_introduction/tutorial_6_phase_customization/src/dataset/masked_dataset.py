import numpy as np

from howtofit.chapter_1_introduction.tutorial_6_phase_customization.src.dataset import (
    dataset as ds,
)

# This module is unchanged from tutorial, except two new methods for creating trimmed data-sets are now included.


class MaskedDataset:
    def __init__(self, dataset, mask):
        """
        A masked dataset, which is an image, noise-map and mask.

        Parameters
        ----------
        dataset: im.Dataset
            The dataset (the image, noise-map, etc.)
        mask: msk.Mask
            The 1D mask that is applied to the dataset.
        """

        self.dataset = dataset
        self.mask = mask
        self.data = dataset.data * np.invert(mask)
        self.noise_map = dataset.noise_map * np.invert(mask)

    @property
    def xvalues(self):
        return np.arange(self.data.shape[0])

    def signal_to_noise_map(self):
        return self.data / self.noise_map

    def with_left_trimmed(self, data_trim_left):

        # Here, we use the existing masked dataset to create a trimmed dataset.

        data_trimmed = self.dataset.data[data_trim_left:]
        noise_map_trimmed = self.dataset.noise_map[data_trim_left:]

        dataset_trimmed = ds.Dataset(data=data_trimmed, noise_map=noise_map_trimmed)

        mask_trimmed = self.mask[data_trim_left:]

        return MaskedDataset(dataset=dataset_trimmed, mask=mask_trimmed)

    def with_right_trimmed(self, data_trim_right):

        # We do the same as above, but removing data to the right.

        data_trimmed = self.dataset.data[:-data_trim_right]
        noise_map_trimmed = self.dataset.noise_map[:-data_trim_right]

        dataset_trimmed = ds.Dataset(data=data_trimmed, noise_map=noise_map_trimmed)

        mask_trimmed = self.mask[:-data_trim_right]

        return MaskedDataset(dataset=dataset_trimmed, mask=mask_trimmed)
