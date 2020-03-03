import autoarray as aa
from autoarray.masked import masked_dataset


class MetaDataset:

    def __init__(
        self, signal_to_noise_limit=None,
    ):

        self.signal_to_noise_limit = signal_to_noise_limit

    def masked_dataset_from_dataset_and_mask(self, dataset, mask):

        masked_imaging = masked_dataset.MaskedImaging(
            imaging=dataset, mask=mask
        )

        if self.signal_to_noise_limit is not None:
            masked_imaging = masked_imaging.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit,
            )

        return masked_imaging
