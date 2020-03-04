import autoarray as aa

# Here, we create masked dataset that will be fitted by our phase. Essentially, all this class does it takes an
# unmasked dataset (e.g. an image and noise map) and applies to a mask to them, such that all entries where the mask is
# True are omitted from the fit and likelihood calution.

# If your model-fitting problem requires masking you'll probably want a module something very similar to this one!


class MaskedDataset:
    def __init__(self, dataset, mask):
        """
        A masked dataset, which is an image, noise-map and mask.

        Parameters
        ----------
        dataset: im.Dataset
            The dataset data_type all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        """

        # We store the unmasked dataset in the masked-dataset, incase we need it for anything.
        self.dataset = dataset

        self.mask = mask

        # In PyAutoAray, we create masked array objects using the functions below. This is probably more complicated
        # than necessary for your model-fitting problem, so if you think masking is something that is necessary think
        # about the most straight forward way that you can mask you data. Perhaps just multiplying it by the mask?

        # We apply the mask but multiplying the unmasked data with it.
        self.data = mask.mapping.array_stored_2d_from_sub_array_2d(
            sub_array_2d=dataset.data
        ).in_2d

        # Same for the noise-map
        self.noise_map = mask.mapping.array_stored_2d_from_sub_array_2d(
            sub_array_2d=dataset.noise_map
        ).in_2d

        # We'll need the masked grid to compute the model Gaussian image.
        self.grid = aa.grid.from_mask(mask=mask)

    @property
    def image(self):
        return self.data

    def signal_to_noise_map(self):
        return self.image / self.noise_map

    # Given a masked data-set, this method created a new masked dataset where the noise-map is scaled such that the
    # signal-to-noise ratio in every pixel does not exceed an input value. This means we can create an altered dataset
    # we may then fit in a phase.

    # In this tutorial, we'll show how such altered datasets can be created and fitted in PyAutoFit. This dataset may be
    # created for a phase in the 'meta_dataset.py' package.

    def signal_to_noise_limited_from_signal_to_noise_limit(self, signal_to_noise_limit):

        dataset_with_signal_to_noise_limit = self.dataset.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=signal_to_noise_limit
        )

        return self.__class__(
            dataset=dataset_with_signal_to_noise_limit, mask=self.mask
        )
