# This module create tags for phases settings that customize the analysis. We tag phases for two reasons:

# 1) Tags describes the phase settings, making it explicit what analysis was used to create the results.

# 2) Tags create unique output paths, ensuring that if you run multiple phases on the same data with different settings
#    each non-linear search (e.g. MultiNest) won't inadvertantly use results generated via a different analysis method.


def phase_tag_from_phase_settings(data_trim_left=None, data_trim_right=None):

    data_trim_left_tag = data_trim_left_tag_from_data_trim_left(
        data_trim_left=data_trim_left
    )

    data_trim_right_tag = data_trim_right_tag_from_data_trim_right(
        data_trim_right=data_trim_right
    )

    # You may well have many more tags which appear here.

    # your_own_tag = your_own_tag_from_your_own_setting(you_own_setting=you_own_setting)

    return (
        "phase_tag"  # For every tag you add, you'll add it to this return statement
        + data_trim_left_tag
        + data_trim_right_tag
        # e.g. + your_own_tag
    )


# This function generates a string we'll use to 'tag' a phase which uses this setting, thus ensuring results are
# output to a unique path.


def data_trim_left_tag_from_data_trim_left(data_trim_left):
    """Generate a data trim left tag, to customize phase names based on how much of the dataset is trimmed to its left.

    This changes the phase name 'phase_tag' as follows:

    data_trim_left = None -> phase_tag
    data_trim_left = 2 -> phase_tag__trim_left_2
    data_trim_left = 10 -> phase_tag__trim_left_10
    """
    if data_trim_left is None:
        return ""
    else:
        return "__trim_left_" + str(data_trim_left)


def data_trim_right_tag_from_data_trim_right(data_trim_right):
    """Generate a data trim right tag, to customize phase names based on how much of the dataset is trimmed to its right.

    This changes the phase name 'phase_tag' as follows:

    data_trim_right = None -> phase_tag
    data_trim_right = 2 -> phase_tag__trim_right_2
    data_trim_right = 10 -> phase_tag__trim_right_10
    """
    if data_trim_right is None:
        return ""
    else:
        return "__trim_right_" + str(data_trim_right)
