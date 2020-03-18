# The 'tagging.py' module is unchanged from the previous tutorial.


def phase_tag_from_phase_settings(data_trim_left=None, data_trim_right=None):

    data_trim_left_tag = data_trim_left_tag_from_data_trim_left(
        data_trim_left=data_trim_left
    )

    data_trim_right_tag = data_trim_right_tag_from_data_trim_right(
        data_trim_right=data_trim_right
    )

    return "phase_tag" + data_trim_left_tag + data_trim_right_tag


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
