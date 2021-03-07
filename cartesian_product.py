import itertools


def cartesian_product(input_dict):
    """Generate cartesian product expansion for hyperparameters

    Args:
        input_dict (dict): dictionary with keys representing hyperparameters and a list of possible values

    Returns:
        list: list of dictionaries representing all possible sets of hyperparameters
    """
    
    return [dict(zip(input_dict.keys(), items)) for items in itertools.product(*input_dict.values())]
