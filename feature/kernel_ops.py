#
# kernel_ops.py
#
# kernel operations for a keypoint-based matching algorithm
# Written by Tianshu Huang for cv-inventory, April 2018
#
# Functions
# ---------
# distance: norm between two vectors
# weighted_stats: compute weighted stats for a datalist
# kernel_transform: get the weighted distance kernel map
#

from numpy.linalg import norm


def distance(x_1, x_2, **kwargs):

    """
    Compute the distance between two vectors.

    Parameters
    ----------
    x_1, x_2 : float[]
        Input vectors
    ird- : float
        Norm to compute; passed on to norm.

    Returns
    -------
    float
        Computed norm
    """

    assert len(x_1) == len(x_2)

    difference = []
    for i in range(len(x_1)):
        difference.append(abs(x_1[i] - x_2[i]))

    return(norm(difference, **kwargs))


def weighted_stats(data):

    """
    Compute several statistics for a weighted data list.
    - Weighted mean
    - Weighted variance of a dataset:
    $$Var(data) = E[(X - \mu)^2] = E(X^2) + E(X)^2
        = \frac{\sum x^2 \alpha_x}{\sum \alpha_x} +
          \left(\frac{\sum x \alpha_x}{\sum \alpha_x}\right)^2$$

    Parameters
    ----------
    data : dict, with entries:
        "data", float[]: input data
        "weight", float[]: data weight

    Returns
    -------
    dict, with entries:
        "mean": float
        "var": float
    """

    assert len(data["data"]) == len(data["weight"])

    mean = 0
    weighted_square = 0
    total_weight = 0
    for i in range(len(data["data"])):
        # update weighted sum
        mean += data["data"][i] * data["weight"][i]
        # update weighted sum of squares
        weighted_square += data["data"][i]**2 * data["weight"][i]
        # update total weight
        total_weight += data["weight"][i]

    if(total_weight == 0):
        return({"mean": 0, "var": 0})

    mean = mean / total_weight

    return({
        "mean": mean,
        "var": weighted_square / total_weight + mean**2,
        "total": total_weight
    })


def kernel_transform(data):

    """
    Get the weighted distance kernel between keypoint matches.

    Parameters
    ----------
    data: dict, with entries:
        "target": float[*][2] of target keypoint coordinates
        "scene": float[*][2] of scene keypoint coordinates
        "weight": confidence values
        "length": number of entries

    Returns
    -------
    dict, with entries:
        "data":
            Array containing the kernel mapping d(x_0, x_1) with
            confidence c_1 * c_2
        "weight":
            Array containing weights of each value
    """

    kernel_map = {"data": [], "weight": []}
    for i in range(data["length"]):
        for j in range(i + 1, data["length"]):
            # filter out points mapped to multiple other points
            if(data["scene"][i] != data["scene"][j] and
               data["target"][i] != data["target"][j]):

                # take the kernel map
                dist_ratio = (
                    distance(data["scene"][i], data["scene"][j]) /
                    distance(data["target"][i], data["target"][j])
                )

                kernel_map["data"].append(dist_ratio)
                kernel_map["weight"].append(
                    data["weight"][i] * data["weight"][j])

    return(kernel_map)


def remove_outliers(dataset, epsilon):

    data, weights = zip(*sorted(zip(dataset["data"], dataset["weight"])))

    total_weight = 0
    for weight in weights:
        total_weight += weight

    current_weight = 0
    index = 0
    while(current_weight < total_weight * (1 - epsilon)):
        current_weight += weights[index]
        index += 1
    dataset["data"] = data[:index]
    dataset["weight"] = weights[:index]
