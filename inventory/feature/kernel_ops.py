#
# kernel_ops.py
#
# kernel operations for a keypoint-based matching algorithm
# Written by Tianshu Huang for cv-inventory, April 2018
#
# Functions
# ---------
# weighted_stats: compute weighted stats for a datalist
# kernel_transform: get the weighted distance kernel map
# remove_outliers: remove the largest epsilon outliers from the dataset
# clustering_stats: compute clustering related statistics
#

import numpy as np
from discrete import gaussian_convolve, weighted_bin, integral_in_range


# -----------------------------------------------------------------------------
#
# weighted_stats
#
# -----------------------------------------------------------------------------
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
        "var": weighted_square / total_weight - mean**2,
        "total": total_weight
    })


# -----------------------------------------------------------------------------
#
# kernel_transform
#
# -----------------------------------------------------------------------------
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
            a_duplicate = (
                np.array_equal(data["scene"][i], data["scene"][j]) or
                np.array_equal(data["target"][i], data["target"][j]))

            if(not a_duplicate):
                # take the kernel map
                dist_ratio = (
                    np.linalg.norm(data["scene"][i] - data["scene"][j]) /
                    np.linalg.norm(data["target"][i] - data["target"][j])
                )

                kernel_map["data"].append(dist_ratio)
                kernel_map["weight"].append(
                    data["weight"][i] * data["weight"][j])

    return(kernel_map)


# -----------------------------------------------------------------------------
#
# remove_outliers
#
# -----------------------------------------------------------------------------
def remove_outliers(dataset, epsilon):

    """
    Remove the epsilon greatest outliers from an input array.

    Parameters
    ----------
    dataset : dict, with entries:
        data: float[], input dataset
        weights: float[], weight of each datapoint
    epsilon : amount of data to remove

    Returns
    -------
    dict
        Dataset, with outliers removed
    """

    data, weights = zip(*sorted(zip(dataset["data"], dataset["weight"])))

    # get total weight
    total_weight = 0
    for weight in weights:
        total_weight += weight

    # iterate until only epsilon remains
    current_weight = 0
    index = 0
    while(current_weight < total_weight * (1 - epsilon)):
        current_weight += weights[index]
        index += 1

    # rebuild dictionary
    dataset["data"] = data[:index]
    dataset["weight"] = weights[:index]
    return(dataset)


# -----------------------------------------------------------------------------
#
# clustering_stats
#
# -----------------------------------------------------------------------------
def clustering_stats(data, sigma):

    """
    Compute statistics related to clustering in a dataset.

    Parameters
    ----------
    data : dict
        Input dataset
    sigma: float
        Coefficient to use

    Returns
    -------
    dict, with entries:
        "center": gaussian maximum with coefficient sigma
        "in_range": integral from center/2 to 3*center/3
    """

    hist = weighted_bin(
        1. / sigma, data["data"], weights=data["weight"], epsilon=0)

    center = 1.0 * (np.argmax(gaussian_convolve(hist, sigma)) - sigma) / sigma
    in_range = integral_in_range(
        hist,
        int(0.5 * sigma * center),
        int(1.5 * sigma * center))

    return({
        "center": center,
        "in_range": in_range,
    })
