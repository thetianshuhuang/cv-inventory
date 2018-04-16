#
# discrete.py
#
# tools for manipulation of discrete distributions
# Written by Tianshu Huang for cv-inventory
#
# Functions
# ---------
# gaussian_convolute: discrete convolution \int gaussian(x-y) * input[y] dx
# integral_in_range: integral of an input range between two indices
# weighted_bin: turn arrays of data and weights into a binned histogram
#

from math import pow, pi
from numpy import convolve, exp


def gaussian_convolve(input, sigma):

    """
    Compute a gaussian convolution with a given sigma value.

    Parameters
    ----------
    input : float[]
        Input array
    sigma : float
        Sigma value to use for the gaussian

    Returns
    -------
    float[]
        Computed convolution
    """

    gaussian = [
        (
            1 / (sigma * pow(2 * pi, 0.5)) *
            exp(-0.5 * ((x / sigma) ** 2))
        )
        for x in range(-sigma, sigma)]

    return(convolve(input, gaussian, mode="full"))


def integral_in_range(data, left, right):

    """
    Get the integral of the dataset data between indices left and right.

    Parameters
    ----------
    data : float[]
        Input data set
    left : int
        Left index
    right : int
        Right index

    Returns
    -------
    float
        Integral from left to right of data
    """

    total = 0.
    for i in range(left, right):
        total += data[i]
    return(total)


def weighted_bin(bin_width, data, **kwargs):

    """
    Split the data into discrete bins.

    Parameters
    ----------
    bin_width : float
        Width of each bin
    data : float[]
        Data array; all data must be positive
    weights= : float[]
        Weight array
    epsilon= : float
        All except epsilon of the data will be contained;
        the rest will be truncated

    Returns
    -------
    array
        Array containing the size of each bin, normalized a proportion of 1
    """

    # set up sorting differently depending on if weights are included
    if("weights" in kwargs):
        # create zip
        assert(len(kwargs["weights"]) == len(data))
        sortarray = zip(data, kwargs["weights"])
        # find total weight
        total_weight = 0
        for weight in kwargs["weights"]:
            total_weight += weight
    else:
        sortarray = zip(data, [1] * len(data))
        total_weight = len(data)

    # set up epsilon, if included; defaults to 0
    if("epsilon" in kwargs):
        epsilon = kwargs["epsilon"]
    else:
        epsilon = 0

    # sort lists
    data, weights = zip(*sorted(sortarray))

    current_weight = 0
    current_bin = 0
    current_index = 0
    output_array = []
    # loop through until all but epsilon are included
    while(current_index < len(data) and
          current_weight < total_weight * (1.0 - epsilon)):

        # process data until it overflows to the next bin
        current_value = 0
        while(current_index < len(data) and
              data[current_index] < current_bin + bin_width):

            current_value += weights[current_index]
            current_weight += weights[current_index]
            current_index += 1

        # append to result
        output_array.append(current_value)
        current_bin += bin_width

    return(output_array)
