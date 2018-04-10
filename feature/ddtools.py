#
# ddtools.py
#
# tools for manipulation of discrete distributions
# Written by Tianshu Huang for the Tracker Synergy project and the Computer
# Vision Inventory Management system
#
# Functions
# ---------
# dnorm: discrete Lp norm ||x||_p
# ddistance: distance on a vector space ||x-y||_p
# gaussian_convolute: discrete convolution \int gaussian(x-y) * input[y] dx
#

from math import pow, pi
from numpy import convolve, exp


def dnorm(vector, **kwargs):

    """
    Compute the discrete Lp norm of a vector.

    Parameters
    ----------
    vector : float[]
        Input vector
    lp= : float
        Norm to compute. Defaults to 2

    Returns
    -------
    float
        Computed norm
    """

    if("lp" in kwargs):
        lp = kwargs["lp"]
    else:
        lp = 2

    norm = 0
    for coord in vector:
        norm += pow(abs(coord), lp)

    return(pow(norm, 1.0 / lp))


def ddistance(x_1, x_2, **kwargs):

    """
    Compute the discrete Lp distance between two vectors.

    Parameters
    ----------
    x_1, x_2 : float[]
        Input vectors
    lp= : float
        Norm to compute; passed on to dnorm.

    Returns
    -------
    float
        Computed norm
    """

    assert len(x_1) == len(x_2)

    difference = []
    for i in range(len(x_1)):
        difference.append(abs(x_1[i] - x_2[i]))

    return(dnorm(difference, **kwargs))


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
        Array containing the size of each bin.
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
