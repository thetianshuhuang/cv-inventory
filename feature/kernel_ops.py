#
# kernel_ops.py
#
# kernel operations for the feature matching algorithm
#


def weighted_stats(data):

    """
    Compute several statistics for a weighted data list.
    - Weighted mean
    - Weighted variance of a dataset:
    $$Var(data) = E(X^2) + E(X)^2
        = \frac{\sum x^2 \alpha_x}{\sum \alpha_x} +
          \left(\frac{\sum x \alpha_x}{\sum \alpha_x}\right)^2$$

    Parameters
    ----------
    data : array
        List of [value, confidence].

    Returns
    -------
    {"mean": float, "var": float}
        - Mean; units = scale (unitless)
        - Weighted variance; units = px^2
    """

    # compute required terms
    mean = 0
    weighted_square = 0
    total_weight = 0
    for value in data:
        # update weighted sum ($\sum x \alpha_x$)
        mean += value[0] * value[1]
        # update weighted sum of squares ($\sum x^2 \alpha_x$)
        weighted_square += value[0]**2 * value[1]
        # update total weight
        total_weight += value[1]

    # return stats
    return({
        "mean": mean,
        "var": weighted_square / total_weight + (mean / total_weight)**2
    })


def vector_median(data):

    """
    Compute the vector median of weighted data matrix data, where the last
    element of each vector in the data matrix is the weight.

    Parameters
    ----------
    data : array
        Data matrix. Each element is in the form:
            $$[x_0, x_1, [...] x_n, \alpha]$$
        Where $\alpha$ is the weight.

    Returns
    -------
    array
        Weighted median vector
    """

    # We assume that the data matrix is rectangular
    dimension = len(data[0]) - 1
    median_vector = []

    # Get total weight
    total_weight = 0
    for vector in data:
        total_weight += data[-1]

    # iterate over every column
    for i in range(dimension):

        # assemble a combined list indexed by the data
        data_list = [[x[i], x[-1]] for x in data]
        data_list.sort()

        # add weight until the median is reached
        cumulative_weight = 0
        j = 0
        while(cumulative_weight < total_weight / 2):
            cumulative_weight += data_list[j][0]
            j += 1

        # append median
        median_vector.append(data_list[j][0])

    return(median_vector)


def kernel_list(data, cutoff):

    """
    Generate the kernel map of a given input vector for the feature
    matching tracker, and output as a list with confidence cutoff.

    Parameters
    ----------
    data : 2-dimensional array
        input data array. Each element contains 5 values:
            $(x_0, y_0, x_1, y_1, \alpha)$
        Where $(x_0, y_0)$ is the source vector, $(x_1, y_1)$ is the
        destination vector, and $\alpha$ is the confidence value for the
        match.
    cutoff : float $\in (0,1)$
        Confidence cutoff value; if $\alpha_i\alpha_j < cutoff$, the point is
        dicarded.

    Returns
    -------
    array
        1-dimensional weighted kernel map with accompanying confidences.
        For data vectors $i$ and $j$:
            $K(i,j) = d(i_1, j_1) / d(i_0, j_0)$
        where $i_0 = (x_0, y_0)$ and similar for $i_1$, $j_0$, $j_1$.
    """

    kernel = []

    # iterate over all unique pairs, except for identity pairs
    for i in range(len(data)):

        for j in range(i + 1, len(data)):

            confidence = data[i][4] * data[j][4]

            if(confidence > cutoff):
                kernel.append([
                    ((data[i][2] - data[j][2])**2 +
                        (data[i][3] - data[j][3])**2) /
                    ((data[i][0] - data[j][0])**2 +
                        (data[i][1] - data[j][1])**2),
                    confidence
                ])

    return(kernel)
