#
# feature.py
#
# feature_tracker.py
# feature identification based tracker
#

from cv_img import load_weighted
from cv_feature import sift_scene
from matplotlib import pyplot as plt
from kernel_ops import weighted_stats, kernel_transform, remove_outliers
from discrete import gaussian_convolve, weighted_bin
import numpy as np


def gaussian_max(data, sigma):
    size = np.argmax(
        gaussian_convolve(
            weighted_bin(
                1. / sigma, data["data"],
                weights=data["weight"], epsilon=0),
            sigma)
    )
    return(1.0 * (size - sigma) / sigma)


def find_target(target, scene):

    sigma = 100

    images = load_weighted([target, scene], 1)

    match_vectors = sift_scene(
        images[0][0],
        images[1][0],
        0.8, plot=True)

    kernel_list = kernel_transform(match_vectors)

    print("Gaussian Maximum: " + str(gaussian_max(kernel_list, sigma)))
    remove_outliers(kernel_list, 0.2)
    print("Weighted Stats:")
    print(weighted_stats(kernel_list))

    plt.imshow(match_vectors["plot"]), plt.show()

    plt.hist(kernel_list["data"], weights=kernel_list["weight"], bins=100)
    plt.show()

    plt.plot(gaussian_convolve(
        weighted_bin(
            1. / sigma,
            kernel_list["data"],
            weights=kernel_list["weight"],
            epsilon=0),
        sigma))
    plt.show()


if(__name__ == "__main__"):

    find_target(
        "reference/scene-wire/tx0.5.jpg",
        "reference/scene-wire/wide.jpg")
    find_target(
        "reference/scene-wire/tx0.5.jpg",
        "reference/scene-nano/sx1.jpg")
