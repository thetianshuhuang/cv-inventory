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
from kmeans import kmeans

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


def integral_in_range(data, left, right):
    total = 0.
    for i in range(left, right):
        total += data[i]
    return(total)


def find_targets(target, scene, roi_size):

    matches = sift_scene(
        target,
        scene,
        max_ratio=0.8, plot=True)

    means = kmeans(matches["scene"], 3, weights=matches["weight"])
    print(means)

    # Plot the data
    plt.imshow(matches["plot"])
    plt.show()

    for index, center in enumerate(means["means"]):

        roi = scene[
            int(center[1] - roi_size):int(center[1] + roi_size),
            int(center[0] - roi_size):int(center[0] + roi_size)]

        if(roi_size > 200):
            find_targets(target, roi, roi_size / 2)
        else:
            find_target(target, roi)


def find_target(target, scene):

    sigma = 100

    match_vectors = sift_scene(target, scene, ratio=1, plot=True)

    kernel_list = kernel_transform(match_vectors)

    center = gaussian_max(kernel_list, sigma)
    print("Gaussian Maximum: " + str(center))
    remove_outliers(kernel_list, 0.2)
    stats = weighted_stats(kernel_list)
    print("Weighted Stats:")
    print(stats)

    hist = weighted_bin(
        1. / sigma,
        kernel_list["data"],
        weights=kernel_list["weight"],
        epsilon=0)
    inrange = integral_in_range(
        hist,
        int(0.5 * sigma * center),
        int(1.5 * sigma * center))
    print(0.5 * sigma * center)
    print(1.5 * sigma * center)
    print("Total in range:")
    print(inrange)

    confidence = 1 - (1.0 / inrange)
    print("Confidence:")
    print(confidence)

    plt.imshow(match_vectors["plot"]), plt.show()

    #plt.hist(kernel_list["data"], weights=kernel_list["weight"], bins=100)
    #plt.show()

    #plt.plot(gaussian_convolve(hist, sigma))
    #plt.show()


if(__name__ == "__main__"):

    target = "reference/scene-wire/tx0.5.jpg"
    scene_pass = "reference/scene-wire/wide.jpg"
    scene_fail = "reference/scene-nano/sx1.jpg"

    images = load_weighted([target, scene_pass], 1)

    find_targets(images[0][0], images[1][0], 400)
    # find_targets(target, scene_fail)

    # images = load_weighted([target, scene], 1)

    # find_target(images[0][0], images[1][0])

    # find_target(
    #    "reference/scene-wire/tx0.5.jpg",
    #    "reference/scene-wire/rx0.125.jpg")
    # find_target(
    #    "reference/scene-wire/tx0.5.jpg",
    #    "reference/scene-nano/rx0.125.jpg")
