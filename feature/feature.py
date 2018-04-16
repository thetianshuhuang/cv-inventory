#
# feature.py
#
# feature_tracker.py
# feature identification based tracker
#

from cv_img import load_weighted
from cv_feature import sift_scene
from matplotlib import pyplot as plt
from kernel_ops import weighted_stats, kernel_transform, remove_outliers, clustering_stats
from discrete import gaussian_convolve, weighted_bin
from kmeans import kmeans

import numpy as np


def is_match(target, scene, **kwargs):

    # Assign default settings
    if("debug" in kwargs and kwargs["debug"]):
        debug_plot = True
    else:
        debug_plot = False
    if("sigma" in kwargs):
        sigma = kwargs["sigma"]
    else:
        sigma = 100
    if("max_ratio" in kwargs):
        max_ratio = kwargs["max_ratio"]
    else:
        max_ratio = 1.0
    if("outliers" in kwargs):
        outliers = kwargs["epsilon"]
    else:
        outliers = 0.2

    # find matches
    matches = sift_scene(target, scene, max_ratio=max_ratio, plot=debug_plot)

    # get kernel
    kernel_list = kernel_transform(matches)
    remove_outliers(kernel_list, outliers)

    # compute statistics
    stats = weighted_stats(kernel_list)
    stats.update(clustering_stats(kernel_list, sigma))

    # Show debug plots
    if(debug_plot):
        # Image
        plt.imshow(matches["plot"]), plt.show()
        # Raw histogram
        plt.hist(kernel_list["data"], weights=kernel_list["weight"], bins=100)
        plt.show()
        # Gaussian convolve
        plt.plot(
            gaussian_convolve(
                weighted_bin(
                    1. / sigma,
                    kernel_list["data"],
                    weights=kernel_list["weight"],
                    epsilon=0),
                sigma))
        plt.show()

    return(stats)


def find_targets(target, scene, roi_size):

    matches = sift_scene(
        target,
        scene,
        max_ratio=0.8, plot=True)

    means = kmeans(matches["scene"], 3, weights=matches["weight"])
    print(means["weights"])
    print(means["means"])

    # Plot the data
    plt.imshow(matches["plot"])
    plt.show()

    center = means["means"][0]

    roi = scene[
        int(center[1] - roi_size):int(center[1] + roi_size),
        int(center[0] - roi_size):int(center[0] + roi_size)]

    if(roi_size > 200):
        find_targets(target, roi, roi_size / 2)
    else:
        is_match(target, roi)


if(__name__ == "__main__"):

    target = "reference/scene-wire/tx0.5.jpg"
    scene_pass = "reference/scene-wire/wide.jpg"
    scene_fail = "reference/scene-nano/sx1.jpg"

    images = load_weighted([target, scene_pass, scene_fail], 1)

    stats = is_match(images[0][0], images[1][0], debug=True)
    print(stats)
