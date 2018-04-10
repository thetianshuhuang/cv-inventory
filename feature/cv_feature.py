#
# cv_feature.py
#
# Opencv-based feature detection tools
# Written by Tianshu Huang for cv-inventory
#
# Functions
# ---------
# knn_plot: generate visual plot for sift_scene if desired
# sift_scene: search for matches in an image with SIFT and FLANN
#

import cv2
from ddtools import ddistance, gaussian_convolve, weighted_bin


def knn_plot(img_target, kp_target, img_scene, kp_scene, matches, max_ratio):

    """
    Generate a visual plot of opencv KNN matches.

    Parameters
    ----------
    img_target, kp_target, img_scene, kp_scene, matches
        Passthrough to cv2.drawMatchesKnn
    max_ratio
        Maximum ratio between best and second best matches to draw
    """

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]
    for i, (best, second) in enumerate(matches):
        # Mirror the Ratio Test
        if(best.distance < max_ratio * second.distance):
            matchesMask[i] = [1, 0]
    # configure pyplot
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    # generate output image
    out_image = cv2.drawMatchesKnn(
        img_target, kp_target, img_scene, kp_scene,
        matches, None, **draw_params)

    return(out_image)


def sift_scene(img_target, img_scene, max_ratio, **kwargs):
    """
    Find matches of an image in a scene with SIFT and FLANN.

    Parameters
    ----------
    img_target : np.array
        Input single channel reference image
    img_scene : np.array
        Scene to find the target in
    max_ratio : float
        Maximum ratio between best match and second match distances;
        matches with a higher ratio are discarded
    plot= : bool
        Returns output plot if set to True

    Returns
    -------
    dict, with entries:
        "target": target match coordinates
        "scene": scene match coordinates
        "weights": confidence value for each match
        "length": number of matches
        "output": optional; matplotlib output
    """

    # Initialize and run SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp_target, des_target = sift.detectAndCompute(img_target, None)
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)

    # Run FLANN
    # configuration
    FLANN_INDEX_KDTREE = 1
    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # get best two matches, return as array of array of best matches
    matches = flann.knnMatch(des_target, des_scene, k=2)

    # Run ratio test
    output = [
        [best, second]
        for i, (best, second) in enumerate(matches)
        if best.distance < max_ratio * second.distance
    ]

    # build return dictionary
    match_vectors = {
        "target": [kp_target[match[0].queryIdx].pt for match in output],
        "scene": [kp_scene[match[0].trainIdx].pt for match in output],
        "weights": [
            match[0].distance * match[0].distance / match[1].distance
            for match in output],
        "length": len(output)
    }

    # Generate output if desired
    if("plot" in kwargs and kwargs["plot"]):
        # add to dictionary
        match_vectors["plot"] = knn_plot(
            img_target, kp_target, img_scene, kp_scene, matches, max_ratio)

    return(match_vectors)


def sift_histogram(matches, bin_size, sigma, epsilon):

    """
    Generate a histogram showing the pairwise distance ratio in a set of
    matches with bin size, convolution coefficient sigma, and tail cutoff
    epsilon.

    Parameters
    ----------
    matches: dict, with entries:
        "target": target coordinates
        "scene": scene coordinates
        "weights": weight of each match
        "length": number of matches
    bin_size: float
        Size of each bin
    sigma: float
        Passed to gaussian_convolve.
    epsilon: float
        Passed to weighted_bin

    Returns
    -------
    float[]
        Binned histogram
    """

    # Assemble weights and histogram from input data
    weights = []
    hist = []
    for i in range(matches["length"]):
        for j in range(i + 1, matches["length"]):
            # filter out distance between points mapped to multiple other
            # points
            if(
               matches["scene"][i] != matches["scene"][j] and
               matches["target"][i] != matches["target"][j]):

                # Take the kernel map from the input coordinates to the
                # distance ratio
                dist_ratio = (
                    ddistance(matches["scene"][i], matches["scene"][j]) /
                    (ddistance(matches["target"][i], matches["target"][j]) + 1)
                )
                hist.append(dist_ratio)
                weights.append(
                    1 / (matches["weights"][i] * matches["weights"][j]))

    # return the convoluted weighted binned histogram
    return(
        gaussian_convolve(
            weighted_bin(
                bin_size, hist, weights=weights, epsilon=epsilon),
            sigma=sigma))
