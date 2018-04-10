#
# cv_feature.py
#
# Opencv-based feature detection tools
# Written Tianshu Huang for the Tracker Synergy project and the Computer
# Vision Inventory Management system
#
# Functions
# ---------
# weighted_grayscale: get a weighted grayscale image
# image_k_means: find the k most prominent colors in an image
# sift_scene: search for matches in an image with SIFT and FLANN
#

import numpy as np
import cv2


def weighted_grayscale(image, weight):

    """
    Get a weighted grayscale image.

    Parameters
    ----------
    image : np.array
        Input BGR image
    weight : array
        [B,G,R] weights

    Returns
    -------
    np.array
        Weighted grayscale image
    """

    weight = [x * 255 / max(weight) for x in weight]
    split_image = cv2.split(image)
    return(np.uint8(
        split_image[0] * (weight[0] / 255.) / 3 +
        split_image[1] * (weight[1] / 255.) / 3 +
        split_image[2] * (weight[2] / 255.) / 3
    ))


def image_k_means(image, k):

    """
    Get dominant colors from a target image.

    Parameters
    ----------
    image : np.array
        Input BGR image
    k : int
        Number of colors to extract

    Returns
    -------
    array
        List of [B,G,R] tuples representing the K dominant colors.
    """

    array = np.copy(image).reshape((-1, 3))
    array = np.float32(array)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    return(center)


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
        # add to dictionary
        match_vectors["plot"] = out_image

    return(match_vectors)
