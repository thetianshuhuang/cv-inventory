#
# cv_feature.py
#
# Opencv-based feature detection tools
# Written by Tianshu Huang for cv-inventory, April 2018
#
# Functions
# ---------
# knn_plot: generate visual plot for sift_scene if desired
# sift_scene: search for matches in an image with SIFT and FLANN
#

import cv2
import math


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


def sift_pdf(ratio):

    """
    Estimate the false positive probability given an input distance ratio
    Estimated from the graph on page 20 of "Distinctive Image Features from
    Scale-Invariant Keypoints", David G. Lowe

    Parameters
    ----------
    ratio: float
        Ratio of best match distance to second best match distance

    Returns
    -------
    float: confidence
        Probability value of that match
    """

    if(ratio < 0.3):
        return(1)
    elif(ratio < 0.6):
        return(0.88 + (0.6 - ratio) * 0.4)
    elif(ratio < 0.8):
        return(0.3 + (0.8 - ratio) * 2.9)
    elif(ratio < 0.9):
        return(0.05 + (0.9 - ratio) * 2.5)
    else:
        return((1 - ratio) * 0.5)


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

    # build return dictionary
    output = {"target": [], "scene": [], "weight": []}

    for i in range(len(matches)):
        output["target"].append(kp_target[matches[i][0].queryIdx].pt)
        output["scene"].append(kp_scene[matches[i][0].trainIdx].pt)
        output["weight"].append(
            sift_pdf(matches[i][0].distance / matches[i][1].distance))

    output["length"] = len(output["target"])

    # Generate output if desired
    if("plot" in kwargs and kwargs["plot"]):
        # add to dictionary
        output["plot"] = knn_plot(
            img_target, kp_target, img_scene, kp_scene, matches, max_ratio)

    return(output)
