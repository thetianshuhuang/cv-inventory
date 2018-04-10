#
# cv_img.py
#
# Opencv-based image manipulation tools
# Written by Tianshu Huang for cv-inventory, April 2018
#
# Functions
# ---------
# weighted_grayscale: get a weighted grayscale image
# image_k_means: find the k most prominent colors in an image
# load_weighted: load images and get weighted grayscale images
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


def load_weighted(files, k):

    """
    Load the k best weighted images as determined by a reference image.

    Parameters
    ----------
    files: str[]
        Additional image filenames to be loaded.
        The first image in the array is the reference image.
    k: int
        Number of images to output

    Returns
    -------
    np.array[][]
        List of [list of k best matches] for each image.
    """

    assert(len(files) > 1)

    images = [cv2.imread(file) for file in files]
    weights = image_k_means(images[0], k)

    output = [
        [weighted_grayscale(image, weight) for weight in weights]
        for image in images
    ]

    return(output)
