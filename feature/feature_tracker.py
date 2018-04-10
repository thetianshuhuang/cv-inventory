# feature_tracker.py
# feature identification based tracker

import cv2
from matplotlib import pyplot as plt
from ddtools import gaussian_convolve, ddistance, weighted_bin
from cv_feature import weighted_grayscale, image_k_means, sift_scene


if(__name__ == "__main__"):
    target = cv2.imread("reference/target2x0.5.jpg")
    scene = cv2.imread("reference/scenex0.125.jpg")
    weights = image_k_means(target, 1)

    match_vectors = sift_scene(
        weighted_grayscale(target, weights[0]),
        weighted_grayscale(scene, weights[0]),
        0.9, plot=True)
    # match_vectors = sift_scene(
    #    cv2.imread("reference/target2x0.5.jpg", 0),
    #    cv2.imread("reference/scene2x0.125.jpg", 0))

    plt.imshow(match_vectors["plot"]), plt.show()

    target = match_vectors["target"]
    scene = match_vectors["scene"]
    length = match_vectors["length"]
    weights = match_vectors["weights"]

    w = []
    hist = []
    for i in range(length):
        for j in range(i + 1, length):
            # filter out distances between points mapped to multiple others
            if(scene[i] != scene[j] and target[i] != target[j]):
                dist_ratio = (
                    ddistance(scene[i], scene[j]) /
                    (ddistance(target[i], target[j]) + 1))
                hist.append(dist_ratio)
                w.append(1 / (weights[i] * weights[j]))

    plt.hist(hist, bins=100, weights=w, range=[0, 5])
    plt.show()
    plt.plot(gaussian_convolve(
        weighted_bin(0.01, hist, weights=w, epsilon=0.2), 20))
    plt.show()
