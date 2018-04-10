#
# feature.py
#
# feature_tracker.py
# feature identification based tracker
#

from matplotlib import pyplot as plt
from cv_img import load_weighted
from cv_feature import sift_scene, sift_histogram

if(__name__ == "__main__"):

    images = load_weighted([
        "reference/target2x0.5.jpg",
        "reference/scene2x0.125.jpg"],
        1)

    match_vectors = sift_scene(
        images[0][0],
        images[1][0],
        0.9, plot=True)

    plt.imshow(match_vectors["plot"]), plt.show()

    plt.plot(sift_histogram(match_vectors, 0.01, 20, 0.2))
    plt.show()
