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


def find_target(target, scene):

    images = load_weighted([target, scene], 1)

    match_vectors = sift_scene(
        images[0][0],
        images[1][0],
        0.8, plot=True)

    x = 0
    for i in match_vectors["weight"]:
        x += i
    print(x)

    kernel_map = kernel_transform(match_vectors)
    remove_outliers(kernel_map, 0.2)

    print(weighted_stats(kernel_map))

    plt.imshow(match_vectors["plot"]), plt.show()

    plt.hist(kernel_map["data"], weights=kernel_map["weight"], bins=100)
    plt.show()


if(__name__ == "__main__"):

    find_target(
        "reference/scene-wire/tx0.5.jpg",
        "reference/scene-wire/rx0.125.jpg")
    find_target(
        "reference/scene-wire/tx0.5.jpg",
        "reference/scene-nano/rx0.125.jpg")
