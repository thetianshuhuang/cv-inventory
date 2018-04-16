#
# feature.py
#
# feature_tracker.py
# feature identification based tracker
#

from cv_img import load_weighted
from cv_feature import sift_scene
from matplotlib import pyplot as plt
import kernel_ops
from discrete import gaussian_convolve, weighted_bin
from kmeans import kmeans


# -----------------------------------------------------------------------------
#
# matches_plot
#
# -----------------------------------------------------------------------------
def matches_plot(matches, kernel, sigma):

    """
    Generate plots visualizing matches and clustering
    (utility function for find_match)

    Parameters
    ----------
    matches : dict
        matches to plot
    kernel : dict
        kernel, with weights
    sigma : float
        coefficient for convolution
    """

    # Image
    plt.imshow(matches["plot"]), plt.show()
    # Raw histogram
    plt.hist(kernel["data"], weights=kernel["weight"], bins=100)
    plt.show()
    # Gaussian convolve
    plt.plot(gaussian_convolve(
        weighted_bin(
            1. / sigma, kernel["data"],
            weights=kernel["weight"], epsilon=0),
        sigma))
    plt.show()


# -----------------------------------------------------------------------------
#
# find_match
#
# -----------------------------------------------------------------------------
def find_match(target, scene, **kwargs):

    """
    Recursively find a match for a target image using a greedy search
    algorithm.

    Parameters
    ----------
    target : np.array
        Single channel input image array
    scene : np.array
        Single channel scene to search in
    debug= : bool
        Show debug images and plots
    sigma= : int
        Constant to use for convolutions
    max_ratio= : float
        Maximum ratio for matches to be considered;
        smaller ratio -> higher confidence
    outliers= : float
        Proportion of outliers to be ignored
    iteration= : int
        Maximum number of iterations
    scale_factor= : 3
        Factor by which the search area narrows each iteration
    """

    # Assign defualts
    settings = {
        "debug": False,
        "sigma": 100,
        "max_ratio": 1.0,
        "outliers": 0.2,
        "iteration": 3,
        "scale_factor": 3,
        "iterations_used": 1
    }
    settings.update(kwargs)

    # find matches
    matches = sift_scene(
        target, scene, max_ratio=settings["max_ratio"], plot=settings["debug"])

    # get kernel
    kernel_list = kernel_ops.remove_outliers(
        kernel_ops.kernel_transform(matches), settings["outliers"])

    # compute statistics
    stats = kernel_ops.weighted_stats(kernel_list)
    stats.update(kernel_ops.clustering_stats(kernel_list, settings["sigma"]))
    confidence = 1 - pow(
        1.15, -1 * (stats["in_range"]**2 / (stats["total"])))
    stats.update({"confidence": confidence})

    # Show debug plots
    if(settings["debug"]):
        matches_plot(matches, kernel_list, settings["sigma"])

    # refine search if necessary
    if(stats["center"] * max(target.shape) * 2 < max(scene.shape) and
       settings["iteration"] > 0):

        # Assemble new ROI
        means = kmeans(matches["scene"], 3, weights=matches["weight"])
        roi_center = means["means"][0]
        roi_size = max(scene.shape) / (settings["scale_factor"] * 2)
        roi = scene[
            int(roi_center[1] - roi_size):int(roi_center[1] + roi_size),
            int(roi_center[0] - roi_size):int(roi_center[0] + roi_size)]

        # Call recursion
        settings["iteration"] -= 1
        settings["iterations_used"] += 1
        stats = find_match(target, roi, **settings)

    stats.update({"iterations_used": settings["iterations_used"]})
    return(stats)


# -----------------------------------------------------------------------------
#
# tests
#
# -----------------------------------------------------------------------------
# 'python feature.py' to run tests
if(__name__ == "__main__"):

    target = "reference/scene-wire/tx0.5.jpg"
    target_2 = "reference/scene-nano/tx0.25.jpg"
    scene_pass = "reference/scene-wire/wide.jpg"
    scene_pass_2 = "reference/scene-wire/sx1.jpg"
    scene_fail = "reference/scene-nano/sx1.jpg"

    messages = [[True, True, False], [False, False, True]]
    true_msg = "TEST: Object in scene, should show a high confidence"
    false_msg = "TEST: Object not in scene, should show a low confidence"

    images = load_weighted([target, scene_pass, scene_pass_2, scene_fail], 1)

    for i in range(1, len(images)):
        if(messages[0][i - 1]):
            print(true_msg)
        else:
            print(false_msg)
        stats = find_match(
            images[0][0], images[i][0], debug=False, scale_factor=3)
        print(stats)

    images = load_weighted([target_2, scene_pass, scene_pass_2, scene_fail], 1)
    for i in range(1, len(images)):
        if(messages[1][i - 1]):
            print(true_msg)
        else:
            print(false_msg)
        stats = find_match(
            images[0][0], images[i][0], debug=False, scale_factor=3)
        print(stats)
