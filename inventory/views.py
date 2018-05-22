from django.shortcuts import render
from django.http import HttpResponse


# -----------------------------------------------------------------------------
#
# Main Landing Page
#
# -----------------------------------------------------------------------------
def main(request):
    return HttpResponse("Test")


# -----------------------------------------------------------------------------
#
# Get search results
#
# -----------------------------------------------------------------------------
def search_target(request, search_terms):
    return HttpResponse("Test")


# -----------------------------------------------------------------------------
#
# Get current image
#
# -----------------------------------------------------------------------------
def get_image(request):
    return HttpResponse("Test")


# -----------------------------------------------------------------------------
#
# Find target in scene
#
# -----------------------------------------------------------------------------
def search_scene(request, target):

    """
    Returns a JsonResponse with the top 3 target ROIs, along with their
    confidence values.

    Parameters
    ----------
    request: django request object
        Standard django request
    target: str
        Name of target object

    Returns
    -------
    JsonResponse
        Top 3 target ROIs, with entries x_1, y_1, x_2, y_2, confidence
    """

    return HttpResponse("Test")
