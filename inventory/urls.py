from django.urls import path

from . import views


urlpatterns = [
    # Main page
    path('', views.main, name='main'),
    # API:
    # Search for item in database
    path(
        'api/search/<search_terms>',
        views.search_target,
        name='search_target'),
    # Search for item in scene
    path(
        'api/find/<target>',
        views.search_scene,
        name='search_scene'),
    # Transmit scene
    path(
        'api/live',
        views.get_image,
        name='get_image'),
]
