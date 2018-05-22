from django.db import models


class Target(models.Model):

    """
    Target object

    Attributes
    ----------
    target_id: models.IntegerField / int
        Unique identifier for the target object
    name: models.CharField / str
        Display name of the target object
    """

    target_id = models.IntegerField
    name = models.CharField(max_length=100)


class TargetImg(models.Model):

    """
    Target image associated with a target object
    Multiple target images can associated with the same Target object

    Attributes
    ----------
    target: models.ForeignKey / ptr
        points to the parent Target object
    image: models.ImageField / img
        Image file.
    """

    target = models.ForeignKey(Target, on_delete=models.CASCADE)
    image = models.ImageField
