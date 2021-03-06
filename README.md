# cv-inventory
Computer Vision based Inventory Management System

## Dependencies
- Python 3
- Opencv 3.4 or greater with opencv_contrib extra modules
- Numpy, scipy, matplotlib
- Django, v2.0 or greater
- Pillow (Python Imaging Library)
- python-Levenshtein (Levenshtein distance for database searching)


## Feature
The algorithm for feature detection consists of two parts; the first pass identifies possible locations of the target object; the second pass checks each location to determine a confidence value.

### Feature
The feature detection system is located in cv_inventory/inventory/feature. To run tests, use one of
```
python feature.py classify
python feature.py search
python feature.py search clean
```


### Image matching
1. Use SIFT, then FLANN, to obtain matches. Express the confidence as
```
(second - best) / second
```
2. Map each match to the kernel space d(x_0, x_1) with confidence c_1 * c_2
3. Compute the kernel space median; this is the scale of the identified object
4. Normalize the kernel space by by the scale
5. Compute the kernel space variance, in pixels
6. Confidence = 1 / (1 + variance)
