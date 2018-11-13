from skimage import io, filters, color, measure
from scipy import ndimage
import numpy as np

MONSTERS_THRESHOLD = 0.23
FIREBALL_THRESHOLD = 0.45 #TODO May need some work here on the auto detection

def count_objects(input_image, threshold, above_threshold=True):
    im = color.rgb2gray(input_image)
    if above_threshold:
        thresholded_image = im>threshold
    else:
        thresholded_image = im<threshold

    objects = ndimage.binary_fill_holes(thresholded_image>0.5)
    object_labels = measure.label(objects)
    return object_labels.max(), thresholded_image

def count_monsters(img):
    return count_objects(img, MONSTERS_THRESHOLD, above_threshold=False)

def count_fireballs(img, fireball_threshold):
    return count_objects(img, fireball_threshold, above_threshold=True)

def is_there_a_big_explosion(input_image, fireball_threshold):
    # Classifies whether there is currently a big fireball exploding in the image.
    # Defined here as having an explosion covering more than 10% of the pixels.
    im = color.rgb2gray(input_image)
    thresholded_image = im > fireball_threshold
    ten_percent_of_pixels = 0.1*64*64
    if np.sum(thresholded_image) > ten_percent_of_pixels:
        return True
    else:
        return False
