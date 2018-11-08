from skimage import io, filters, color, measure
from scipy import ndimage

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
