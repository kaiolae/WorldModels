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

#Very simple wall-detection based on color of walls wrt. floor/ceiling.
#Detects walls if they are so close that the corner pixel is covered by the wall.
#Wall has red-component around 110, ceiling/floor around 80. Intermediate threshold lets us separate the two.
def is_there_a_lefhand_wall(input_image, wall_threshold):
    #If the wall goes up to the 10th pixel from top, it's close enough that we define that as being next to the wall.
    desired_pixel_row = 10
    desired_pixel_column = 0

    #Channel 0 is red.
    if input_image[desired_pixel_row][desired_pixel_column][0] > wall_threshold:
        return True
    else:
        return False

def is_there_a_righthand_wall(input_image, wall_threshold):
    #If the wall goes up to the 10th pixel from top, it's close enough that we define that as being next to the wall.
    desired_pixel_row = 10
    desired_pixel_column = -1

    #Channel 0 is red.
    if input_image[desired_pixel_row][desired_pixel_column][0] > wall_threshold:
        return True
    else:
        return False
