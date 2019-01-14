from skimage import io, filters, color, measure
from scipy import ndimage
import numpy as np

MONSTERS_THRESHOLD = 0.2
#EXPLOSION_THRESHOLD = 0.45 #TODO May need some work here on the auto detection
#FIREBALL_EDGE_THRESHOLD = 0.22 #Reliably separates fireballs from other items - only they have this strong edges.
FIREBALL_THRESHOLD = 0.65

#TODO Maybe for walls, we should average a larger area rather than just measure one pixel?

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
    img = img[28:32,:,:] #This cuts out only the segment with monsters "heads". Counting this is more reliable, because monster bodies may get disconnected due to a fireball in the middle - making 1 monster count as 2.
    #Getting only green-component, which indicates monsters best.
    img = img[:,:,1]
    thresholded_img = ndimage.binary_fill_holes(img<MONSTERS_THRESHOLD)
    labels= measure.label(thresholded_img)
    return labels.max(), thresholded_img

def count_fireballs_edge(img):
    im = color.rgb2gray(one_image)
    edge_roberts = filtes.roberts(im) #Roberts edge detection

    #Only strong edges give result
    return count_objects(edge_roberts, FIREBALL_EDGE_THRESHOLD)

def count_fireballs(img):
    return count_objects(img, FIREBALL_THRESHOLD)


def is_there_a_big_explosion(input_image):
    #Explosions are the only bright items in the top half of the image - fireballs never go up there.
    top_half=input_image[:28,:,:]
    return count_objects(top_half, FIREBALL_THRESHOLD)[0]>0

def is_there_a_big_explosion_deprecated(input_image, fireball_threshold):
    # Classifies whether there is currently a big fireball exploding in the image.
    # Defined here as having an explosion covering more than 10% of the pixels.
    im = color.rgb2gray(input_image)
    thresholded_image = im > EXPLOSION_THRESHOLD
    ten_percent_of_pixels = 0.1*64*64
    if np.sum(thresholded_image) > ten_percent_of_pixels:
        return True
    else:
        return False

#Very simple wall-detection based on color of walls wrt. floor/ceiling.
#Detects walls if they are so close that the corner pixel is covered by the wall.
#Wall has red-component around 110, ceiling/floor around 80. Intermediate threshold lets us separate the two.
def is_there_a_lefthand_wall(input_image, wall_threshold):
    #If the wall goes up to the 10th pixel from top, it's close enough that we define that as being next to the wall.
    desired_pixel_row_min = 0
    desired_pixel_row_max = 10
    desired_pixel_column = 0

    #Channel 0 is red.
    if np.average(input_image[desired_pixel_row_min:desired_pixel_row_max,desired_pixel_column,0]) > wall_threshold:
        return True
    else:
        return False

def is_there_a_righthand_wall(input_image, wall_threshold):
    #If the wall goes up to the 10th pixel from top, it's close enough that we define that as being next to the wall.
     
    desired_pixel_row_max = 10
    desired_pixel_row_min = 0
    desired_pixel_column = -1

    #Channel 0 is red.
    if np.average(input_image[desired_pixel_row_min:desired_pixel_row_max,desired_pixel_column,0]) > wall_threshold:
        return True
    else:
        return False

def is_there_a_wall(input_image, wall_threshold):
    is_left = is_there_a_lefthand_wall(input_image, wall_threshold)
    if not is_left:
        return is_there_a_righthand_wall(input_image, wall_threshold)
    else:
        return True
