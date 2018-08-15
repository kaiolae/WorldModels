# python 02_train_vae.py --new_model

from VAE.world_model_vae2 import VAE
import argparse
import numpy as np
import config
import pickle

# Only for GPU use:
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
from keras import backend as K
K.set_session(sess)

img_rows, img_cols, img_chns = 64, 64, 3
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)


def main(args):

    #TODO Loading just one set of images. Load more later.
    wm_images = np.load('./data/obs_data_doomrnn_1.npy')
    wm_images_as_numpy = np.array(wm_images[0])
    counter = 0
    for d in wm_images:
        if counter != 0:
            wm_images_as_numpy = np.concatenate((wm_images_as_numpy, np.array(d)))
        counter += 1
    wm_images_as_numpy = np.asarray(wm_images)
    print("Shape after load: ", wm_images_as_numpy.shape)
    wm_images_as_numpy = wm_images_as_numpy.astype('float32') / 255.
    wm_images_as_numpy = wm_images_as_numpy.reshape((wm_images_as_numpy.shape[0],) + original_img_size)

    # training
    history = VAE.train(wm_images_as_numpy)
            #validation_data=(x_test, None))

    VAE.save_weights('./models/world_model_vae.h5')

    # save training history
    fname = './models/world_model_training_history.h5'
    with open(fname, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__":

    #Kept argument parsing from original Keras code. Consider updating.
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--start_batch', type=int, default=0, help='The start batch number')
    parser.add_argument('--max_batch', type=int, default=0, help='The max batch number')

    parser.add_argument('--epochs', type=int, default=1, help='The number of passes through the entire data set.')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    args = parser.parse_args()

    main(args)

    #TODO Generate (or fetch) some DOOM training data and try training.
