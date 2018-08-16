# python 02_train_vae.py --new_model
import os

from VAE.world_model_vae_2 import VAE
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

BATCH_SIZE = 100


def main(args):

    start_batch = args.start_batch
    max_batch = args.max_batch
    new_model = args.new_model
    epochs = args.epochs
    save_interval = args.save_interval
    if args.savefolder:
        savefolder = args.savefolder
    else:
        savefolder = "./models/"

    vae = VAE()
    history = []

    if not new_model:
        try:
            vae.set_weights('./vae/weights.h5')
        except:
            print("Either set --new_model or ensure ./vae/weights.h5 exists")
            raise
    first_item = True

    for batch_num in range(start_batch, max_batch + 1):
        print('Building batch {}...'.format(batch_num))

        for env_name in config.train_envs:
            print("Loading data from ", './data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
            try:
                new_data = np.load('./data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
                print("Shape after load: ", new_data.shape)
                if first_item:
                    print("Initializing data")
                    data = new_data
                    first_item = False
                else:
                    print("concatenating")
                    data = np.concatenate([data, new_data])
                    print("Shape after concat: ", data.shape)
                print('Found {}...current data size = {} episodes'.format(env_name, len(data)))
            except:
                pass

    if first_item == False:  # i.e. data has been found for this batch number
        # Putting all images into one large 4D numpy array (img nr, width, height, color channels)
        data_as_numpy = np.array(data[0])
        counter = 0
        for d in data:
            if counter != 0:
                data_as_numpy = np.concatenate((data_as_numpy, np.array(d)))
            counter += 1
        data_as_numpy = np.asarray(data_as_numpy)

        data_as_numpy = data_as_numpy.astype('float32') / 255.
        data_as_numpy = data_as_numpy.reshape((data_as_numpy.shape[0],) + original_img_size)
        print("data shape: ", data_as_numpy.shape)
        # TODO Figure out what format to deliver training data in
        #                data = np.array([item for obs in data for item in obs])
        print("Training VAE on data with shape", data_as_numpy.shape)
        #Training - imitating Ha's way of doing epochs and batches.
        num_batches = int(np.floor(data_as_numpy.size / BATCH_SIZE))
        print("Num batches: ", num_batches)
        for epoch in range(epochs):
            np.random.shuffle(data_as_numpy)
            for idx in range(num_batches):
                batch = data_as_numpy[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]
                history.append(vae.train(batch,1, save_interval, savefolder))
    else:
        print('no data found for batch number {}'.format(batch_num))

    vae.save_weights(os.path.join(savefolder,'final_weights.h5'))

    # save training history
    fname = os.path.join(savefolder,'world_model_training_history.h5')
    with open(fname, 'wb') as file_pi:
        pickle.dump(history, file_pi)


if __name__ == "__main__":

    #Kept argument parsing from original Keras code. Consider updating.
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--start_batch', type=int, default=0, help='The start batch number')
    parser.add_argument('--max_batch', type=int, default=0, help='The max batch number')
    parser.add_argument('--save_interval', type=int, default=0, help='How many epochs between storing model. Default is only after last epoch.')

    parser.add_argument('--epochs', type=int, default=1, help='The number of passes through the entire data set.')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    parser.add_argument('--savefolder', type=str, help="Folder to store results in. Default is ./models/")
    args = parser.parse_args()

    main(args)

    #TODO Generate (or fetch) some DOOM training data and try training.
