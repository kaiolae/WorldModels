# python 02_train_vae.py --new_model

from VAE.world_model_vae import VAE
import argparse
import numpy as np
import config

# Only for GPU use:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
from keras import backend as K
K.set_session(sess)


def main(args):
    start_batch = args.start_batch
    max_batch = args.max_batch
    new_model = args.new_model
    epochs = args.epochs

    vae = VAE()

    if not new_model:
        try:
            vae.set_weights('./vae/weights.h5')
        except:
            print("Either set --new_model or ensure ./vae/weights.h5 exists")
            raise

    for i in range(epochs):
        for batch_num in range(start_batch, max_batch + 1):
            print('Building batch {}...'.format(batch_num))
            first_item = True

            for env_name in config.train_envs:
                print("Loading data from ", './data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
                try:
                    new_data = np.load('./data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
                    print("Shape after load: ", new_data.shape)
                    if first_item:
                        data = new_data
                        first_item = False
                    else:
                        data = np.concatenate([data, new_data])
                        print("Shape after concat: ", data.shape)
                    print('Found {}...current data size = {} episodes'.format(env_name, len(data)))
                except:
                    pass

            if first_item == False:  # i.e. data has been found for this batch number
                #Putting all images into one large 4D numpy array (img nr, width, height, color channels)
                data_as_numpy = np.array(data[0])
                counter = 0
                for d in data:
                  if counter != 0:
                    data_as_numpy=np.concatenate((data_as_numpy, np.array(d)))
                  counter+=1
                data_as_numpy = np.asarray(data_as_numpy)

                print("data shape: ", data_as_numpy.shape)
#TODO Figure out what format to deliver training data in
#                data = np.array([item for obs in data for item in obs])
                print("Training VAE on data with shape", data_as_numpy.shape)
                vae.train(data_as_numpy)
            else:
                print('no data found for batch number {}'.format(batch_num))


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
