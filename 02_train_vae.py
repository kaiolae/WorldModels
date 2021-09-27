# Example usage: python 02_train_vae.py --start_batch 1 --max_batch 1 --save_interval 25 --epochs 100 --new_model
# See meaning of parameters below.
# Make sure the right virtualenv is loaded before running:
# export WORKON_HOME=~/.virtualenvs
# source /usr/local/bin/virtualenvwrapper.sh
# workon worldmodels
import os

from VAE.world_model_vae import VAE
import argparse
import numpy as np
import pickle

# Only for GPU use:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"#"5,6,7,8"

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)
import tensorflow.compat.v1.keras.backend as K
K.set_session(sess)


def main(args):

    input_data_folder = args.input_data_folder
    start_batch = args.start_batch
    max_batch = args.max_batch
    new_model = args.new_model
    epochs = args.epochs
    save_interval = args.save_interval
    if args.savefolder:
        savefolder = args.savefolder
    else:
        savefolder = "./models/"

    if not input_data_folder:
        input_data_folder = "./data/"

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)


    vae = VAE()
    history = []

    if not new_model:
        try:
            # Fetching stored weights and history.
            vae.set_weights(savefolder+'final_full_vae_weights.h5')
            history_file = savefolder+'world_model_training_history.h5'
            with open(history_file, 'rb') as pickle_file:
                history = pickle.load(pickle_file, encoding='latin1') 
        except:
            print("Either set --new_model or ensure ./vae/final_full_vae_weights.h5 exists")
            raise
    first_item = True

    for batch_num in range(start_batch, max_batch + 1):
        print('Building batch {}...'.format(batch_num))
        if args.environment_name:
            env_name = args.environment_name
        else:
            env_name="doomrnn"
        print("Loading data from ", input_data_folder+'obs_data_' + env_name + '_' + str(batch_num) + '.npy')
        try:
            new_data = np.load(input_data_folder+'obs_data_' + env_name + '_' + str(batch_num) + '.npy',allow_pickle=True)
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
        print("Combining all data") 
        data_as_numpy = np.array(data[0])
        counter = 0
        for d in data:
            print("Data concatenated: ", counter, " of ", data.shape)
            if counter != 0:
                data_as_numpy = np.concatenate((data_as_numpy, np.array(d)))
            counter += 1
        data_as_numpy = np.asarray(data_as_numpy)

        data_as_numpy = data_as_numpy.astype('float32') / 255.
        print("data shape: ", data_as_numpy.shape)
        print("Training VAE on data with shape", data_as_numpy.shape)

        history.append(vae.train(data_as_numpy,epochs, save_interval, savefolder).history)
    else:
        print('no data found for batch number {}'.format(batch_num))

    vae.save_weights(os.path.join(savefolder,'final_'))

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
    parser.add_argument('--input_data_folder', type=str, help="Folder to load training data from.")
    parser.add_argument('--environment_name', type=str, help="Name of the environment data was generated from")

    args = parser.parse_args()

    main(args)
