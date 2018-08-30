# Takes a trained VAE, as well as observation and action data. Generates I/O data for the world model RNN.
# Inputs to RNN are current latent vector z (generated by VAE from current observation),
# as well as current action. Desired output is the correct next z-vector. Loss is given by the MDN-RNN.

# Takes as arguments the file where observation and action-data can be found, as well as the folder in which to store
# results.
import os

from VAE.world_model_vae import VAE
import argparse
import numpy as np
import glob

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
from keras import backend as K
K.set_session(sess)

def main(args):
    obs_folder = args.obs_folder
    obs_filename_base = 'obs_data_doomrnn_'
    actions_filename_base = 'action_data_doomrnn_'
    obs_file_pattern = os.path.join(obs_folder,obs_filename_base+'*')
    action_file_pattern = os.path.join(obs_folder,actions_filename_base+'*')
    print("Obs file pattern: ", obs_file_pattern)
    savefolder = args.savefolder
    vae_weights = args.loaded_vae_weights

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    vae = VAE()

    try:
      vae.set_weights(vae_weights)
    except:
      print(vae_weights, " does not exist - ensure you have run 02_train_vae.py first")
      raise

    obs_data = []
    action_data = []
    for file_number in range(1,len(glob.glob(obs_file_pattern))+1):
        obs_file = os.path.join(obs_folder, obs_filename_base)+str(file_number) + ".npy"
        action_file = os.path.join(obs_folder, actions_filename_base) + str(file_number) + ".npy"
        print("Loading obs file ", obs_file)
        for episode in np.load(obs_file):
            obs_data.append(episode)
        print("loading action file ", action_file)
        for episode in np.load(action_file):
            action_data.append(episode)
    print("-----LOADING FILES DONE -------")

    obs_data = np.array(obs_data)
    action_data = np.array(action_data)

    print("Obs data has shape ", obs_data.shape)
    print("action data has shape ", action_data.shape)

    #If the image data is integral (range [0, 255]), we convert to float in range [0,1]
    #print(obs_data)
    converted_obs_data=[]
    if np.issubdtype(obs_data[0][0].dtype, np.integer):
        print("Image data was of integer type. Converting to float before further processing.")
        for episode in obs_data:
            converted_episode = []
            for image in episode:
                converted_image = image.astype('float32') / 255.
                converted_episode.append(converted_image)
            converted_obs_data.append(converted_episode)
        obs_data = converted_obs_data

    #Need to store each ep separately. we cant predict btw episodes
    #TODO Note: There are equally many actions and observations. I guess the final action can just be discarded?
    z_sequences = [] #One for each ep
    action_sequences = [] #One for each ep
    for episode_number in range(len(obs_data)):
        observations = np.array(obs_data[episode_number])
        # Generating all latent codes for this episode
        latent_values = vae.generate_latent_variables(observations)
        z_sequences.append(latent_values)
        action_sequences.append(np.array(action_data[episode_number]))

        print("Added latent sequences of length ", len(latent_values), " and action sequence of length ", len(action_sequences[-1]))
        print("Array sizes: ", len(z_sequences), ", ", len(action_sequences))
    z_sequences = np.array(z_sequences) #Will this work? Has sub-arrays of differing lengths.


    np.savez_compressed(os.path.join(savefolder, "rnn_training_data.npz"), action=action_data, latent = z_sequences)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--obs_folder', type=str, help="Path to observations-folder.",
                        default = "./data/")
    parser.add_argument('--loaded_vae_weights', type=str, help="Path to load VAE weights from.",
                        default = "./models/final_full_vae_weights.h5")
    parser.add_argument('--action_file', type=str, help="Path to actions-file.",
                        default = "./data/action_data_doomrnn_1.npy")
    parser.add_argument('--savefolder', type=str, help="Folder to store results in. Default is ./rnn-data/",
                        default = "./rnn-data/")
    args = parser.parse_args()

    main(args)
