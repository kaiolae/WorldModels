# Trains MDN-RNN using stored z-vectors and stored actions. Goal is to predict next
# z-vector from previous one and action.

#TODO may get some tips from this Keras World Models implementation https://github.com/AppliedDataSciencePartners/WorldModels/blob/master/rnn/arch.py


import argparse
import os
import pickle

import keras
import numpy as np
import mdn
from RNN import world_model_rnn

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #TODO Move to 0. 
 
# Do other imports now...
import keras
import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
from keras import backend as K, Input

K.set_session(sess)

#TODO Handle interrupted sequences, and add dying-prediction


VAL_SPLIT = 0.15
BATCH_SIZE = 256 # Fant ikke Ha's verdi i farta

def main(args):
    skip_ahead = args.skip_ahead
    training_data_file = args.training_data_file
    epochs = args.epochs
    sequence_length = args.sequence_length
    num_mixtures = args.num_mixtures
    upper_level_folder = args.upper_level_folder_name
    data_scaling_factor = args.data_scaling_factor
    data_size = args.data_size

    if not os.path.exists(upper_level_folder):
        os.makedirs(upper_level_folder)

    savefolder = upper_level_folder+"/trained_sequential_rnn"
    savefolder += "_" + str(num_mixtures) + "mixtures"
    savefolder += "_" + args.output_folder_name

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    #Loading z-values and actions. Expecting a compressed npz-file
    rnn_training_data = np.load(training_data_file)
    if data_size==-1:
        action_file = rnn_training_data['action']
        latent_file = rnn_training_data['latent']
    else:
        action_file = rnn_training_data['action'][:data_size]
        latent_file = rnn_training_data['latent'][:data_size]

    rnn = world_model_rnn.RNN(sequence_length=sequence_length, num_mixtures=num_mixtures)

    #TODO: Now, what to do about the fact that episodes may have different lengths?
    #I'll start with just getting this to work for fixed-length sequences, then add dying and variable length after.

    #The RNN-mdn code from https://github.com/cpmpercussion/creative-prediction/blob/master/notebooks/7-MDN-Robojam-touch-generation.ipynb
    #model=keras.Sequential()
    #model.add(keras.layers.LSTM(NUM_LSTM_UNITS, batch_input_shape=(None, SEQ_LENGTH, LATENT_VECTOR_SIZE),
    #                            return_sequences=False))
    # TODO Return sequences returns the hidden state, and feeds that to the next layer. When I do this with the MDN,
    # I get an error, because it doenvs not expect that input. I need to find a way to store the hidden state (for the
    # controller) without return sequences?

    #Setting up the training data
    #TODO Fix incomplete sequences
    X = []
    y = []
    print("Latent file size: ", len(latent_file))
    print("First dim: ", len(latent_file[3]))
    print("second dim: ", len(latent_file[0][0]))
    print("contents: ", latent_file[0][0])
    print("action contents: ", action_file[0][0])
    for i in range(len(latent_file)): #for each episode
        observations = latent_file[i] #All N observations (z-vectors) in an episode
        if len(observations) < sequence_length+1: #If we can't generate a full sequence, we skip this episode.
            continue
        actions = np.array(action_file[i]) #All N actions in an episode
        observations_and_actions = [] #Concatenating for each timestep.
        print("Obs, action len: ", len(observations), ", ", len(actions))
        for timestep in range(len(observations)):
            observations_and_actions.append(np.concatenate([observations[timestep],[actions[timestep]]]))
        for j in range(0, len(observations) - sequence_length, skip_ahead):
            X.append(observations_and_actions[j:j+sequence_length]) #the N prev obs. and actions
            y.append(observations[j+1:j+sequence_length+1]) #The next observations



    X=np.array(X)
    y=np.array(y)


    X = np.multiply(X, data_scaling_factor)
    Y = np.multiply(y, data_scaling_factor)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    #Training the model
    history = rnn.train(X, y, epochs, BATCH_SIZE, savefolder, validation_split=VAL_SPLIT)
    rnn.save_weights(os.path.join(savefolder,"rnn_trained_model.h5"))

    # save training history
    fname = os.path.join(savefolder,'training_history.h5')
    with open(fname, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)




if __name__ == "__main__":

    #Kept argument parsing from original Keras code. Consider updating.
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--training_data_file', type=str, help="Path to RNN training data.",
                        default = "./rnn_data_64_dim/rnn_training_data.npz")
    parser.add_argument('--epochs', type=int, help="How many passes through full training set.",
                        default = 100)
    parser.add_argument('--sequence_length', type=int, help="How many steps per sequence during training.",
                        default = 60)
    #parser.add_argument('--savefolder', type=str, help="Folder to store resulting rnn-model in. Default is ./rnn-model/",
    #                    default = "./rnn-model/")
    parser.add_argument('--num_mixtures', type=int, help="How many components in the mixture model.",
                    default = 5)
    parser.add_argument('--skip_ahead', type=int, help="How many steps in the sequence to skip forward between samples.",
                    default = 5)

    parser.add_argument('--output_folder_name', type=str, help="Unique name to store each run in unique folder.",
                        default = "run1")
    parser.add_argument('--upper_level_folder_name', type=str, help="Upper level folder to group several runs.",
                        default = "results")
    parser.add_argument('--data_scaling_factor', type=float, help="If we want to scale all data by a fixed scalar (may help avoid nans).",
                        default = 1.0)

    parser.add_argument('--data_size', type=int, help="How many of the episodes in the file to train on. Default is all.",
                        default = -1)

    args = parser.parse_args()

    main(args)
