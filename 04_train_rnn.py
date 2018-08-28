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

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
from keras import backend as K, Input

K.set_session(sess)

#TODO Handle interrupted sequences, and add dying-prediction


SEQ_LENGTH = 30
SKIP_AHEAD = 3 #How many steps to skip forward when cutting out the next training sequence.
EPOCHS = 100
VAL_SPLIT = 0.15

def main(args):
    training_data_file = args.training_data_file
    savefolder = args.savefolder

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    #Loading z-values and actions. Expecting a compressed npz-file
    rnn_training_data = np.load(training_data_file)
    action_file = rnn_training_data['action']
    latent_file = rnn_training_data['latent']

    rnn = world_model_rnn.RNN()

    #TODO: Now, what to do about the fact that episodes may have different lengths?
    #I'll start with just getting this to work for fixed-length sequences, then add dying and variable length after.

    #The RNN-mdn code from https://github.com/cpmpercussion/creative-prediction/blob/master/notebooks/7-MDN-Robojam-touch-generation.ipynb
    #model=keras.Sequential()
    #model.add(keras.layers.LSTM(NUM_LSTM_UNITS, batch_input_shape=(None, SEQ_LENGTH, LATENT_VECTOR_SIZE),
    #                            return_sequences=False))
    # TODO Return sequences returns the hidden state, and feeds that to the next layer. When I do this with the MDN,
    # I get an error, because it does not expect that input. I need to find a way to store the hidden state (for the
    # controller) without return sequences?
    #model.add(keras.layers.LSTM(NUM_LSTM_UNITS)) #TODO Why does it crash when we only use 1 LSTM layer??
    #model.add(mdn.MDN(LATENT_VECTOR_SIZE, NUM_MIXTURES))

    #Setting up the training data
    #TODO Fix incomplete sequences
    X = []
    y = []
    print("Latent file size: ", len(latent_file))
    print("First dim: ", len(latent_file[0]))
    print("second dim: ", len(latent_file[0][0]))
    print("contents: ", latent_file[0][0])
    print("action contents: ", action_file[0][0])
    for i in range(len(latent_file)): #for each episode
        observations = latent_file[i] #All N observations (z-vectors) in an episode
        if len(observations) < SEQ_LENGTH: #If we can't generate a full sequence, we skip this episode.
            continue
        actions = np.array(action_file[i]) #All N actions in an episode
        observations_and_actions = [] #Concatenating for each timestep.
        for timestep in range(len(observations)):
            observations_and_actions.append(np.concatenate([observations[timestep],[actions[timestep]]]))
        for j in range(0, len(observations) - SEQ_LENGTH, SKIP_AHEAD):
            X.append(observations_and_actions[j:j+SEQ_LENGTH]) #the N prev obs. and actions
            y.append(observations[j+SEQ_LENGTH]) #The next obs

    X=np.array(X)
    y=np.array(y)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    #Training the model
    history = rnn.train(X, y, EPOCHS, validation_split=VAL_SPLIT)
    rnn.save_weights(os.path.join(savefolder,"rnn_trained_model.h5"))

    # save training history
    fname = os.path.join(savefolder,'training_history.h5')
    with open(fname, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)




if __name__ == "__main__":

    #Kept argument parsing from original Keras code. Consider updating.
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--training_data_file', type=str, help="Path to RNN training data.",
                        default = "./rnn-data/rnn_training_data.npz")
    parser.add_argument('--savefolder', type=str, help="Folder to store resulting rnn-model in. Default is ./rnn-model/",
                        default = "./rnn-model/")
    args = parser.parse_args()

    main(args)
