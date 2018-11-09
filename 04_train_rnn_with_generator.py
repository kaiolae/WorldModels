# Trains MDN-RNN using stored z-vectors and stored actions. Goal is to predict next
# z-vector from previous one and action.

#TODO may get some tips from this Keras World Models implementation https://github.com/AppliedDataSciencePartners/WorldModels/blob/master/rnn/arch.py


import argparse
import os
import pickle
import sys

import keras
import numpy as np
import mdn
from RNN import world_model_rnn

import tensorflow as tf

from batch_generator import KerasBatchGenerator

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
from keras import backend as K, Input

K.set_session(sess)

#TODO Handle interrupted sequences, and add dying-prediction


SKIP_AHEAD = 1 #How many steps to skip forward when cutting out the next training sequence.
VAL_SPLIT = 0.15
BATCH_SIZE = 256 # Fant ikke Ha's verdi i farta


def main(args):
    training_data_file = args.training_data_file
    epochs = args.epochs
    sequence_length = args.sequence_length
    num_mixtures = args.num_mixtures

    savefolder = "trained_sequential_rnn"
    savefolder += "_" + str(num_mixtures) + "mixtures"
    savefolder += "_" + args.output_folder_name

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    #Loading z-values and actions. Expecting a compressed npz-file
    rnn_training_data = np.load(training_data_file)
    action_data = rnn_training_data['action']
    observation_data = rnn_training_data['latent']

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
    print("Latent file size: ", len(observation_data))
    print("First dim: ", len(observation_data[0]))
    print("second dim: ", len(observation_data[0][0]))
    print("contents: ", observation_data[0][0])
    print("action contents: ", action_data[0][0])

    #TODO if desired, set up a validatio data generator too.
    train_data_generator = KerasBatchGenerator(observation_data, action_data, sequence_length, BATCH_SIZE)

    #Training the model
    #history = rnn.train(X, y, epochs, BATCH_SIZE, validation_split=VAL_SPLIT)
    #Training with generator to save memory.
    history = rnn.model.fit_generator(train_data_generator.generate(),
                            train_data_generator.get_total_num_train_samples()//BATCH_SIZE,
                            epochs=epochs
                            )
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
    parser.add_argument('--output_folder_name', type=str, help="Unique name to store each run in unique folder.",
                        default = "run1")
    args = parser.parse_args()

    main(args)
