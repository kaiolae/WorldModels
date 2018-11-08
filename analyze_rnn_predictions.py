# Code that helps avoid overusing memory

import tensorflow as tf

import mdn

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

#Importing the VAE and RNN.
import os
import sys
#Adding WorldModels path to pythonpath
nb_dir = os.path.split(os.getcwd())[0]
print(nb_dir)
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

#Importing the VAE
from VAE.world_model_vae import VAE
from RNN.world_model_rnn import RNN
import numpy as np

LATENT_SPACE_DIMENSIONALITY = 64
RNN_SIZE = 512
VAE_PATH = "vae_model_64_dim/final_full_vae_weights.h5"

def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature."""
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist

class RNNAnalyzer:

    def __init__(self, rnn_load_path, num_mixtures, temperature):
        self.vae=VAE()
        vae.set_weights(VAE_PATH)

        self.rnn = RNN(decoder_mode=True, num_mixtures=num_mixtures)
        self.last_loaded_from = rnn_load_path
        self.rnn.set_weights(rnn_load_path)
        self.frame_count = 0
        self.z = None
        self.num_mixtures = num_mixtures
        self.temperature = temperature


    def _reset(self):
        self.rnn.set_weights(self.last_loaded_from)
        self.z = None


    def decode_with_vae(self, latent_vector_sequence):
        reconstructions = self.vae.decoder.predict(np.array(latent_vector_sequence))
        return reconstructions




    def predict_one_step(self, action, previous_z=[]):
        #Predicts one step ahead from the previous state.
        #If previous z is given, we predict with that as input. Otherwise, we dream from the previous output we generated.

        self.frame_count += 1
        prev_z = np.zeros((1, 1, LATENT_SPACE_DIMENSIONALITY))
        if len(previous_z)>0:
            prev_z[0][0] = previous_z
        else:
            prev_z[0][0] = self.z

        prev_z[0][0].append(action) #Adding the action that is the final bit of input.

        mixture_params = self.rnn.predict(prev_z)
        predicted_latent = mdn.sample_from_output(mixture_params[0], LATENT_SPACE_DIMENSIONALITY, self.num_mixtures, temp=self.temperature)
        mixture_weights = softmax(mixture_params[0][-self.num_mixtures:], t=self.temperature)

        self.z = predicted_latent

        return predicted_latent, mixture_weights



