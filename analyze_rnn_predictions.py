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

import random
#Importing the VAE
from VAE.world_model_vae import VAE
from RNN.world_model_rnn import RNN
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML

CHANGE_ACTION_PROB = 0.1
LATENT_SPACE_DIMENSIONALITY = 64
RNN_SIZE = 512
VAE_PATH = "/home/kaiolae/code/word_models_keras_test/WorldModels/dec6_models/final_full_vae_weights.h5"

###Helper methods for post-processing and analyses.

def get_random_starting_sequence(obs_data, action_data, minimal_length=100):
    #Gets a random sequence for starting dreams with real data (note: I usually don't do this - only for some
    #specific tests).
    rand_seq_id = random.randint(0,len(obs_data))
    rand_seq = obs_data[rand_seq_id]
    while len(rand_seq) < minimal_length:
        rand_seq_id = random.randint(0,len(obs_data))
        rand_seq = obs_data[rand_seq_id]
    return obs_data[rand_seq_id], action_data[rand_seq_id]

def plot_movie_mp4(image_array):
    dpi = 2.0
    xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    #fig = plt.figure(figsize=(1,1), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
    display(HTML(anim.to_html5_video()))


def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature."""
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist

def sample_from_one_specific_mixture(mdn, mixture_number, params, output_dim, total_num_mixes, sigma_temp):

    #To make dreams only from 1 of the trained mixtures - helps see what's inside there.
    mus, sigs, pi_logits = mdn.split_mixture_params(params, output_dim, total_num_mixes)
    mus_vector = mus[mixture_number*output_dim:(mixture_number+1)*output_dim]
    sig_vector = sigs[mixture_number*output_dim:(mixture_number+1)*output_dim] * sigma_temp  # adjust for temperature
    cov_matrix = np.identity(output_dim) * sig_vector
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample



class RNNAnalyzer:

    def __init__(self, rnn_load_path, num_mixtures, temperature, io_scaling = 1.0):
        self.vae=VAE()
        self.vae.set_weights(VAE_PATH)

        self.rnn = RNN(decoder_mode=True, num_mixtures=num_mixtures)
        self.last_loaded_from = rnn_load_path
        self.rnn.set_weights(rnn_load_path)
        self.frame_count = 0
        self.z = None
        self.num_mixtures = num_mixtures
        self.temperature = temperature
        self.ioscaling = io_scaling

        self.prev_action = 0

    def _reset(self):
        self.rnn.set_weights(self.last_loaded_from)
        self.z = None

    #Decode a sequence with the VAE and visualize it
    def decode_and_visualize(self, latent_vector_sequence):
        plot_movie_mp4(self.decode_with_vae(latent_vector_sequence))

    def generate_random_action(self):
        #Generates random actions with a high probability of repeating the previous one.
        if random.random()<CHANGE_ACTION_PROB:
            self.prev_action = random.uniform(-1.0,1.0)
        return self.prev_action


    def decode_with_vae(self, latent_vector_sequence):
        reconstructions = self.vae.decoder.predict(np.array(latent_vector_sequence))
        return reconstructions

    def warm_up_lstm(self, actions, latent_vectors):
        #Warms up the LSTM with actual data - getting it into a "realistic" state.
        latent_vectors = np.multiply(latent_vectors, self.ioscaling)
        for i in range(latent_vectors.shape[0]):
            self.predict_one_step(actions[i], latent_vectors[i])


    def warm_up_lstm_with_single_input(self, single_latent_vector, warm_up_steps):
        #Warms up the LSTM with self-generated data. In effect, this just discards the initial frames
        #when we later measure something - since these unconditioned frames are not representattive of the
        #actual training.
        latent_vector = np.multiply(single_latent_vector, self.ioscaling)

        predicted_latent, _ = self.predict_one_step(self.generate_random_action(), latent_vector)
        for i in range(warm_up_steps):
            random_action = self.generate_random_action()
            predicted_latent, _ = self.predict_one_step(random_action, predicted_latent)

    #TODO Before using these predictions, perhaps I need to condition it for 60 timesteps first, to get it into a good state?
    def predict_one_step(self, action, previous_z=[], sigma_temp = 1.0, force_prediction_from_mixture = -1):
        #Predicts one step ahead from the previous state.
        #If previous z is given, we predict with that as input. Otherwise, we dream from the previous output we generated.

        #Scaling inputs
        if len(previous_z) > 0:
            previous_z = np.array(previous_z)
            previous_z = np.multiply(previous_z, self.ioscaling)

        self.frame_count += 1
        prev_z = np.zeros((1, 1, LATENT_SPACE_DIMENSIONALITY))
        if len(previous_z)>0:
            prev_z[0][0] = previous_z
        else:
            prev_z[0][0] = self.z

        rnn_input = np.append(prev_z[0][0], action)

        #print("Inserting to RNN:")
        #print(rnn_input)
        mixture_params = self.rnn.model.predict(np.array([[rnn_input]]))

        #If requested, sample from one specific mixture
        if force_prediction_from_mixture != -1:
            predicted_latent = sample_from_one_specific_mixture(mdn, force_prediction_from_mixture, mixture_params[0], LATENT_SPACE_DIMENSIONALITY, self.num_mixtures, sigma_temp)
        else:
            predicted_latent = mdn.sample_from_output(mixture_params[0], LATENT_SPACE_DIMENSIONALITY, self.num_mixtures, temp=self.temperature, sigma_temp=sigma_temp)
        mixture_weights = softmax(mixture_params[0][-self.num_mixtures:], t=self.temperature)
        #print("Got out from RNN after sampling: ")
        #print(predicted_latent)
        #Downscaling to output size.
        #predicted_latent = np.divide(predicted_latent, self.ioscaling)
        self.z = predicted_latent

        return predicted_latent[0], mixture_weights


if __name__ == "__main__":
    print("TEst")
    rnn_path = "rnn_model_64_dim/rnn_trained_model.h5"
    analyzer = RNNAnalyzer(rnn_path, 5, 1.0)

    #Get data to test predictions.
    #Getting data to feed into the VAE and RNN
    import numpy as np
    data = np.load("rnn_data_64_dim/rnn_training_data.npz")
    action_file = data['action']
    latent_file = data['latent']

    single_action_sequence = action_file[6] #A random sequence.
    single_latent_sequence = latent_file[6]
    print("Actions length: ", len(single_action_sequence))
    print("Latent vectors length: ", len(single_latent_sequence))

    pred_lat, mix_weights = analyzer.predict_one_step(single_action_sequence[0], single_latent_sequence[0])
    print("Pred lat is ", pred_lat)
    print("mix weights is ", mix_weights)
