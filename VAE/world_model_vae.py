#Much of this is from https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459

import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

#TODO Consider extracting these parameters as function arguments

#Tried to make these parameters match the original World Models paper.

INPUT_DIM = (32,32,3)
#INPUT_DIM = (64,64,3)

#Encoding parameters
CONV_FILTERS = [32,64,64,128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

#Decoding parameters
CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 64 #Dimensions in latent space

#Training parameters
EPOCHS = 1
BATCH_SIZE = 100


#Samples z-values, given the vectors of means and deviations.
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE():

    def __init__(self):
        print("VAE init")
        self.models = self._build()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        print("VAE init done")


    def _build(self):

        vae_x = Input(shape=INPUT_DIM)
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0])(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0])(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0])(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0])(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM)(vae_z_in)
        vae_z_log_var = Dense(Z_DIM)(vae_z_in)

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(Z_DIM,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1,1,DENSE_SIZE))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])
        vae_d4_model = vae_d4(vae_d3_model)

        #### DECODER ONLY

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        #### MODELS

        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        #Reconstruction loss. Mean squared error between input image and reconstruction.
        def vae_r_loss(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            #Mean squared error -same as original paper.
            return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)

        #KL-loss. Ensures the probability distribution modelled by Z behaves nicely.
        #Follows formula from original paper. Difference from Keras cookbook: There, this loss was multiplied by
        #-0.0005 instead of -0.5. Weird.
        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)

        #Final loss just sums the two. In Keras, the mean, rather than sum, was used here. Shouldn't make a difference?
        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred)# + vae_kl_loss(y_true, y_pred)
        print("Compiling")
        #Compiling the network, and returning the models.
        vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

        return (vae, vae_encoder, vae_decoder)

    #Loading weights from file
    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    #Training the VAE
    def train(self, data, validation_split=0.2):

        #Not part of original code. Consider dropping.
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')
        callbacks_list = [earlystop]

        print("VAE input: ", data.shape)

        self.model.fit(data, data,
                       shuffle=True,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_split=validation_split,
                       callbacks=callbacks_list)

        self.model.save_weights('./vae/weights.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)


    #Kept from the old code. Consider if it can be better to have this elsewhere.
    def generate_rnn_data(self, obs_data, action_data):
        rnn_input = []
        rnn_output = []

        for i, j in zip(obs_data, action_data):
            rnn_z_input = self.encoder.predict(np.array(i))
            conc = [np.concatenate([x, y]) for x, y in zip(rnn_z_input, j)]
            rnn_input.append(conc[:-1])
            rnn_output.append(np.array(rnn_z_input[1:]))

        rnn_input = np.array(rnn_input)
        rnn_output = np.array(rnn_output)

        return (rnn_input, rnn_output)

    #Some functions to test the VAE

    #Generates latent z-values for all pictures in one rollout.
    def generate_latent_variables(self,input):
        z_mean = self.encoder.predict(input, batch_size=1)
        return z_mean

    #Regenerates pictures based on latent values.
    def generate_picture_from_latent(self, latent_variables):
        decoded_images = self.decoder.predict(latent_variables)
        return decoded_images
