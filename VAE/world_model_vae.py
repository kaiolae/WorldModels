import os

import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import cifar10
from keras import layers
import keras

img_rows, img_cols, img_chns = 64, 64, 3
latent_dim = 64
intermediate_dim = 128
epsilon_std = 1.0
epochs = 50
filters = 32
num_conv = 3
batch_size = 256

img_size = (img_rows, img_cols, img_chns)
original_dim = img_rows * img_cols * img_chns


class VAE():

    def __init__(self):
        print("VAE init")
        self.models = self._build()
        self.model = self.models[0] #The full VAE model
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = (img_rows, img_cols, img_chns)
        self.z_dim = latent_dim
        print("VAE init done")

    def _build(self):
        # Enc
        input_img = Input(shape=img_size, name='encoder_input')
        x = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu')(input_img)
        x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(x)
        # x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x) # try a max pooling layer here instead of the previous stride
        x = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(x)
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(intermediate_dim, activation='relu', name='latent_project')(x)

        print("Shape before flattening:", shape_before_flattening)

        # mean and var
        z_mean = Dense(latent_dim, name='Z_mean')(x)
        z_log_var = Dense(latent_dim, name='Z_var')(x)

        # make an encoder model (not used until after training)
        encoder = Model(input_img, z_mean)

        # sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
            return z_mean + K.exp(z_log_var) * epsilon

        z = layers.Lambda(sampling, name="Z_sample")([z_mean, z_log_var])

        # dec
        decoder_input = layers.Input(K.int_shape(z)[1:])
        y = Dense(intermediate_dim, activation='relu')(decoder_input)  # (z)
        y = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(y)
        y = Reshape(shape_before_flattening[1:])(y)
        y = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu',
                            name='deconv_1')(y)  # deconv 1
        y = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=(2, 2), activation='relu',
                            name='deconv_2')(y)  # deconv 2
        y = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',
                            name='deconv_3')(y)  # deconv 3, upsamp
        y = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid', name="mean_squash")(y)  # mean squash
        decoder = Model(decoder_input, y, name="Decoder")
        z_decoded = decoder(z)  # y

        def xent(y_true, y_pred):
            return keras.metrics.binary_crossentropy(y_true, y_pred)

        def kl_measure(loc, log_var):
            return -0.5 * K.mean(1 + log_var - K.square(loc) - K.exp(log_var), axis=-1)

        def kl_custom_metric(y_true, y_pred):
            # Ignore input and take from z tensors.
            return kl_measure(z_mean, z_log_var)

        class VAELayer(keras.layers.Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(VAELayer, self).__init__(**kwargs)

            def vae_loss(self, x, z_decoded):
                x = K.flatten(x)
                z_decoded = K.flatten(z_decoded)
                r_loss = original_dim * xent(x, z_decoded)
                kl_loss = kl_measure(z_mean, z_log_var)
                print("KL Shape:", K.int_shape(kl_loss))
                print("Xent shape:", K.int_shape(r_loss))
                return K.mean(r_loss + kl_loss)

            def call(self, inputs):
                x = inputs[0]
                z_decoded = inputs[1]
                loss = self.vae_loss(x, z_decoded)
                self.add_loss(loss, inputs=inputs)
                return x

        y = VAELayer()([input_img, z_decoded])

        vae = Model(input_img, y, name="VAE")
        vae.compile(optimizer='adam', metrics=['mse', 'binary_crossentropy'])

        # def vae_loss(y_true, y_pred):
        #     loss_r = original_dim * xent(y_true, y_pred)
        #     loss_kl = kl_measure(z_mean, z_log_var)
        #     return K.mean(loss_r + loss_kl)

        # end-to-end autoencoder
        # vae = Model(input_img, z_decoded, name="VAE")
        # vae.add_loss(vae_loss)
        # vae.compile(optimizer='adam')

        vae.summary()

        # # encoder, from inputs to latent space
        # encoder = Model(x, z_mean)

        # # generator, from latent space to reconstructed inputs
        # decoder_input = Input(shape=(latent_dim,))
        # _h_decoded = decoder_h(decoder_input)
        # _x_decoded_mean = decoder_mean(_h_decoded)
        return (vae, encoder, decoder)

    # Loading weights from file
    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def load_encoder_weights(self,filepath):
        self.encoder.load_weights(filepath)

    def load_decoder_weights(self,filepath):
        self.decoder.load_weights(filepath)

    # Training the VAE
    def train(self, data, epochs, save_interval=0, savefolder = "./models/"):

        print("VAE input: ", data.shape)

        #For storing models during runs.
        #TODO Add option to specify folder.
        if save_interval:
            filepath = os.path.join(savefolder,"weights-{epoch:02d}-{loss:.2f}.hdf5")
            model_saver = ModelCheckpoint(filepath=filepath, monitor='val_vae_loss',save_best_only=True,
                                          save_weights_only=True, mode='min', period=save_interval)
            callbacks_list = [model_saver]
        else:
            callbacks_list=[]


        return self.model.fit(data,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks_list)

    def save_weights(self, filepath):
        self.model.save_weights(filepath+"full_vae_weights.h5")
        self.encoder.save_weights(filepath+"encoder_only_weights.h5")
        self.decoder.save_weights(filepath+"decoder_only_weights.h5")
        #TODO Consider if we have to save encoder/decoder too.

    # Kept from the old code. Consider if it can be better to have this elsewhere.
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

    # Some functions to test the VAE

    # Generates latent z-values for all pictures in one rollout.
    def generate_latent_variables(self, input):
        z_mean = self.encoder.predict(input, batch_size=1)
        return z_mean

    # Regenerates pictures based on latent values.
    def generate_picture_from_latent(self, latent_variables):
        decoded_images = self.decoder.predict(latent_variables)
        return decoded_images
