import os

import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import cifar10

# Parameters - TODO abstract
img_rows, img_cols, img_chns = 64, 64, 3
latent_dim = 32
intermediate_dim = 128
epsilon_std = 1.0
filters = 64
num_conv = 3
batch_size = 100
# tensorflow uses channels_last
# theano uses channels_first
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)


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

        # encoder architecture
        x = Input(shape=original_img_size)
        conv_1 = Conv2D(img_chns,
                        kernel_size=(2, 2),
                        padding='same', activation='relu')(x)
        conv_2 = Conv2D(filters,
                        kernel_size=(2, 2),
                        padding='same', activation='relu',
                        strides=(2, 2))(conv_1)
        conv_3 = Conv2D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_2)
        conv_4 = Conv2D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_3)
        flat = Flatten()(conv_4)
        hidden = Dense(intermediate_dim, activation='relu')(flat)

        # mean and variance for latent variables
        z_mean = Dense(latent_dim)(hidden)
        z_log_var = Dense(latent_dim)(hidden)

        # sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                      mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # decoder architecture
        decoder_hid = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(int(filters * img_rows / 2 * img_cols / 2), activation='relu')

        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, int(img_rows / 2), int(img_cols / 2))
        else:
            output_shape = (batch_size, int(img_rows / 2), int(img_cols / 2), filters)

        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv_1 = Conv2DTranspose(filters,
                                           kernel_size=num_conv,
                                           padding='same',
                                           strides=1,
                                           activation='relu')
        decoder_deconv_2 = Conv2DTranspose(filters,
                                           kernel_size=num_conv,
                                           padding='same',
                                           strides=1,
                                           activation='relu')
        decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                                  kernel_size=(3, 3),
                                                  strides=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        decoder_mean_squash = Conv2D(img_chns,
                                     kernel_size=2,
                                     padding='valid',
                                     activation='sigmoid')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        # Custom loss layer
        """class CustomVariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomVariationalLayer, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean_squash):
                x = K.flatten(x)
                x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
                xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
                kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss + kl_loss)

            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean_squash = inputs[1]
                loss = self.vae_loss(x, x_decoded_mean_squash)
                self.add_loss(loss, inputs=inputs)
                return x
                
            y = CustomVariationalLayer()([x, x_decoded_mean_squash])
                        """


        # Extra methods just for printing the loss.
        # Reconstruction loss. Mean squared error between input image and reconstruction.
        def vae_r_loss(y_true, y_pred):
            #TODO Some small difference here from Ha's code too.
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            # Mean squared error -same as original paper.
            return K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)
            #return img_rows * img_cols * metrics.binary_crossentropy(y_true_flat,y_pred_flat)


        # KL-loss. Ensures the probability distribution modelled by Z behaves nicely.
        # Follows formula from original paper. Difference from Keras cookbook: There, this loss was multiplied by
        # -0.0005 instead of -0.5. Weird.
        def vae_kl_loss(y_true, y_pred):
            #TODO Hardmaru had some additional trick to limit the min value here...
            #Testing the trick to limit min value
            kl_tolerance = 0.01
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            kl_loss = K.maximum(kl_loss, kl_tolerance*latent_dim)
            return K.mean(kl_loss)
            #return - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            #return - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        #Final loss just sums the two. In Keras, the mean, rather than sum, was used here. Shouldn't make a difference?
        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)



        # entire model
        vae = Model(x, x_decoded_mean_squash)
        vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss, vae_loss])
        vae.summary()

        #Just the encoder and decoder
        vae_encoder = Model(x,z)

        #Building a separate decoder
        vae_z_input = Input(shape=(latent_dim,))
        decoded = decoder_hid(vae_z_input)
        decoded = decoder_upsample(decoded)
        decoded = decoder_reshape(decoded)
        decoded = decoder_deconv_1(decoded)
        decoded = decoder_deconv_2(decoded)
        decoded = decoder_deconv_3_upsamp(decoded)
        decoded = decoder_mean_squash(decoded)
        vae_decoder = Model(vae_z_input, decoded)

        return (vae, vae_encoder, vae_decoder)

    # Loading weights from file
    def set_weights(self, filepath):
        self.model.load_weights(filepath)

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


        return self.model.fit(data, data,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks_list)


    def save_weights(self, filepath):
        self.model.save_weights(filepath)
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
