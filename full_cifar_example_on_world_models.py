import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import cifar10

# Parameters
img_rows, img_cols, img_chns = 64, 64, 3
latent_dim = 64
intermediate_dim = 1024
epsilon_std = 1.0
epochs = 1
batch_size = 100 #TODO Not sure
filters = 64 #TODO Not sure what this variable means


# tensorflow uses channels_last
# theano uses channels_first
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)


# encoder architecture
x = Input(shape=original_img_size)
conv_1 = Conv2D(64,
                kernel_size=(5,5),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(64,
                kernel_size=(5,5),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(32,
                kernel_size=6,
                padding='same', activation='relu',
                strides=2)(conv_2)
conv_4 = Conv2D(3,
                kernel_size=6,
                padding='same', activation='relu',
                strides=2)(conv_3)
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
                                   kernel_size=5,
                                   padding='same',
                                   strides=2,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=5,
                                   padding='same',
                                   strides=2,
                                   activation='relu')
decoder_deconv_3_upsamp = Conv2DTranspose(32,
                                          kernel_size=(6,6),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(3,
                             kernel_size=6,
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
class CustomVariationalLayer(Layer):
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

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


# load World Model data

#TODO Loading just one set of images. Load more later.
wm_images = np.load('./data/obs_data_doomrnn_1.npy')
print("Shape after load: ", wm_images.shape)
wm_images = wm_images.astype('float32') / 255.
wm_images = wm_images.reshape((wm_images.shape[0],) + original_img_size)

# training
history = vae.fit(wm_images,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)
        #validation_data=(x_test, None))

# encoder from learned model
encoder = Model(x, z_mean)

# generator / decoder from learned model
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

# save all 3 models for future use - especially generator
vae.save('../models/world_model_vae.h5')
encoder.save('../models/world_model_encoder.h5')
generator.save('../models/world_model_decoder.h5')

# save training history
fname = 'world_model_training_history.h5'
with open(fname, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)