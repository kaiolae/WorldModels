from keras.datasets import cifar10
from VAE.world_model_vae import VAE

#Size of the cifar-images
original_img_size = (32, 32, 3)

#Load dataset
(x_train, _), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

#Training
vae = VAE()
vae.train(x_train)