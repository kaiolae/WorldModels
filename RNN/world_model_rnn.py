import keras
from keras.callbacks import EarlyStopping
#Parameters of the MDN-RNN
import mdn

LATENT_VECTOR_SIZE = 64
NUM_LSTM_UNITS = 256
ACTION_DIMENSIONALITY = 1 #TODO Is this right?

class RNN():

    def __init__(self, sequence_length = None, decoder_mode = False, num_mixtures = 5):
        if decoder_mode:
            self.models = self._build_decoder(num_mixtures)
        else:
            assert(sequence_length!=None)
            self.models = self._build(sequence_length, num_mixtures) #TODO Do I need sequence length?
        self.model = self.models[0] #Used during training
        self.forward = self.models[1] #Used during prediction


    def _build(self, sequence_length, num_mixtures):

        #TODO: Now, what to do about the fact that episodes may have different lengths?
        #I'll start with just getting this to work for fixed-length sequences, then add dying and variable length after.

        #Building MDN-RNN
        #Testing to build this like the other Keras World Model implementation - because I need to capture hidden states.

        # Seq-to-seq predictions from https://github.com/cpmpercussion/keras-mdn-layer/blob/master/notebooks/MDN-RNN-time-distributed-MDN-training.ipynb

        #### THE MODEL THAT WILL BE TRAINED
        #TODO Do I need to give the seq length here, or is that flexible?
        #inputs = keras.layers.Input(shape=(sequence_length, LATENT_VECTOR_SIZE+ACTION_DIMENSIONALITY), name='inputs')
        inputs = keras.layers.Input(shape=(sequence_length, LATENT_VECTOR_SIZE+ACTION_DIMENSIONALITY), name='inputs')
        lstm_output = keras.layers.LSTM(NUM_LSTM_UNITS, name='lstm', return_sequences=True)(inputs)
        #lstm_layer = keras.layers.LSTM(NUM_LSTM_UNITS, name='lstm', return_sequences=True, return_state=True)

        #TODO If I want to use the internal RNN state for agent control, I can plug it in here.
        #lstm_output, _, _ = lstm_layer(inputs) #This is the trick to not pass the returned_states to the mdn!
        #mdn = Dense(GAUSSIAN_MIXTURES * (3 * Z_DIM))(lstm_output)  # + discrete_dim
        mdn_output = keras.layers.TimeDistributed(mdn.MDN(LATENT_VECTOR_SIZE, num_mixtures, name='mdn_outputs'), name='td_mdn')(lstm_output)

        rnn = keras.models.Model(inputs=inputs, outputs=mdn_output)

        #### THE MODEL USED DURING PREDICTION
        #TODO Do I really need this forward-model?
        #state_input_h = keras.Input(shape=(NUM_LSTM_UNITS,))
        #state_input_c = keras.Input(shape=(NUM_LSTM_UNITS,))
        #state_inputs = [state_input_h, state_input_c]
        #_, state_h, state_c = lstm_layer(rnn_x, initial_state=[state_input_h, state_input_c])

        #forward = keras.Model([rnn_x] + inputs, [state_h, state_c])
        rnn.summary()
        #default Adam LR is 0.001. Trying half of that.
        #adam = keras.optimizers.Adam(lr=0.0005)
        adam = keras.optimizers.Adam()
        rnn.compile(loss=mdn.get_mixture_loss_func(LATENT_VECTOR_SIZE,num_mixtures), optimizer=adam)

        return (rnn, None)

    #TODO Having trouble with compiling. Is it my code, or the MDN-RNN that does not handle seq-to-seq problems?
    #Testing with a seq-to-1 setup. Could still learn to predict??
    def _build_sequential(self, sequence_length, num_mixtures):

        # The RNN-mdn code from https://github.com/cpmpercussion/creative-prediction/blob/master/notebooks/7-MDN-Robojam-touch-generation.ipynb
        model=keras.Sequential()
        model.add(keras.layers.LSTM(NUM_LSTM_UNITS, input_shape=(sequence_length, LATENT_VECTOR_SIZE+ACTION_DIMENSIONALITY),
                                return_sequences=False, name="Input_LSTM"))
        # TODO Return sequences returns the hidden state, and feeds that to the next layer. When I do this with the MDN,
        # I get an error, because it does not expect that input. I need to find a way to store the hidden state (for the
        # controller) without return sequences?
        #model.add(keras.layers.LSTM(NUM_LSTM_UNITS))
        model.add(mdn.MDN(LATENT_VECTOR_SIZE, num_mixtures, name="Output_MDN"))


        model.compile(loss=mdn.get_mixture_loss_func(LATENT_VECTOR_SIZE,num_mixtures), optimizer=keras.optimizers.Adam())
        model.summary()
        return (model, None)

    def _build_decoder(self, num_mixtures):
        #Decoder for using the trained model
        decoder = keras.Sequential()
        decoder.add(keras.layers.LSTM(NUM_LSTM_UNITS, batch_input_shape=(1,1, LATENT_VECTOR_SIZE+ACTION_DIMENSIONALITY),
                                return_sequences=False, stateful=True, name="Input_LSTM"))
        decoder.add(mdn.MDN(LATENT_VECTOR_SIZE, num_mixtures, name="decoder_output_MDN"))
        decoder.compile(loss=mdn.get_mixture_loss_func(LATENT_VECTOR_SIZE, num_mixtures), optimizer=keras.optimizers.Adam())
        decoder.summary()

        #decoder.load_weights(path_to_weights)
        return (decoder, None)


    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output, epochs, batch_size, validation_split=0.2):
        #Stops training if val loss stops improving.
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')

        #Stops training if val loss stops improving.
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        nan_callback = keras.callbacks.TerminateOnNaN()


        callbacks_list = [checkpoint_callback, nan_callback]#[earlystop]
        print("RNN input shape ", rnn_input.shape)
        print("RNN output shape ", rnn_output.shape)

        return self.model.fit(rnn_input, rnn_output,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=validation_split,
                              verbose=1, callbacks=callbacks_list)



    def save_weights(self, filepath):
        self.model.save_weights(filepath)
