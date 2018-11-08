import keras

#Parameters of the MDN-RNN
import mdn

LATENT_VECTOR_SIZE = 64
BATCH_SIZE = 64 # Fant ikke Ha's verdi i farta
NUM_LSTM_UNITS = 256
NUM_MIXTURES = 5
ACTION_DIMENSIONALITY = 1 #TODO Is this right?

class RNN():

    def __init__(self, sequence_length = None, decoder_mode = False, path_to_model = ''):
        if decoder_mode:
            assert(path_to_model!='')
            self.models = self._build_decoder()
        else:
            assert(sequence_length!=None)
            self.models = self._build(sequence_length) #TODO Do I need sequence length?
        self.model = self.models[0] #Used during training
        self.forward = self.models[1] #Used during prediction


    def _build(self, sequence_length):

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
        mdn_output = keras.layers.TimeDistributed(mdn.MDN(LATENT_VECTOR_SIZE, NUM_MIXTURES, name='mdn_outputs'), name='td_mdn')(lstm_output)

        rnn = keras.models.Model(inputs=inputs, outputs=mdn_output)

        #### THE MODEL USED DURING PREDICTION
        #TODO Do I really need this forward-model?
        #state_input_h = keras.Input(shape=(NUM_LSTM_UNITS,))
        #state_input_c = keras.Input(shape=(NUM_LSTM_UNITS,))
        #state_inputs = [state_input_h, state_input_c]
        #_, state_h, state_c = lstm_layer(rnn_x, initial_state=[state_input_h, state_input_c])

        #forward = keras.Model([rnn_x] + inputs, [state_h, state_c])
        rnn.summary()
        rnn.compile(loss=mdn.get_mixture_loss_func(LATENT_VECTOR_SIZE,NUM_MIXTURES), optimizer='adam')

        return (rnn, None)

    #TODO Having trouble with compiling. Is it my code, or the MDN-RNN that does not handle seq-to-seq problems?
    #Testing with a seq-to-1 setup. Could still learn to predict??
    def _build_sequential(self, sequence_length):

        # The RNN-mdn code from https://github.com/cpmpercussion/creative-prediction/blob/master/notebooks/7-MDN-Robojam-touch-generation.ipynb
        model=keras.Sequential()
        model.add(keras.layers.LSTM(NUM_LSTM_UNITS, input_shape=(sequence_length, LATENT_VECTOR_SIZE+ACTION_DIMENSIONALITY),
                                return_sequences=False, name="Input_LSTM"))
        # TODO Return sequences returns the hidden state, and feeds that to the next layer. When I do this with the MDN,
        # I get an error, because it does not expect that input. I need to find a way to store the hidden state (for the
        # controller) without return sequences?
        #model.add(keras.layers.LSTM(NUM_LSTM_UNITS))
        model.add(mdn.MDN(LATENT_VECTOR_SIZE, NUM_MIXTURES, name="Output_MDN"))


        model.compile(loss=mdn.get_mixture_loss_func(LATENT_VECTOR_SIZE,NUM_MIXTURES), optimizer=keras.optimizers.Adam())
        model.summary()
        return (model, None)

    def _build_decoder(self, path_to_weights):
        #Decoder for using the trained model - loading it from the given weights.
        decoder = keras.Sequential()
        decoder.add(keras.layers.LSTM(NUM_LSTM_UNITS, batch_input_shape=(1,1, LATENT_VECTOR_SIZE+ACTION_DIMENSIONALITY),
                                return_sequences=False, stateful=True, name="Input_LSTM"))
        decoder.add(mdn.MDN(LATENT_VECTOR_SIZE, NUM_MIXTURES, name="decoder_output_MDN"))
        decoder.compile(loss=mdn.get_mixture_loss_func(LATENT_VECTOR_SIZE, NUM_MIXTURES), optimizer=keras.optimizers.Adam())
        decoder.summary()

        decoder.load_weights(path_to_weights)
        return (decoder, None)


    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output, epochs, validation_split=0.2):
        #earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        #"callbacks_list = [earlystop]
        print("RNN input shape ", rnn_input.shape)
        print("RNN output shape ", rnn_output.shape)

        return self.model.fit(rnn_input, rnn_output,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=BATCH_SIZE,
                       validation_split=validation_split,
                              verbose=1)
                       #callbacks=callbacks_list)


    def save_weights(self, filepath):
        self.model.save_weights(filepath)
