#Based on the doom settings from original paper.
from collections import namedtuple

model_rnn_size = 512
model_num_mixture = 5
model_restart_factor = 10.
model_state_space = 2 # includes C and H concatenated if 2, otherwise just H

TEMPERATURE = 1.25 # train with this temperature

# hyperparameters for our model. I was doing this on an older tf version, when HParams was not available ...

HyperParams = namedtuple('HyperParams', ['max_seq_len',
                                         'seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'restart_factor',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                        ])

def default_hps():
  return HyperParams(max_seq_len=1000, # train on sequences of 1000
                     seq_width=64,    # width of our data (64)
                     rnn_size=model_rnn_size,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=model_num_mixture,   # number of mixtures in MDN
                     restart_factor=model_restart_factor, # factor of importance for restart=1 rare case for loss.
                     learning_rate=0.001,
                     decay_rate=0.99999,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=0)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=2, use_recurrent_dropout=0, is_training=0)