# WorldModels

## Training

Training the world model consists in 4 steps: 1) Gathering training data, 2) Training the VAE to learn a compressed representation of that data, 3) Converting data to latent vectors with the VAE, and 4) Training the RNN on sequences of latent vectors.

1) Gathering data. Data in this case is sequences of images from gameplay. To gather images from 2000 episodes of doom gameplay, run the command 

```sh
python 01_generate_data.py "doomrnn" --total_episodes 2000 --batch_size 200 --store_folder training_data_folder/
```

2) Training the VAE. To train on the recorded data, you can run the command

```sh
python 02_train_vae.py --savefolder trained_models_folder/ --input_data_folder training_data_folder/ --epochs 10 --start_batch 0 --max_batch 9 --new_model
```

3) Converting data to latent vectors. Example command storing the latent vectors back with the trained models

```sh
python 03_generate_rnn_data.py --obs_folder training_data_folder/ --loaded_vae_weights trained_models_folder/final_full_vae_weights.h5 --savefolder trained_models_folder/
```

4) Training the RNN on the sequences of actions and latent vectors. Example:
```
python 04_train_rnn.py --upper_level_folder_name trained_rnn_models --skip_ahead 3 --epochs 1000 --num_mixtures 5 --output_folder_name run_1 --training_data_file trained_models_folder/rnn_format_data/rnn_training_data.npz
```
## Analyzing

After training, we can analyze how the World Model has learned to make predictions. The scripts in the notebooks-folder contain various ways to make such analyses, including:
<dl>
  <dt><strong>Analyze mixture weights against generated images</strong></dt>
  <dd>Analyzes to which degree specific components of the mixture density RNN correspond to specific situations/events in the predicted images.</dd>
  <dt><strong>Visualize Mixture Weights and Events</strong></dt>
  <dd>Shows the weights of mixture components together with predicted images - allowing us to spot potential relationships</dd>
  <dt><strong>Show Trained RNN Dreams</strong></dt>
  <dd>Shows sequences of predicted images, and allows us to make "committed" sequences, that is, prediction sequences generated from a single mixture component.</dd>
</dl>

See the paper for details on these analyses and how we interpret them.
