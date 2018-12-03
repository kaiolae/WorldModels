# Trains the VAE one batch at a time, also doing 10 epochs.
python 02_train_vae.py --start_batch 0 --max_batch 0 --epochs 10 --savefolder vae_from_large_dataset/ --input_data_folder /mnt/data/doom_data_large/ --new_model
for batch in `seq 1 19`;
do
python 02_train_vae.py --max_batch $batch --start_batch $batch --epochs 10 --savefolder vae_from_large_dataset/ --input_data_folder /mnt/data/doom_data_large/
done
