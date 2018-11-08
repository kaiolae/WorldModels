for i in `seq 1 5`;
do
python3 04_train_rnn.py --epochs 50 --num_mixtures 1 --output_folder_name run$i --training_data_file rnn_data_64_dim/rnn_training_data.npz
python3 04_train_rnn.py --epochs 50 --num_mixtures 2 --output_folder_name run$i --training_data_file rnn_data_64_dim/rnn_training_data.npz
python3 04_train_rnn.py --epochs 50 --num_mixtures 4 --output_folder_name run$i --training_data_file rnn_data_64_dim/rnn_training_data.npz
python3 04_train_rnn.py --epochs 50 --num_mixtures 8 --output_folder_name run$i --training_data_file rnn_data_64_dim/rnn_training_data.npz
done
