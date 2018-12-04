#Generates batches of data for Keras on the fly, to avoid memory issues.
import copy

import numpy as np


class KerasBatchGenerator(object):

    def __init__(self, observation_data, action_data, sequence_length, batch_size, skip_ahead):
        print("Making generator of size ", len(observation_data))
        #observation_data is an array with shape (num_episodes, num_observations_in_episode, len_observation).
        self.observation_data = []
        self.observations_and_actions = []

        self.x_data = []
        self.y_data = []

        #Populating self data.
        for episode_num in range(len(observation_data)):
            if len(observation_data[episode_num]) < sequence_length+1: #If we can't generate a full sequence, we skip this episode.
                continue
            else:
                self.observation_data.append(observation_data[episode_num])
                observation_and_action_sequence = []
                for step_num in range(0, len(self.observation_data[episode_num]), skip_ahead):
                    starting_frame = step_num*skip_ahead
                    if starting_frame+sequence_length+1 > len(observation_data[episode_num]):
                        break
                    x_sequence = []
                    y_sequence = []
                    for frame_counter in range(starting_frame, starting_frame+sequence_length):
                        observation_with_action = copy.deepcopy(observation_data[episode_num][frame_counter])
                        observation_with_action = np.append(observation_with_action,action_data[episode_num][frame_counter])
                        x_sequence.append(observation_with_action)
                        y_sequence.append(observation_data[episode_num][frame_counter+1])
                    self.x_data.append(x_sequence)
                    self.y_data.append(y_sequence)

        assert(len(self.x_data) > 0)
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        #A dictionary giving a unique key to each element in the data-arrays. Will be useful to ensure I train
        #on every data sample. I don't want to flatten the arrays, since I can't wrap sequences during training.
        self.data_dict={}
        datacounter = 0
        for episode_num in range(len(self.observation_data)):
            for step_num in range(0,len(self.observation_data[episode_num]), skip_ahead):
                self.data_dict[datacounter] = (episode_num, step_num)
                datacounter += 1

        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        #An index for each data sample, shuffled randomly. Go through all these, and we're through the dataset.
        #Random shuffling is important, to get stochastic gradient descent right.
        self.data_indices = np.arange(0,datacounter)
        np.random.shuffle(self.data_indices)

    def get_total_num_train_samples(self):
        return self.data_indices.size

    def generate(self):
        x = np.zeros((self.batch_size, self.sequence_length, len(self.observations_and_actions[0][0])))
        y = np.zeros((self.batch_size, self.sequence_length, len(self.observation_data[0][0])))
        data_counter = 0
        while True:
            #A tuple (episode_num, step_num) saying where the current sequence starts.
            while x.shape[0] < self.batch_size:

                current_episode_num, current_step_num = self.data_dict[self.data_indices[self.current_idx]]
                if current_step_num + self.sequence_length +1 >= len(self.observation_data[current_episode_num]):
                    continue #Skipping sequences that are too short.

                current_observations_and_actions = self.observations_and_actions[current_episode_num]
                current_observations_only = self.observation_data[current_episode_num]
                x[data_counter, :] = current_observations_and_actions[current_step_num:current_step_num + self.sequence_length]
                y[data_counter, :] = current_observations_only[current_step_num+1:current_step_num + self.sequence_length+1]
                print("Generating data. x shape: ", x.shape)
                data_counter+=1
                self.current_idx+=1
                if self.current_idx >= len(self.data_indices):
                    self.current_idx=0 #Epoch done. Wrapping back to beginning of dataset.
            yield x, y
