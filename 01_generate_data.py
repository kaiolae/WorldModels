# xvfb-run -s "-screen 0 1400x900x24" python generate_data.py car_racing --total_episodes 200 --start_batch 0 --time_steps 300
#TODO Update and test on Doom.

import numpy as np
import random
import config
# import matplotlib.pyplot as plt

from envcar import make_env #For Doom: Import from env instead
import os
import argparse
from cv2 import resize
#Ha used 10,000 episodes. Minimum length 100, max 2100.

SCREEN_X = 64
SCREEN_Y = 64

def _process_frame(frame):
  obs = np.array(frame[0:400, :, :]).astype(np.float)/255.0
  obs = np.array(resize(obs, (SCREEN_Y, SCREEN_X)))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs


def main(args):
    print("main")
    env_name = args.env_name
    total_episodes = args.total_episodes
    start_batch = args.start_batch
    time_steps = args.time_steps
    render = False #args.render
    batch_size = args.batch_size
    run_all_envs = args.run_all_envs

    store_folder = args.store_folder
    if not os.path.exists(store_folder):
        os.makedirs(store_folder)

    if run_all_envs:
        envs_to_generate = config.train_envs
    else:
        envs_to_generate = [env_name]

    print("envs:", envs_to_generate)
    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))

        env = make_env(current_env_name)
        s = 0
        batch = start_batch

        batch_size = min(batch_size, total_episodes)

        total_frames = 0
        while s < total_episodes:
            obs_data = []
            action_data = []

            for i_episode in range(batch_size):
                print('-----')
                observation = env.reset()
                #observation = config.adjust_obs(observation)

                # plt.imshow(observation)
                # plt.show()

                env.render()
                done = False
                action = np.random.rand() *2.0 -1.0
                t = 0
                obs_sequence = []
                action_sequence = []
                repeat = np.random.randint(1, 11)

                while t < time_steps:  # and not done:
                    t = t + 1
                    if t==1 or t % repeat == 0:
                        #action = np.random.rand() * 2.0 - 1.0
                        #For Doom, the action above is correct. For carracing action has 3 numbers
                        steer_action = np.random.rand()*2.0-1.0
                        gas_action = np.random.rand()-0.5
                        brake_action = np.random.rand()-0.5
                        repeat = np.random.randint(1, 11)
                    action=np.array([steer_action, gas_action, brake_action])
                    action_sequence.append(action)

                    observation, reward, done, info = env.step(action)
                    observation = _process_frame(observation)

                    obs_sequence.append(np.array(observation))


                    if render:
                        env.render()

                    if done: #If we were killed
                        break

                total_frames += t
                print("dead at", t, "total recorded frames for this worker", total_frames)


                obs_data.append(np.array(obs_sequence))
                action_data.append(np.array(action_sequence))

                print("Batch {} Episode {} finished after {} timesteps".format(batch, i_episode, t + 1))
                print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

                s = s + 1

            print("Saving dataset for batch {}".format(batch))
            np.save(store_folder+'/obs_data_' + current_env_name + '_' + str(batch), obs_data)
            print("Saving actions for batch {}".format(batch))
            np.save(store_folder+'/action_data_' + current_env_name + '_' + str(batch), action_data)

            batch = batch + 1

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('env_name', type=str, help='name of environment')
    parser.add_argument('--store_folder', type=str, help='where to store rollouts')
    parser.add_argument('--total_episodes', type=int, default=200, help='total number of episodes to generate')
    parser.add_argument('--start_batch', type=int, default=0, help='start_batch number')
    parser.add_argument('--time_steps', type=int, default=2100, help='how many timesteps at start of episode?')
    parser.add_argument('--render', action='store_true', help='render the env as data is generated')
    parser.add_argument('--batch_size', type=int, default=200, help='how many episodes in a batch (one file)')
    parser.add_argument('--run_all_envs', action='store_true',
                        help='if true, will ignore env_name and loop over all envs in train_envs variables in config.py')

    args = parser.parse_args()
    main(args)
