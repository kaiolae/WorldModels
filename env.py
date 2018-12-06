import numpy as np
import gym
from gym.utils import seeding
from gym import spaces

#import doom_py
import tensorflow as tf

from scipy.misc import imresize as resize
from gym.spaces.box import Box
#from ppaquette_gym_doom.doom_take_cover import DoomTakeCoverEnv
from vizdoomgym.envs import VizdoomTakeCover
#from doomrnn import reset_graph, model_path_name, model_rnn_size, model_state_space, ConvVAE, Model, hps_sample
from Constants import model_rnn_size, model_state_space, hps_sample

import os
import sys

SCREEN_Y = 64
SCREEN_X = 64

def _process_frame(frame):
  obs = np.array(frame[0:400, :, :]).astype(np.float)/255.0
  obs = np.array(resize(obs, (SCREEN_Y, SCREEN_X)))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

#Doom environment setup by David Ha.
#Similifying it first just to test the VAE. Consider adding features again later.
#TODO Clean up here - consider if VAE and RNN needs to be mixed in here.

class DoomTakeCoverWrapper(VizdoomTakeCover): #DoomTakeCoverEnv):
    def __init__(self, render_mode=False): #, load_model=True):
        super(DoomTakeCoverWrapper, self).__init__()

        self.no_render = True
        if render_mode:
            self.no_render = False
        self.current_obs = None

        #Removing all the models-stuff. I just want to play randomly and render for now.
        #reset_graph()
        #self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
        #self.rnn = Model(hps_sample, gpu_mode=False)

        #if load_model:
        #    self.vae.load_json(os.path.join(model_path_name, 'vae.json'))
        #    self.rnn.load_json(os.path.join(model_path_name, 'rnn.json'))

        #TODO KOEChange
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=())
        self.outwidth = hps_sample.seq_width
        self.obs_size = self.outwidth + model_rnn_size * model_state_space

        self.observation_space = Box(low=0, high=255, shape=(SCREEN_Y, SCREEN_X, 3))
        self.actual_observation_space = spaces.Box(low=-50., high=50., shape=(self.obs_size,))

        #self.zero_state = self.rnn.sess.run(self.rnn.zero_state)

        self._seed()

        #self.rnn_state = None
        #self.z = None
        self.restart = None
        self.frame_count = None
        self.viewer = None
        self.reset()

    def _step(self, action):

        # update states of rnn
        self.frame_count += 1

        #prev_z = np.zeros((1, 1, self.outwidth))
        #prev_z[0][0] = self.z

        prev_action = np.zeros((1, 1))
        prev_action[0] = action

        prev_restart = np.ones((1, 1))
        prev_restart[0] = self.restart

        #s_model = self.rnn

        #feed = {s_model.input_z: prev_z,
        #        s_model.input_action: prev_action,
        #        s_model.input_restart: prev_restart,
        #        s_model.initial_state: self.rnn_state
        #        }

        #self.rnn_state = s_model.sess.run(s_model.final_state, feed)

        # actual action in wrapped env:

        threshold = 0.3333
        #full_action = [0] * 43
        full_action=-1

        #In the new vizdoom, action -1 is stay, 0 is left, 1 is right.
        if action < -threshold:
            #full_action[11] = 1
            full_action = 0

        if action > threshold:
            #full_action[10] = 1
            full_action=1

        #The obs returned here is full resolution, 480 by 640.
        obs, reward, done, _ = super(DoomTakeCoverWrapper, self).step(full_action)
        small_obs = _process_frame(obs) #Reduces OBS resolution
        self.current_obs = small_obs
        #self.z = self._encode(small_obs)

        if done:
            self.restart = 1
        else:
            self.restart = 0

        #return self._current_state(), reward, done, {}
        return self.current_obs, reward, done, {}

    #def _encode(self, img):
    #    simple_obs = np.copy(img).astype(np.float) / 255.0
    #    simple_obs = simple_obs.reshape(1, 64, 64, 3)
    #    mu, logvar = self.vae.encode_mu_logvar(simple_obs)
    #    return (mu + np.exp(logvar / 2.0) * self.np_random.randn(*logvar.shape))[0]

    #def _decode(self, z):
    #    # decode the latent vector
    #    img = self.vae.decode(z.reshape(1, 64)) * 255.
    #    img = np.round(img).astype(np.uint8)
    #    img = img.reshape(64, 64, 3)
    #    return img

    def _reset(self):
        obs = super(DoomTakeCoverWrapper, self).reset()
        small_obs = _process_frame(obs)
        self.current_obs = small_obs
        #self.rnn_state = self.zero_state
        #self.z = self._encode(small_obs)
        self.restart = 1
        self.frame_count = 0
        return small_obs
        #return self._current_state()

    #def _current_state(self):
    #    if model_state_space == 2:
    #        return np.concatenate([self.z, self.rnn_state.c.flatten(), self.rnn_state.h.flatten()], axis=0)
    #    return np.concatenate([self.z, self.rnn_state.h.flatten()], axis=0)

    def _seed(self, seed=None):
        if seed:
            tf.set_random_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None  # If we don't None out this reference pyglet becomes unhappy
            return
        try:
            state = self.game.get_state()
            img = state.image_buffer
            small_img = self.current_obs
            if img is None:
                img = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
            if small_img is None:
                small_img = np.zeros(shape=(SCREEN_Y, SCREEN_X, 3), dtype=np.uint8)
            small_img = resize(small_img, (img.shape[0], img.shape[0]))
            #vae_img = self._decode(self.z)
            #vae_img = resize(vae_img, (img.shape[0], img.shape[0]))
            #all_img = np.concatenate((img, small_img, vae_img), axis=1)
            #img = all_img
            img = small_img
            if mode == 'rgb_array':
                return img
            elif mode is 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
#        except doom_py.vizdoom.ViZDoomIsNotRunningException:
        except E: 
            pass  # Doom has been closed

def make_env(env_name, seed=-1, render_mode=False):
  #if env_name == 'doomrnn':
  print('making rnn doom environment')
  env = DoomTakeCoverWrapper(render_mode=render_mode)
    #Disabling car racing from now
  #elif env_name == 'car_racing':
  #  from custom_envs.car_racing import CarRacing
   # env = CarRacing()
   # if (seed >= 0):
   #   env.seed(seed)
  #else:
  #print("couldn't find this env")

  return env
