train_envs = ['doomrnn']
test_envs = ['doomrnn']

def adjust_obs(obs):
    return obs.astype('float32') / 255.
