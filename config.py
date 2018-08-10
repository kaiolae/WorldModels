train_envs = ['doom']
test_envs = ['doom']

def adjust_obs(obs):
    return obs.astype('float32') / 255.