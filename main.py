#%%

from Model import DQN
from GYM import Pacman
from Agent import Agent

import os
import pickle
import torch
import matplotlib.pyplot as plt


def get_index(STORE, save=True):

    if not os.path.exists(STORE):
        os.makedirs(STORE)

    try:
        f = open(f"{STORE}/index.txt", "r")
        name = int(f.read())+1
    except:
        name = 1

    if save:
        f = open(f"{STORE}/index.txt", 'w+')
        f.write(str(name))
        f.close()
    return f'index_{str(name)}'


def save_config(STORE, agent_name, config):


    path = os.path.join(STORE, 'config')
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{os.path.join(path, agent_name)}.pkl', 'wb') as f:
        pickle.dump(config, f)


if __name__ == '__main__':

    do_trian = True
    do_eval = False
    do_preprocesing = True


    STORE = 'rezults'
    agent_name = get_index(STORE=STORE, save=do_trian)


    n_channels = 3 if do_preprocesing == False else 1
    num_episodes = 800
    start_steps = 500
    learning_rate = 0.0001


    discount_factor = 0.97
    eps_decay_steps = 250000


    batch_size = 48
    copy_steps = 10000
    steps_train = 4
    vizual_on_epoch = 40


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = Pacman(do_preprocesing=do_preprocesing)


    o = env.preprocess_observation(env.reset())
    input_shape = (None, o.shape[0], o.shape[1], o.shape[2] if len(o.shape)==3 else 1)
    image_size = (o.shape[0], o.shape[1], o.shape[2] if len(o.shape)==3 else 1)

    # observation = env.reset()

    # import matplotlib.pyplot as plt
    # plt.imshow(env.preprocess_observation(observation))
    # plt.savefig('img.png')

    # plt.show()

    # plt.imshow(observation)

    # plt.show()


#%%



    m = env.get_action_meanings()


    dqn_config = {
        "state_size": image_size, 
        "channels": n_channels, 
        "action_size": env.action_space()
    }

    agent_config = {
        "env": env,
        "epsilon": 0.5,
        "eps_min": 0.05,
        "eps_max": 1.0,
        "eps_decay_steps": eps_decay_steps,
        "n_outputs": env.action_space(),
        "buffer_len": 20000,
        "current": DQN(**dqn_config), 
        "target": DQN(**dqn_config),
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "store": STORE,
        "agent_name": agent_name,
        "device": device,
        "batch_size": batch_size
    }

    save_config(STORE=STORE, agent_name=agent_name, config=agent_config)

    agent = Agent(**agent_config)

    if do_trian:
        agent.train(discount_factor=discount_factor, num_episodes=num_episodes, start_steps=start_steps, 
                copy_steps=copy_steps, steps_train=steps_train, vizual_on_epoch=vizual_on_epoch)

    if do_eval:
        agent.load(agent_name='index_7')
        agent.evaluate()




# %%

# %%
