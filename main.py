from Model import DQN
from GYM import Pacman
from Agent import Agent

import torch
import matplotlib.pyplot as plt


def get_index(STORE, save=True):
    try:
        f = open(f"{STORE}/index.txt", "r")
        name = int(f.read())+1
    except:
        name = 1

    if save:
        f = open(f"{STORE}/index.txt", 'w')
        f.write(str(name))
        f.close()
    return f'index_{str(name)}'



if __name__ == '__main__':

    STORE = './rezults'
    agent_name = get_index(STORE=STORE, save=True)


    image_size = (88, 80)
    n_channels = 1
    num_episodes = 800
    start_steps = 500

    batch_size = 64
    copy_steps = 20
    steps_train = 4
    vizual_on_epoch = 20
    discount_factor = 0.97

    do_trian = True
    do_eval = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_shape = (None, 88, 80, 1)


    env = Pacman()

    dqn_config = {
        "state_size": image_size, 
        "channels": n_channels, 
        "action_size": 9
    }

    agent_config = {
        "env": env,
        "epsilon": 0.5,
        "eps_min": 0.05,
        "eps_max": 1.0,
        "eps_decay_steps": 500000,
        "n_outputs": env.action_space(),
        "buffer_len": 20000,
        "current": DQN(**dqn_config), 
        "target": DQN(**dqn_config),
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "store": STORE,
        "agent_name": agent_name,
        "device": device,
        "batch_size": batch_size
    }

    agent = Agent(**agent_config)

    if do_trian:
        agent.train(discount_factor=discount_factor, num_episodes=num_episodes, start_steps=start_steps, 
                copy_steps=copy_steps, steps_train=steps_train, vizual_on_epoch=vizual_on_epoch)

    if do_eval:
        agent.load(agent_name=agent_name)
        agent.evaluate()



