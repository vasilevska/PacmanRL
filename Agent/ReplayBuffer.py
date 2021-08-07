import torch
import numpy as np
from collections import namedtuple, deque

from torch._C import device
from cpprb import ReplayBuffer as RB

import warnings
warnings.filterwarnings('ignore')




class ReplayMemory(object):

    def __init__(self, capacity, device=None):
        self.memory = deque([],maxlen=capacity)
        self.device = device


        self.state_dtype=np.float32
        self.action_dtype=np.int64
        self.default_dtype=np.float32


    def push(self, state, action, reward, next_state, done, *args):
        self.memory.append(np.array([
            self._preprocess(var=state, dtype=self.state_dtype), 
            self._preprocess(var=action, dtype=self.action_dtype), 
            self._preprocess(var=reward, dtype=self.default_dtype), 
            self._preprocess(var=next_state, dtype=self.state_dtype), 
            self._preprocess(var=done, dtype=self.action_dtype)
        ]))


    def sample(self, batch_size):
        batch_idxs = np.random.randint(len(self), size=batch_size)
        batches = np.array(self.memory)[batch_idxs]
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in batch_idxs:
            data = self.memory[idx]
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            dones.append(data[4])

        states = torch.from_numpy(np.array(states)).to(self.device)
        actions = torch.from_numpy(np.array(actions).reshape((batch_size, -1))).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).to(self.device)
        dones = torch.from_numpy(np.array(dones)).to(self.device)
       

        return states, actions, rewards, next_states, donesc

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def _preprocess(var, dtype):
        if torch.is_tensor(var):
            var = var.cpu().detach().numpy()
        if hasattr(var, 'shape'):
            var = var.astype(dtype)
            var = np.squeeze(var)
        else:
            var = dtype(var)

        return var



class ReplayBuffer:


    def __init__(self, capacity, state_size, device=None, *args, **kwargs):

        env_dict = {
            "state": {"shape": state_size, "dtype": np.float32},
            "action": {"shape": 1,"dtype": np.int64},
            "reward": {},
            "next_state": {"shape": state_size, "dtype":np.float32},
            "done": {}
            }

        self.memory = RB(capacity, env_dict)
        self.device = device


    def push(self, state, action, reward, next_state, done, *args):
        self.memory.add(
            state=self._to_numpy(state), 
            action=self._to_numpy(action), 
            reward=self._to_numpy(reward), 
            next_state=self._to_numpy(next_state), 
            done=self._to_numpy(done)
            )


    def sample(self, batch_size):

        sample = self.memory.sample(batch_size)
        states = self._to_tensor(sample['state'])
        actions = self._to_tensor(sample['action'])
        rewards = self._to_tensor(sample['reward'])
        next_states = self._to_tensor(sample['next_state'])
        dones = self._to_tensor(sample['done'])

        return states, actions, rewards, next_states, dones

    def _to_numpy(self, val):

        if torch.is_tensor(val):
            return val.cpu().detach()

        return val

    def _to_tensor(self, val):

        if torch.is_tensor(val):
            return val

        return torch.from_numpy(val).to(self.device)
