import torch
import random
import numpy as np
import gym
from datetime import datetime
import matplotlib.pyplot as plt


class Pacman:

  def __init__(self, do_preprocesing=True):
    self.env = gym.make("MsPacman-v0")
    self.do_preprocesing = do_preprocesing

  def action_space(self):
    return self.env.action_space.n

  def get_action_meanings(self):
    return self.env.env.get_action_meanings()

  def reset(self):
    return self.env.reset()

  def step(self, n):
    return self.env.step(n)

  def render(self):
    return self.env.render()

  def close(self):
    return self.env.close()



  def preprocess_observation(self, obs, channels=None, state_size=None, *args, **kwargs):


      if self.do_preprocesing == False:
        return obs



      # Crop and resize the image
      img = obs[1:176:2, ::2]
      return rgb2gray(img)
      # Convert the image to greyscale
      img = img.mean(axis=2)
      # Improve image contrast
      img[img==123] = 0
      # Next we normalize the image from -1 to +1
      img = (img - 128) / 128-1
      # img = torch.from_numpy(img.reshape(channels, state_size[0], state_size[1]))
      # img = img.type(torch.float32)
      return img


  
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

if __name__ == '__main__':
  env = gym.make("MsPacman-v0")
  n_outputs = env.action_space.n
  print(n_outputs)
  print(env.env.get_action_meanings())
  observation = env.reset()
  for i in range(22):
    if i > 20:
      plt.imshow(observation)
      plt.show()
  observation, _, _, _ = env.step(1)

