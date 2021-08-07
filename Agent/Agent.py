import os
import torch
import numpy as np
from collections import deque, Counter

from Agent.ReplayBuffer import ReplayMemory
from Agent.Visualize import visualize_result

class Agent:

    def __init__(self, env, current, target, epsilon=0.5, eps_min = 0.05, eps_max = 1.0, 
                        eps_decay_steps = 500000, n_outputs=9, buffer_len=20000, 
                        batch_size=128, learning_rate=0.001, optimizer="Adam", store=None, agent_name=None, device=None, *args, **kwargs):

        self.env = env
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_steps = eps_decay_steps
        self.n_outputs = n_outputs
        self.buffer_len = buffer_len
        self.batch_size = batch_size

        self.store = store
        self.agent_name = agent_name
        self.device = device

        # self.exp_buffer = deque(maxlen=buffer_len)
        self.memory = ReplayMemory(capacity=buffer_len, device=self.device)



        # We create "live" and "target" networks from the original paper.
        self.current = current.to(device)
        self.target = target.to(device)

        for p in self.target.parameters():
            p.requires_grad = False
        self.update_target_model()

        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.current.parameters(), lr=learning_rate)

    def epsilon_greedy(self, action, step):
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps) #Decaying policy with more steps
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            return action

    def sample_memories(self, batch_size=None):
        
        if batch_size == None:
            batch_size = self.batch_size


        perm_batch = np.random.permutation(len(self.exp_buffer))[:batch_size]

        mem = np.array(self.exp_buffer)[perm_batch]

        return mem[:,0],mem[:,1],mem[:,2], mem[:,3], mem[:,4]

        return torch.from_numpy(mem[:,0]), torch.from_numpy(mem[:,1]), torch.from_numpy(mem[:,2]), torch.from_numpy(mem[:,3]), torch.from_numpy(mem[:,4])
                        
    def update_target_model(self):
        self.target.load_state_dict(self.current.state_dict())


    def train(self, num_episodes, discount_factor=0.95, start_steps=50, 
                    copy_steps=20, steps_train=4, vizual_on_epoch=None, *args, **kwargs):

        history = []
        td_errors = []

        global_step = 0
        epoch = 0

        for i in range(num_episodes):

            cum_reward = 0
            done = False
            state = self.env.reset()
            actions_counter = Counter()
            episodic_loss = []
            # while the state is not the terminal state

            while not done:

                # get the preprocessed game screen
                state = self.env.preprocess_observation(obs=state, channels=self.target.channels, state_size=self.target.state_size)

                state = torch.from_numpy(state)
                state = state.float()

                with torch.no_grad():
                    actions = self.target(state)

                # get the action
                action = np.argmax(actions, axis=-1)
                actions_counter[str(action)] += 1

                # select the action using epsilon greedy policy
                action = self.epsilon_greedy(action=action, step=global_step)
                # now perform the action and move to the next state, next_obs, receive reward
                next_state, reward, done, _ = self.env.step(action)

                self.memory.push(
                    state=state, 
                    action=action, 
                    reward=reward,
                    next_state=self.env.preprocess_observation(obs=next_state, channels=self.target.channels, state_size=self.target.state_size), 
                    done=done)


                if global_step % steps_train == 0 and global_step > start_steps:

                    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                    qs = self.current(states)


                    qs_selected = torch.sum(qs * actions, dim = 1)



                    with torch.no_grad():
                        qs_next = self.target(next_states)
                        qs_opt = torch.max(qs_next, dim = 1)[0]
                        qs_target = torch.squeeze(rewards) + (1. - torch.squeeze(dones)) * discount_factor * qs_opt


                    with torch.no_grad():
                        td_errors.append(torch.abs(qs_target - qs_selected).mean().item())

                    self.optimizer.zero_grad()
                    loss = (torch.nn.functional.mse_loss(qs_selected, qs_target)).mean() #TODO HUBER LOSS
                    loss.backward()
                    self.optimizer.step()
                    train_loss = torch.mean(loss).item()

                    episodic_loss.append(train_loss)


                # after some interval we copy our main Q network weights to target Q network                
                if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                    self.update_target_model()

                state = next_state
                cum_reward += reward
                global_step += 1

                history.append(cum_reward)
                print('Epochs per episode:', epoch, 'Episode Reward:', cum_reward, 'Episode number:', len(history))

            if (vizual_on_epoch != None) & (epoch + 1) % vizual_on_epoch == 0:
                fig = visualize_result(returns=history, td_errors=td_errors, policy_errors=None)

                if self.store != None:
                    fig.savefig(os.path.join(self.store, 'plots', self.agent_name + '.png'), dpi=400)
                    self.save()


            epoch += 1


    def save(self):
        path = os.path.join(self.store, 'agents', self.agent_name + '.pth')
        torch.save(self.target.state_dict(), path)

    def load(self, agent_name):
        path = os.path.join(self.store, 'agents', agent_name + '.pth')
        self.current.load_state_dict(torch.load(path))
        self.target.load_state_dict(torch.load(path))



    def evaluate(self):

        os.environ['DISPLAY'] = ':0'


        observation = self.env.reset()
        new_observation = observation
        done = False
        actions_counter = Counter()

        while True:
            #set input to network to be difference image
            state = self.env.preprocess_observation(obs=observation, channels=self.target.channels, state_size=self.target.state_size)

            state = torch.from_numpy(state)
            state = state.float()

            # feed the game screen and get the Q values for each action
            with torch.no_grad():
                actions = self.target(state)
            # get the action
            action = np.argmax(actions, axis=-1)
            actions_counter[str(action)] += 1
            # select the action using epsilon greedy policy
            # action = epsilon_greedy(action, global_step)
            self.env.render()
            observation = new_observation
            # now perform the action and move to the next state, next_obs, receive reward
            new_observation, reward, done, _ = self.env.step(action)
            if done:
                # observation = self.env.reset()
                break
        self.env.close()

    