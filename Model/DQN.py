import torch
import torch.nn as nn



class DQN(nn.Module):

    def __init__(self, state_size, channels, action_size, num_hidden=1, kernel_size=8, *args, **kwargs):
        self.state_size=state_size
        self.channels=channels
        self.action_size=action_size
        modules = []

        # prev_ch = channels
        # next_ch = 2*channels
        # # for _ in range(num_hidden):
        # #     modules.append(nn.Conv2d(in_channels=prev_ch, out_channels=next_ch, kernel_size=kernel_size, padding=1, dtype=torch.float32))
        # #     modules.append(nn.BatchNorm2d(next_ch)),
        # #     modules.append(nn.ReLU())
        # #     prev_ch = next_ch
        # #     next_ch = 2*prev_ch

        modules.append(nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(8, 8), stride=4, dtype=torch.float32))
        modules.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, dtype=torch.float32))
        modules.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dtype=torch.float32))

        modules.append(nn.Flatten())
        modules.append(nn.Linear(2688, 128, dtype=torch.float32))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(128, self.action_size, dtype=torch.float32))



        super(DQN, self).__init__()
        self.neuralnet = nn.Sequential(*modules)


    def forward(self, x):

        if len(x.shape) <=3 :
            x = x.reshape((-1, self.channels, self.state_size[0], self.state_size[1]))

        res=self.neuralnet(x)

        return res



