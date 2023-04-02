import torch
import torch.nn as nn


class dqnAgent(nn.Module):

    def __init__(self, num_ant, num_bits):
        super(dqnAgent, self).__init__()

        self.num_ant = num_ant
        self.num_bits = num_bits
        self.state_dim = 2 * self.num_ant
        self.action_dim = self.num_ant * (2 ** self.num_bits)

        self.gamma = 0
        self.gamma_final = 0
        self.final_epsilon = 0.09
        self.initial_epsilon = 0.99
        self.discount_epsilon = 0.9995
        self.iteration_epsilon = 1000
        self.replay_memory_size = 8192
        self.minibatch_size = 32

        self.fc1 = nn.Linear(self.state_dim, 8*self.state_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(8*self.state_dim, 2*self.action_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(2*self.action_dim, self.action_dim)
        # self.act3 = nn.ReLU()
        # self.fc4 = nn.Linear(2*self.action_dim, self.action_dim)

    def forward(self, x):
        out = self.fc1(x) # x.shape must be (*, self.state_dim)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        # out = self.act3(out)
        # out = self.fc4(out)

        return out # out.shape must be (*, self.action_dim)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)
