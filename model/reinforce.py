# Trying out the most basic Deep RL algorithm: the REINFORCE algorithm

import numpy as np
import torch.nn as nn
import torch
import gym

EPOCHS = 1000
BATCH_SIZE = 128

def generate_action_sequence(model, starting_observation):
    observations, actions, rewards = [], [], []
    observations.append(starting_observation)
    observation = starting_observation

    while True:
        action = model.sample(torch.tensor(observation).float())
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        observation = obs
        observations.append(observation)
        rewards.append(reward)
        if terminated or truncated:
            observation, info = env.reset(seed=42)
            return observations, actions, actions

obs_dim = 4
act_dim = 1


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.pi_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        ).float()

    # Compute \pi(s, a)
    def forward(self, s, a):
        mean_a = self.pi_net(s)
        model = torch.distributions.normal.Normal(mean_a, 1.0)
        return model.log_prob(a)

    def sample(self, s):
        mean_a = self.pi_net(s)
        model = torch.distributions.normal.Normal(mean_a, 1.0)
        return model.sample()


env = gym.make('InvertedPendulum-v4')
observation, info = env.reset(seed=42)

model = Policy()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(EPOCHS):

    print(f"STARTING EPOCH {i}")

    total_survival = 0

    pseudo_loss = 0 
    for j in range(BATCH_SIZE):
        # Generate the sample
        observations, actions, rewards = generate_action_sequence(model, observation)
        total_survival += len(actions)
        # Concatenate a list of numpy arrays into a torch tensor
        observations = torch.tensor(np.stack(observations[:-1], axis=0))
        actions = torch.tensor(np.stack(actions))
        rewards = torch.tensor(np.stack(rewards))

        observations = observations.float()

        # Compute the log probabilities of doing each of the actions.
        log_probabilities = model.forward(observations, actions)
        pseudo_loss += -(log_probabilities).sum() * rewards.sum() / BATCH_SIZE

    pseudo_loss.backward()
    optimizer.step()

    print(f"MEAN SURVIVAL {total_survival / BATCH_SIZE}")




env.close()
