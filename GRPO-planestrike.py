# Adapted from https://superb-makemake-3a4.notion.site/group-relative-policy-optimization-GRPO-18c41736f0fd806eb39dc35031758885

import gymnasium as gym
import torch
import numpy as np
from collections import deque
from planestrike import PlaneStrikeEnv
from gymnasium.wrappers import FlattenObservation
import wandb

BOARD_SIZE = 8
PLANE_SIZE = 8
GAMMA = 0.9

wandb.login()

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project='GRPO-planestrike'
)

class PolicyNet(torch.nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        # self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(BOARD_SIZE**2, 2*BOARD_SIZE**2)
        self.fc2 = torch.nn.Linear(2*BOARD_SIZE**2, BOARD_SIZE**2)

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def collect_trajectory(env, net):
    observation, _ = env.reset()
    log_probs = []
    observations = []
    chosen_actions = []
    episode_reward = 0

    for t in range(BOARD_SIZE**2):
        if isinstance(observation, tuple):
            observation = observation[0]        
        observations.append(observation)
        
        input = torch.from_numpy(observation).float()
        logits = net(input)
        probs = torch.nn.functional.softmax(logits, dim=0)
        action = torch.multinomial(probs, 1).item()

        observation, reward, done, info = env.step(action)
        log_prob = torch.log(probs[action])
        log_probs.append(log_prob.item())
        chosen_actions.append(action)
        episode_reward += reward * (GAMMA**t)

        if done:
            break

    normalized_reward = episode_reward / PLANE_SIZE # because max reward possible in this env is 8
    return observations, log_probs, chosen_actions, normalized_reward


def grpo_update(trajectories, net, optimizer, n_iterations=20):
    rewards = [r for o, l, a, r in trajectories]
    mean_reward = sum(rewards) / len(rewards)
    std_reward = np.std(rewards) + 1e-8
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    for i_iter in range(n_iterations):
        loss = 0
        # iterating over each trajectory in the group
        for traj, advantage in zip(trajectories, advantages):
            (observations, log_probs, chosen_actions, _) = traj
            trajectory_loss = 0
            # iterating over each time step in the trajectory
            for t in range(len(observations)):              
                new_policy_probs = torch.nn.functional.softmax(net(torch.from_numpy(observations[t]).float()), dim=0)
                new_log_probs = torch.log(new_policy_probs)[chosen_actions[t]]

                ratio = torch.exp(new_log_probs - log_probs[t])
                clipped_ratio = torch.clamp(ratio, min=1 - eps, max=1 + eps)
                trajectory_loss += -clipped_ratio * advantage
            trajectory_loss /= len(observations)
            loss += trajectory_loss
        loss /= len(trajectories)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


gym.register(
    id="gymnasium_env/PlaneStrike-v0",
    entry_point=PlaneStrikeEnv,
)

env = gym.make('gymnasium_env/PlaneStrike-v0', board_size=BOARD_SIZE)
net = PolicyNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
episode_reward_window = deque(maxlen=100)

# GRPO specific parameters
trajectories_per_update = 5  # group size
# epsilon for clipping
eps = 0.2

# training loop
for i_episode in range(500000):
    trajectories = []
    episode_rewards = []

    import concurrent.futures

    def collect_trajectory_wrapper(_):
        return collect_trajectory(env, net)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(collect_trajectory_wrapper, range(trajectories_per_update)))

    for result in results:
        observations, log_probs, chosen_actions, normalized_reward = result
        trajectories.append((observations, log_probs, chosen_actions, normalized_reward))
        episode_rewards.append(normalized_reward * PLANE_SIZE)  # unnormalize for tracking

    # update policy using grpo on the collected trajectories
    grpo_update(trajectories, net, optimizer)

    episode_reward_window.extend(episode_rewards)
    avg_reward = sum(episode_reward_window) / len(episode_reward_window)

    if avg_reward > 7.5:
        print('solved at episode', i_episode)
        print(f'avg reward: {avg_reward:.2f}')
        break

    if i_episode % 50 == 0:
        print(f'episode {i_episode}, avg reward: {avg_reward:.2f}')
        wandb.log(data={'avg reward': avg_reward}, step=i_episode)

env.close()
