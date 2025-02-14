from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import random

# We always use square board, so only one size is needed
BOARD_SIZE = 8
PLANE_SIZE = 8

# Reward discount factor
GAMMA = 0.5

# Plane direction
PLANE_HEADING_RIGHT = 0
PLANE_HEADING_UP = 1
PLANE_HEADING_LEFT = 2
PLANE_HEADING_DOWN = 3

# Hidden board cell status; 'occupied' means it's part of the plane
HIDDEN_BOARD_CELL_OCCUPIED = 1
HIDDEN_BOARD_CELL_UNOCCUPIED = 0

# Visible board cell status
BOARD_CELL_HIT = 1
BOARD_CELL_MISS = -1
BOARD_CELL_UNTRIED = 0


def play_game(predict_fn):
  """Play one round of game to gather logs for TF/JAX training."""

  env = gym.make('PlaneStrike-v0', board_size=BOARD_SIZE)
  observation = env.reset()

  game_board_log = []
  predicted_action_log = []
  action_result_log = []
  while True:
    probs = predict_fn(np.expand_dims(observation, 0))[0]
    probs = np.array(probs)
    probs = [
        p * (index not in predicted_action_log) for index, p in enumerate(probs)
    ]
    probs = [p / sum(probs) for p in probs]
    assert sum(probs) > 0.999999
    game_board_log.append(np.copy(observation))
    strike_pos = np.random.choice(BOARD_SIZE**2, p=probs)
    observation, reward, done, _ = env.step(strike_pos)
    action_result_log.append(reward)
    predicted_action_log.append(strike_pos)
    if done:
      env.close()
      return np.asarray(game_board_log), np.asarray(
          predicted_action_log), np.asarray(action_result_log)


def compute_rewards(game_result_log, gamma=GAMMA):
  """Compute discounted rewards for TF/JAX training."""
  discounted_rewards = []
  discounted_sum = 0

  for reward in game_result_log[::-1]:
    discounted_sum = reward + gamma * discounted_sum
    discounted_rewards.append(discounted_sum)
  return np.asarray(discounted_rewards[::-1])


def initialize_random_hidden_board(board_size):
  """Initialize the hidden board."""

  hidden_board = np.ones(
      (board_size, board_size)) * HIDDEN_BOARD_CELL_UNOCCUPIED

  # Populate the plane's position
  # First figure out the plane's orientation
  #   0: heading right
  #   1: heading up
  #   2: heading left
  #   3: heading down

  plane_orientation = random.randint(0, 3)

  # Figrue out the location of plane core as the '*' below
  #   | |      |      | |    ---
  #   |-*-    -*-    -*-|     |
  #   | |      |      | |    -*-
  #           ---             |
  if plane_orientation == PLANE_HEADING_RIGHT:
    plane_core_row = random.randint(1, board_size - 2)
    plane_core_column = random.randint(2, board_size - 2)
    # Populate the tail
    hidden_board[plane_core_row][plane_core_column -
                                 2] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row - 1][plane_core_column -
                                     2] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row + 1][plane_core_column -
                                     2] = HIDDEN_BOARD_CELL_OCCUPIED
  elif plane_orientation == PLANE_HEADING_UP:
    plane_core_row = random.randint(1, board_size - 3)
    plane_core_column = random.randint(1, board_size - 3)
    # Populate the tail
    hidden_board[plane_core_row +
                 2][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row + 2][plane_core_column +
                                     1] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row + 2][plane_core_column -
                                     1] = HIDDEN_BOARD_CELL_OCCUPIED
  elif plane_orientation == PLANE_HEADING_LEFT:
    plane_core_row = random.randint(1, board_size - 2)
    plane_core_column = random.randint(1, board_size - 3)
    # Populate the tail
    hidden_board[plane_core_row][plane_core_column +
                                 2] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row - 1][plane_core_column +
                                     2] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row + 1][plane_core_column +
                                     2] = HIDDEN_BOARD_CELL_OCCUPIED
  elif plane_orientation == PLANE_HEADING_DOWN:
    plane_core_row = random.randint(2, board_size - 2)
    plane_core_column = random.randint(1, board_size - 2)
    # Populate the tail
    hidden_board[plane_core_row -
                 2][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row - 2][plane_core_column +
                                     1] = HIDDEN_BOARD_CELL_OCCUPIED
    hidden_board[plane_core_row - 2][plane_core_column -
                                     1] = HIDDEN_BOARD_CELL_OCCUPIED

  # Populate the cross
  hidden_board[plane_core_row][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
  hidden_board[plane_core_row +
               1][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
  hidden_board[plane_core_row -
               1][plane_core_column] = HIDDEN_BOARD_CELL_OCCUPIED
  hidden_board[plane_core_row][plane_core_column +
                               1] = HIDDEN_BOARD_CELL_OCCUPIED
  hidden_board[plane_core_row][plane_core_column -
                               1] = HIDDEN_BOARD_CELL_OCCUPIED

  return hidden_board

# Rewards for each strike
HIT_REWARD = 1
MISS_REWARD = 0
REPEAT_STRIKE_REWARD = -1


class PlaneStrikeEnv(gym.Env):
  """A class that defines the Plane Strike environement."""

  def __init__(self, board_size) -> None:
    super().__init__()
    assert board_size >= 4
    self.board_size = board_size
    self.set_board()

  def step(self, action):
    if self.hit_count == self.plane_size:
      return self.observation, 0, True, {}

    action_x = action // self.board_size
    action_y = action % self.board_size
    # Hit
    if self.hidden_board[action_x][
        action_y] == HIDDEN_BOARD_CELL_OCCUPIED:
      # Non-repeat move
      if self.observation[action_x][action_y] == BOARD_CELL_UNTRIED:
        self.hit_count = self.hit_count + 1
        self.observation[action_x][action_y] = BOARD_CELL_HIT
        # Successful strike
        if self.hit_count == self.plane_size:
          # Game finished
          return self.observation, HIT_REWARD, True, {}
        else:
          return self.observation, HIT_REWARD, False, {}
      # Repeat strike
      else:
        return self.observation, REPEAT_STRIKE_REWARD, False, {}
    # Miss
    else:
      # Unsuccessful strike
      self.observation[action_x][action_y] = BOARD_CELL_MISS
      return self.observation, MISS_REWARD, False, {}

  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    self.set_board()
    return self.observation, {}

#   def render(self, mode='human'):
#     print(self.observation)
#     return

  def close(self):
    return

  def set_board(self):
    self.plane_size = PLANE_SIZE
    self.hit_count = 0
    self.observation = np.zeros((self.board_size, self.board_size))
    self.hidden_board = initialize_random_hidden_board(self.board_size)
    self.action_space = spaces.Discrete(self.board_size * self.board_size)
    self.observation_space = spaces.MultiDiscrete(
        3 * np.ones((self.board_size, self.board_size)), start=-1 * np.ones((self.board_size, self.board_size)))
