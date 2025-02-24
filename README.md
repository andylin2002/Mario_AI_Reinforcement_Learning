## Overview

This project implements a Deep Q-Network (DQN) for training an agent to play Super Mario Bros game. The model leverages convolutional neural networks (CNNs) to process game frames and reinforcement learning techniques to optimize Mario's actions based on rewards.

## Dependence

python == 3.9.20
gym == 0.21.0
gym-super-mario-bros == 7.3.0 
numpy == 1.25.2
pillow == 11.0.0

## Custom Rewards

The training process is guided by a custom reward function to encourage better gameplay strategies:

- Encouraging Forward Movement: Rewards are given for moving right.

- Jumping Mechanics: Rewards are assigned based on jump height.

- Coin Collection: Bonus rewards for collecting coins.

- Defeating Enemies: Extra points for eliminating enemies.

- Reaching the Goal: Large rewards for completing levels.

- Avoiding Repeated Jumps: Penalty for unnecessary jumps.

- Falling into Holes: Heavy penalty for losing a life.
