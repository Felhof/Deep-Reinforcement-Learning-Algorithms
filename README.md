# DEEP REINFORCEMENT LEARNING ALGORITHMS
This repository contains PyTorch implementations of various Deep Reinforcement Learning algorithms and a comparison of
their results.


## Algorithms
The following algorithms have been implemented so far:

1. REINFORCE/Vanilla Policy Gradient (VPG): [OpenAI's Spinning Up](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
2. Deep Q Learning (DQN): [Mnih et al. 2013](https://arxiv.org/pdf/1312.5602.pdf)
3. Trust Region Policy Gradient (TRPG): [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/trpo.html) / [(Schulman et al 2015b)](https://arxiv.org/abs/1502.05477) *
4. Proximal Policy Optimization (PPO): [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) / [(Schulman et al 2017)](https://arxiv.org/abs/1707.06347)
5. Soft Actor Critic (SAC) for discrete environments: [Spinning Up (continuous version)](https://spinningup.openai.com/en/latest/algorithms/sac.html) / [(Christodoulou 2019)](https://arxiv.org/abs/1910.07207)

The policy gradient algorithms 1, 3, and 4 are using Generalized Advantage Estimation [(Schulman et al 2015a)](https://arxiv.org/abs/1506.02438)

\* the implementation of TRPG occasionally fails during learning due to numerical issues. Results are from successful runs only

## Results

### Space Invaders
This is the result of the DQN agent on the [gymnasium Atari Space Invader](https://gymnasium.farama.org/environments/atari/space_invaders/) environment. The training setup is similar to the original 
[paper by (Mnih et al 2013b)](https://arxiv.org/abs/1312.5602): Agent observations are the last four frames which are scaled to 84x84 grayscale. The exact hyperparameters can be found in 
train_DQN_for_Space_Invaders.py and are also similar to (Mnih et al 2013b). However, I only trained for 5 million steps as opposed to 50 million in the original paper.

![DQN Space Invaders](results/dqn_space_invaders_agent.gif)

![DQN Space Invaders Learning Curve](results/dqn_space_invaders_learning_curve.png)

### Cartpole

The algorithms were trained on OpenAI Gym's [implementation](https://www.gymlibrary.ml/environments/classic_control/cart_pole/)
of the Cart Pole Environment. Each agent was trained for 400 training steps with episodes automatically terminating after
200 timesteps. For the exact hyperparameters see the training scripts (train_X_for_cartpole.py). The y value of the learning curves
represents the mean score of running the algorithm 5 times and the shaded area around the learning curve corresponds to
the standard deviation. The following curves were smoothed using a moving average with a window size of 4.

![Cartpole Results](results/cartpole_learning_curves.png)

Note that just looking at the learning curves is not sufficient to compare two algorithms. Firstly, the same amount of
training steps does not necessarily require the same amount of computing power and training time. For example, DQN can do
a training step after every timestep after an initial period of exploration. On the other hand, VPN must complete multiple
full episodes for every training step. Furthermore, no hyperparameter tuning was done before running the algorithms. Doing
so might significantly improve performance. Hence, the learning curves only serve to demonstrate the correct implementation
of the algorithms and their learning behaviour.

## Acknowledgements
The implementations of the algorithms in this repository are my own, but it was immensely useful to look at the 
[Spinning Up repository](https://github.com/openai/spinningup) and [Deep Reinforcement Learning Algorithms in PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
 when I was stuck or looking for things to improve.

This [Medium article](https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737) by Rohan Tangri helped me understand
Generalized Advantage Estimation.
