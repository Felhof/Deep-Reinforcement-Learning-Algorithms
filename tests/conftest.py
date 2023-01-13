from typing import Callable

import gymnasium as gym
import pytest
import torch.nn
from utilities.config import Config
from utilities.environments import BaseEnvironmentWrapper


@pytest.fixture
def cartpole_config() -> Callable[[str, str], Config]:
    def _create_config(results_filename: str, dtype_name: str = "float32") -> Config:
        return Config(
            hyperparameters={
                "policy_gradient": {
                    "episodes_per_training_step": 30,
                    "value_updates_per_training_step": 20,
                    "discount_rate": 0.99,
                    "gae_exp_mean_discount_rate": 0.92,
                    "policy_net_parameters": {
                        "hidden_layer_sizes": [128],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                        "learning_rate": 0.001,
                    },
                    "value_net_parameters": {
                        "hidden_layer_sizes": [128],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                        "learning_rate": 0.001,
                    },
                    "dtype_name": dtype_name,
                },
                "TRPG": {
                    "kl_divergence_limit": 0.01,
                    "backtracking_coefficient": 0.5,
                    "backtracking_iterations": 10,
                    "damping_coefficient": 1e-8,
                    "conjugate_gradient_iterations": 10,
                },
                "PPO": {"clip_range": 0.1},
                "DQN": {
                    "discount_rate": 0.99,
                    "q_net_parameters": {
                        "hidden_layer_sizes": [64],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                    },
                    "q_net_learning_rate": 0.001,
                    "minibatch_size": 256,
                    "buffer_size": 40000,
                    "initial_exploration_rate": 1,
                    "random_episodes": 3,
                    "gradient_clipping_norm": 0.7,
                },
            },
            episode_length=5,
            training_steps_per_epoch=5,
            epochs=3,
            target_score=200,
            results_filename=results_filename,
        )

    return _create_config


@pytest.fixture
def mountain_car_config() -> Callable[[str, str], Config]:
    def _create_config(results_filename, dtype_name: str = "float32") -> Config:
        return Config(
            hyperparameters={
                "policy_gradient": {
                    "episodes_per_training_step": 30,
                    "value_updates_per_training_step": 20,
                    "discount_rate": 0.99,
                    "gae_exp_mean_discount_rate": 0.92,
                    "policy_net_parameters": {
                        "hidden_layer_sizes": [128],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                        "learning_rate": 0.001,
                    },
                    "value_net_parameters": {
                        "hidden_layer_sizes": [128],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                        "learning_rate": 0.001,
                    },
                    "dtype_name": dtype_name,
                },
                "TRPG": {
                    "kl_divergence_limit": 0.01,
                    "backtracking_coefficient": 0.5,
                    "backtracking_iterations": 10,
                    "damping_coefficient": 1e-8,
                    "conjugate_gradient_iterations": 10,
                },
                "PPO": {"clip_range": 0.1},
                "DQN": {
                    "discount_rate": 0.99,
                    "q_net_parameters": {
                        "hidden_layer_sizes": [64],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                    },
                    "q_net_learning_rate": 0.001,
                    "minibatch_size": 256,
                    "buffer_size": 40000,
                    "initial_exploration_rate": 1,
                    "random_episodes": 3,
                    "gradient_clipping_norm": 0.7,
                },
            },
            episode_length=5,
            training_steps_per_epoch=5,
            epochs=3,
            target_score=200,
            results_filename=results_filename,
        )

    return _create_config


@pytest.fixture
def mountain_car_continuous_config() -> Callable[[str, str], Config]:
    def _create_config(results_filename, dtype_name: str = "float32") -> Config:
        return Config(
            hyperparameters={
                "policy_gradient": {
                    "episodes_per_training_step": 30,
                    "value_updates_per_training_step": 20,
                    "discount_rate": 0.99,
                    "gae_exp_mean_discount_rate": 0.92,
                    "policy_net_parameters": {
                        "hidden_layer_sizes": [128],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                        "learning_rate": 0.001,
                    },
                    "value_net_parameters": {
                        "hidden_layer_sizes": [128],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                        "learning_rate": 0.001,
                    },
                    "use_double_precision": True,
                    "dtype_name": dtype_name,
                },
                "TRPG": {
                    "kl_divergence_limit": 0.01,
                    "backtracking_coefficient": 0.5,
                    "backtracking_iterations": 10,
                    "damping_coefficient": 1e-8,
                    "conjugate_gradient_iterations": 10,
                },
                "PPO": {"clip_range": 0.1},
                "DQN": {
                    "discount_rate": 0.99,
                    "q_net_parameters": {
                        "hidden_layer_sizes": [64],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                    },
                    "q_net_learning_rate": 0.001,
                    "minibatch_size": 256,
                    "buffer_size": 40000,
                    "initial_exploration_rate": 1,
                    "random_episodes": 3,
                    "gradient_clipping_norm": 0.7,
                },
            },
            episode_length=5,
            training_steps_per_epoch=5,
            epochs=3,
            target_score=200,
            results_filename=results_filename,
        )

    return _create_config


@pytest.fixture
def adventure_config() -> Callable[[str, str], Config]:
    def _create_config(results_filename, dtype_name: str = "float32") -> Config:
        return Config(
            hyperparameters={
                "policy_gradient": {
                    "episodes_per_training_step": 30,
                    "value_updates_per_training_step": 20,
                    "discount_rate": 0.99,
                    "gae_exp_mean_discount_rate": 0.92,
                    "policy_net_parameters": {
                        "hidden_layer_sizes": [128],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                        "learning_rate": 0.001,
                    },
                    "value_net_parameters": {
                        "hidden_layer_sizes": [128],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                        "learning_rate": 0.001,
                    },
                    "use_double_precision": True,
                    "dtype_name": dtype_name,
                },
                "TRPG": {
                    "kl_divergence_limit": 0.01,
                    "backtracking_coefficient": 0.5,
                    "backtracking_iterations": 10,
                    "damping_coefficient": 1e-8,
                    "conjugate_gradient_iterations": 10,
                },
                "PPO": {"clip_range": 0.1},
                "DQN": {
                    "discount_rate": 0.99,
                    "q_net_parameters": {
                        "hidden_layer_sizes": [64],
                        "activations": [
                            torch.nn.ReLU(),
                            torch.nn.Tanh(),
                        ],
                    },
                    "q_net_learning_rate": 0.001,
                    "minibatch_size": 256,
                    "buffer_size": 40000,
                    "initial_exploration_rate": 1,
                    "random_episodes": 3,
                    "gradient_clipping_norm": 0.7,
                },
            },
            episode_length=5,
            training_steps_per_epoch=5,
            epochs=3,
            target_score=200,
            results_filename=results_filename,
        )

    return _create_config


@pytest.fixture
def cartpole_environment() -> BaseEnvironmentWrapper:
    return BaseEnvironmentWrapper(
        gym.make("CartPole-v1")
    )


@pytest.fixture
def mountain_car_environment() -> BaseEnvironmentWrapper:
    return BaseEnvironmentWrapper(
        gym.make("MountainCar-v0")
    )


@pytest.fixture
def continuous_mountain_car_environment() -> BaseEnvironmentWrapper:
    return BaseEnvironmentWrapper(
        gym.make("MountainCarContinuous-v0")
    )
