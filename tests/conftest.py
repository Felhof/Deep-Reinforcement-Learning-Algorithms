from typing import Callable

import pytest
import torch.nn
from utilities.config import Config


@pytest.fixture
def cartpole_config() -> Callable[[str, str], Config]:
    def _create_config(results_filename: str, dtype_name: str = "float32") -> Config:
        return Config(
            environment_name="CartPole-v1",
            action_type="Discrete",
            number_of_actions=2,
            observation_dim=4,
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
            environment_name="MountainCar-v0",
            action_type="Discrete",
            number_of_actions=3,
            observation_dim=2,
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


# TODO: Implement non-categorical actions to make a test with this config work
@pytest.fixture
def mountain_car_continuous_config() -> Callable[[str, str], Config]:
    def _create_config(results_filename, dtype_name: str = "float32") -> Config:
        return Config(
            environment_name="MountainCarContinuous-v0",
            action_type="Continuous",
            number_of_actions=1,
            observation_dim=2,
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
