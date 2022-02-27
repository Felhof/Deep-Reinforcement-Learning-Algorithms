from agents import trainer, VPN  # type: ignore
import torch.nn
from utilities.config import Config  # type: ignore

NUMBER_OF_ACTIONS: int = 2
ACTION_DIM: int = 1
OBSERVATION_DIM: int = 4

config = Config(
    environment_name="CartPole-v1",
    action_dim=ACTION_DIM,
    observation_dim=OBSERVATION_DIM,
    number_of_actions=NUMBER_OF_ACTIONS,
    hyperparameters={
        "VPN": {
            "episode_length": 200,
            "episodes_per_training_step": 30,
            "value_updates_per_training_step": 10,
            "discount_rate": 0.99,
            "generalized_advantage_estimate_exponential_mean_discount_rate": 0.92,
            "policy_parameters": {
                "sizes": [OBSERVATION_DIM, 128, NUMBER_OF_ACTIONS],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "q_net_parameters": {
                "sizes": [OBSERVATION_DIM, 128, NUMBER_OF_ACTIONS],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "policy_learning_rate": 0.001,
            "q_net_learning_rate": 0.001,
        }
    },
    training_steps_per_epoch=1000,
    epochs=5,
)

if __name__ == "__main__":
    trainer = trainer.Trainer(config)
    trainer.train_agents([VPN.VPN])
