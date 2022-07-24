from agents import trainer, VPG
import torch.nn
from utilities.config import Config

NUMBER_OF_ACTIONS: int = 2
ACTION_DIM: int = 1
OBSERVATION_DIM: int = 4

config = Config(
    environment_name="CartPole-v1",
    action_dim=ACTION_DIM,
    observation_dim=OBSERVATION_DIM,
    number_of_actions=NUMBER_OF_ACTIONS,
    hyperparameters={
        "policy_gradient": {
            "episodes_per_training_step": 30,
            "value_updates_per_training_step": 20,
            "discount_rate": 0.99,
            "generalized_advantage_estimate_exponential_mean_discount_rate": 0.92,
            "policy_parameters": {
                "sizes": [OBSERVATION_DIM, 128, NUMBER_OF_ACTIONS],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "value_net_parameters": {
                "sizes": [OBSERVATION_DIM, 128, NUMBER_OF_ACTIONS],
                "activations": [
                    torch.nn.ReLU(),
                    torch.nn.Tanh(),
                ],
            },
            "policy_learning_rate": 0.001,
            "value_net_learning_rate": 0.001,
        }
    },
    episode_length=250,
    training_steps_per_epoch=1000,
    epochs=6,
)

if __name__ == "__main__":
    vpn_trainer = trainer.Trainer(config)
    vpn_trainer.train_agents([VPG.VPG])
