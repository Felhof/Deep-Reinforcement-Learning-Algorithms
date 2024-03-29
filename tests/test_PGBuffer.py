import numpy as np
from utilities.buffer.PGBuffer import PGBuffer


def test_stores_and_returns_transition_batches_correctly(cartpole_config) -> None:
    episode_length = 5
    n_episodes = 2
    config = cartpole_config("")
    buffer = PGBuffer(config, n_episodes, observation_dim=(4,))

    obs_1 = np.random.rand(episode_length, 4).astype("float32")
    actions_1 = np.random.randint(0, 2, episode_length, dtype="int")
    values_1 = np.array([0.7, 0.3, 0.1, 0.5, 0.6], dtype="float32")
    rewards_1 = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype="float32")

    obs_2 = np.random.rand(episode_length, 4).astype("float32")
    actions_2 = np.random.randint(0, 2, episode_length, dtype="int")
    values_2 = np.array([0.2, 0.8, 0.9, 0.3, 0.1], dtype="float32")
    rewards_2 = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype="float32")

    expected_obs = np.stack([obs_1, obs_2])
    expected_actions = np.stack([actions_1, actions_2])
    expected_rewards = np.stack([rewards_1, rewards_2])

    expected_reward_to_go_1 = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype="float32")
    expected_reward_to_go_2 = np.array([2.0, 1.0, 0.0, 0.0, 0.0], dtype="float32")
    expected_rewards_to_go = np.stack(
        [expected_reward_to_go_1, expected_reward_to_go_2]
    )

    expected_advantage_1 = np.array(
        [2.8956451, 2.52376487, 1.89368127, 0.54752008, -0.60000002], dtype="float32"
    )
    expected_advantage_2 = np.array(
        [2.54605883, 1.04749542, -0.04776533, 0.60961198, 0.88999999], dtype="float32"
    )
    expected_advantages = np.vstack([expected_advantage_1, expected_advantage_2])

    buffer.add_transition_data(obs_1, actions_1, values_1, rewards_1)
    buffer.add_transition_data(obs_2, actions_2, values_2, rewards_2, last_value=1.0)

    (
        actual_obs,
        actual_actions,
        actual_rewards,
        actual_advantages,
        actual_rewards_to_go,
    ) = buffer.get_data()

    assert np.array_equal(actual_obs, expected_obs)
    assert np.array_equal(actual_actions, expected_actions)
    assert np.array_equal(actual_rewards, expected_rewards)
    assert np.allclose(actual_advantages, expected_advantages)
    assert np.allclose(actual_rewards_to_go, expected_rewards_to_go)
