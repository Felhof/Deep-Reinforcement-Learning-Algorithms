from utilities.result_reading import get_result_data_for_agent
from utilities.plotter import Plotter

if __name__ == "__main__":
    vpg_results = get_result_data_for_agent(agent_label="VPG", filenames=["VPG_cartpole_rewards"])
    ppo_results = get_result_data_for_agent(agent_label="PPO", filenames=["PPO_cartpole_rewards2"])
    trpg_results = get_result_data_for_agent(agent_label="TRPG", filenames=['TRPG_cartpole_rewards_01',
                                                                            'TRPG_cartpole_rewards_02',
                                                                            'TRPG_cartpole_rewards_03',
                                                                            'TRPG_cartpole_rewards_04',
                                                                            'TRPG_cartpole_rewards_05'])
    sac_results = get_result_data_for_agent(agent_label="SAC", filenames=["SAC_cartpole_rewards"])
    dqn_results = get_result_data_for_agent(agent_label="DQN", filenames=["DQN_cartpole_rewards"])
    Plotter().plot([vpg_results, ppo_results, trpg_results, sac_results, dqn_results], n_episodes=400, title="Results for Cartpole (Discrete)",
           filename="cartpole_learning_curves", show_plot=True, show_std=True, show_target_score=False)

