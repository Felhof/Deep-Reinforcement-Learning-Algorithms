from utilities.result_reading import get_result_data_for_agent
from utilities.plotter import Plotter

if __name__ == "__main__":
    p = Plotter()
    vpg_results = get_result_data_for_agent(agent_label="VPG", filenames=["VPG_cartpole_rewards"])
    ppo_results = get_result_data_for_agent(agent_label="PPO", filenames=["PPO_cartpole_rewards"])
    trpg_results = get_result_data_for_agent(agent_label="TRPG", filenames=['TRPG_02', 'TRPG_03', 'TRPG_04', 'TRPG_08', 'TRPG_12'])
    dqn_results = get_result_data_for_agent(agent_label="DQN", filenames=["DQN_Cartpole"])
    p.plot([vpg_results, ppo_results, dqn_results, trpg_results], n_episodes=400, title="Cartpole learning curves",
           filename="learning_curves", show_plot=True, show_std=True, show_target_score=False)

