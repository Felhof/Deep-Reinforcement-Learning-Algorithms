from collections import deque
from typing import Dict, List

from agents.DQN import DQN
from agents.PPO import PPO
from agents.TRPG import TRPG
from agents.VPG import VPG
import matplotlib.pyplot as plt
import numpy as np
from utilities.result_data import (
    agent_type_to_label,
    AgentType,
    EpochRewards,
    ResultData,
)


agent_type_to_color: Dict[AgentType, str] = {
    DQN: "#0000FF",
    VPG: "#00FF00",
    TRPG: "#FF0000",
    PPO: "#fcad03",
}


def smooth_values(rewards: List[float], window_length: int = 4) -> List[float]:
    window: deque[float] = deque()
    smoothed_rewards: List[float] = []
    for reward in rewards:
        if len(window) == window_length:
            window.popleft()
        window.append(reward)
        smoothed_rewards.append(float(np.mean(window)))
    return smoothed_rewards


class Plotter:
    def __init__(self: "Plotter", target_score: int = 200) -> None:
        self.ax = plt.gca()
        self.target_score = target_score
        self.current_y_max = float("-inf")
        self.current_y_min = float("inf")

    def _adjust_y_limits(
        self: "Plotter", agent_overall_results: List[List[float]]
    ) -> None:
        for episode_results in agent_overall_results:
            for timestep_result in episode_results:
                if timestep_result > self.current_y_max:
                    self.current_y_max = timestep_result
                elif timestep_result < self.current_y_min:
                    self.current_y_min = timestep_result

        y_max = max(float(self.target_score), self.current_y_max)
        self.ax.set_ylim([self.current_y_min, y_max * 1.1])

    def _draw_horizontal_line_with_label(
        self: "Plotter", y_value: float, x_min: float, x_max: float, label: str
    ) -> None:
        self.ax.hlines(
            y=y_value,
            xmin=x_min,
            xmax=x_max,
            linewidth=2,
            color="r",
            linestyles="dotted",
            alpha=0.5,
        )
        self.ax.text(x_max, y_value * 0.965, label)

    def plot(
        self: "Plotter",
        results: List[ResultData],
        show_std: bool = False,
        n_episodes: int = 200,
        show_target_score: bool = True,
        title: str = "learning curve",
        filename: str = "results.png",
        show_plot: bool = False,
        smooth_curves: bool = True,
    ) -> None:
        for agent_results in results:
            label = agent_type_to_label[agent_results.agent_type]
            color = agent_type_to_color[agent_results.agent_type]
            epoch_rewards_to_plot = (
                [
                    smooth_values(rewards)
                    for rewards in agent_results.average_epoch_rewards
                ]
                if smooth_curves
                else agent_results.average_epoch_rewards
            )
            self._plot_agents_average_rewards_over_all_epochs(
                epoch_rewards_to_plot,
                n_episodes=n_episodes,
                color=color,
                label=label,
                show_std=show_std,
            )

        if show_target_score:
            self._show_target_score(timesteps=n_episodes)

        self._save_plot(title=title, filename=filename)

        if show_plot:
            plt.show()

    def _plot_agents_average_rewards_over_all_epochs(
        self: "Plotter",
        epoch_rewards: EpochRewards,
        n_episodes: int = 200,
        color: str = "#00FF00",
        label: str = "VPG",
        show_std: bool = False,
    ) -> None:
        def get_mean_result_for_episode(
            results: List[List[float]], timestep: int
        ) -> float:
            results_for_episode = [
                episode_results[timestep] for episode_results in results
            ]
            return float(np.mean(results_for_episode))

        mean_results = [
            get_mean_result_for_episode(epoch_rewards, t) for t in range(n_episodes)
        ]

        self.ax.plot(
            list(range(n_episodes)),
            mean_results,
            label=label,
            color=color,
        )

        x_vals = list(range(n_episodes))

        if show_std:

            def get_std_at_timestep(results: List[List[float]], timestep: int) -> float:
                results_at_timestep = [
                    episode_results[timestep] for episode_results in results
                ]
                return float(np.std(results_at_timestep))

            rewards_std = [
                get_std_at_timestep(epoch_rewards, t) for t in range(n_episodes)
            ]
            mean_results_plus_std = [
                timestep_mean + timestep_std
                for (timestep_mean, timestep_std) in zip(mean_results, rewards_std)
            ]
            mean_results_minus_std = [
                timestep_mean - timestep_std
                for (timestep_mean, timestep_std) in zip(mean_results, rewards_std)
            ]

            self.ax.plot(x_vals, mean_results_plus_std, color=color, alpha=0.1)
            self.ax.plot(x_vals, mean_results_minus_std, color=color, alpha=0.1)
            self.ax.fill_between(
                x_vals,
                y1=mean_results_minus_std,
                y2=mean_results_plus_std,
                color=color,
                alpha=0.1,
            )

        self.ax.set_xlim([0, x_vals[-1]])

        self._adjust_y_limits(epoch_rewards)

    def _show_target_score(self: "Plotter", timesteps: int = 200) -> None:
        self._draw_horizontal_line_with_label(
            y_value=200,
            x_min=0,
            x_max=timesteps * 1.02,
            label="Target \n score",
        )

    def _save_plot(
        self: "Plotter",
        title: str = "learning curve",
        filename: str = "results.png",
    ) -> None:
        # Shrink current axis's height by 10% on the bottom
        box = self.ax.get_position()
        self.ax.set_position(
            [box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95]
        )

        # Put a legend below current axis
        self.ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=3,
        )

        self.ax.set_title(title, fontsize=15, fontweight="bold")
        self.ax.set_ylabel("Episode Scores")
        self.ax.set_xlabel("Episode Number")
        for spine in ["right", "top"]:
            self.ax.spines[spine].set_visible(False)

        plt.savefig(f"results/{filename}", bbox_inches="tight")
