from typing import Dict, List, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from agents.DQN import DQN
from agents.VPG import VPG


AgentType = Union[Type[DQN], Type[VPG]]

agent_type_to_label: Dict[AgentType, str] = {DQN: "DQN", VPG: "VPG"}
agent_type_to_color: Dict[AgentType, str] = {DQN: "#0000FF", VPG: "#00FF00"}


class Plotter:
    def __init__(self: "Plotter") -> None:
        self.ax = plt.gca()
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
        self.ax.set_ylim([self.current_y_min, self.current_y_max + 20])

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

    def plot_average_agent_overall_results(
        self: "Plotter",
        agent_overall_results: List[List[float]],
        agent_type: AgentType,
        show_std: bool = False,
        show_solution_score: bool = True,
    ) -> None:
        def get_mean_result_at_timestep(
            results: List[List[float]], timestep: int
        ) -> float:
            results_at_timestep = [
                episode_results[timestep] for episode_results in results
            ]
            return np.mean(results_at_timestep)

        timesteps = len(agent_overall_results[0])
        x_vals = list(range(timesteps))

        mean_results = [
            get_mean_result_at_timestep(agent_overall_results, t)
            for t in range(timesteps)
        ]

        color = agent_type_to_color[agent_type]

        self.ax.plot(
            list(range(timesteps)),
            mean_results,
            label=agent_type_to_label[agent_type],
            color=color,
        )

        if show_std:

            def get_std_at_timestep(results: List[List[float]], timestep: int) -> float:
                results_at_timestep = [
                    episode_results[timestep] for episode_results in results
                ]
                return np.std(results_at_timestep)

            result_std = [
                get_std_at_timestep(agent_overall_results, t) for t in range(timesteps)
            ]
            mean_results_plus_std = [
                timestep_mean + timestep_std
                for (timestep_mean, timestep_std) in zip(mean_results, result_std)
            ]
            mean_results_minus_std = [
                timestep_mean - timestep_std
                for (timestep_mean, timestep_std) in zip(mean_results, result_std)
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

        self._adjust_y_limits(agent_overall_results)

        if show_solution_score:
            self._draw_horizontal_line_with_label(
                y_value=200,
                x_min=0,
                x_max=timesteps * 1.02,
                label="Target \n score",
            )

    def create_plot(
        self: "Plotter",
        title: str = "learning curve",
        filename: str = "results.png",
        show: bool = True,
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
        self.ax.set_ylabel("Rolling Episode Scores")
        self.ax.set_xlabel("Episode Number")
        for spine in ["right", "top"]:
            self.ax.spines[spine].set_visible(False)

        plt.savefig(f"{filename}")

        if show:
            plt.show()
