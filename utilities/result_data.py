from dataclasses import dataclass
from typing import Dict, Type, Union

from agents.DQN import DQN
from agents.PPO import PPO
from agents.TRPG import TRPG
from agents.VPG import VPG
from utilities.types import EpochRewards

AgentType = Union[Type[DQN], Type[VPG], Type[TRPG], Type[PPO]]
agent_type_to_label: Dict[AgentType, str] = {
    DQN: "DQN",
    VPG: "VPG",
    TRPG: "TRPG",
    PPO: "PPO",
}
agent_label_to_type: Dict[str, AgentType] = {
    "DQN": DQN,
    "VPG": VPG,
    "TRPG": TRPG,
    "PPO": PPO,
}


@dataclass
class ResultData:
    average_epoch_rewards: EpochRewards
    agent_type: AgentType
