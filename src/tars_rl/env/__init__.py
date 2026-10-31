"""Flag arena: TARS on a flat floor walking to a random checkered flag, in MJX."""

from tars_rl.env.config import EnvConfig, RewardConfig
from tars_rl.env.flag_env import FlagEnv, State
from tars_rl.env.rollout import FlagRollout
from tars_rl.env.sensors import FLAG_OBS_DIM, ground_truth_flag

__all__ = [
    "FLAG_OBS_DIM",
    "EnvConfig",
    "FlagEnv",
    "FlagRollout",
    "RewardConfig",
    "State",
    "ground_truth_flag",
]
