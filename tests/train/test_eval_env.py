"""The env a policy is scored in (`tars_rl.train.eval.eval_env_config`)."""

from tars_rl.env.config import EnvConfig
from tars_rl.env.terrain import TerrainConfig
from tars_rl.train.eval import eval_env_config


def test_a_terrain_run_is_scored_on_the_floor_asked_for():
    trained = EnvConfig(terrain=TerrainConfig(), friction_cone="elliptic", flags="single").to_dict()
    flat = eval_env_config(trained, None, None)
    assert flat.terrain is None, "a terrain run's flat score must be on the flat floor"
    assert flat.friction_cone == "elliptic" and flat.flags == "single", "the rest is the run's"
    rough = eval_env_config(trained, "chain", TerrainConfig(difficulty_max=0.5))
    assert rough.terrain == TerrainConfig(difficulty_max=0.5) and rough.flags == "chain"
