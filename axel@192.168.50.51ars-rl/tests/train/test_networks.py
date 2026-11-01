"""Network builder tests: descriptions, default == brax/T15 exactly, every block builds."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest
from brax.training.agents.ppo import networks as ppo_networks

from tars_rl.train.config import PROFILES, get_profile
from tars_rl.train.eval_networks import candidates
from tars_rl.train.networks import NetSpec, NetworkSpec, make_ppo_networks, tower_sizes
from tars_rl.train.policy import META_FILE, load_policy, network_from_meta, save_policy

OBS, ACT = 11, 4


@pytest.mark.parametrize("text", [
    "mlp-4x128-swish", "mlp-0x1-relu", "mlp-3x256-gelu-layer", "simba-2x128-relu-layer-e4",
    "bro-1x64-mish-rms",
])
def test_short_form_roundtrip(text):
    assert str(NetSpec.parse(text)) == text


def test_network_spec_parse_and_dict():
    n = NetworkSpec.parse("policy=simba-1x32-relu-rms-e2 value=bro-2x64-elu-layer")
    assert n.policy.block == "simba" and n.value.width == 64
    assert NetworkSpec.parse(str(n)) == n
    assert NetworkSpec.from_dict(json.loads(json.dumps(n.to_dict()))) == n
    assert NetworkSpec.parse("bro-1x8-relu-layer").policy == NetworkSpec.parse(
        "bro-1x8-relu-layer").value


@pytest.mark.parametrize("bad", [
    NetSpec(block="transformer"), NetSpec(activation="sigmoid"), NetSpec(norm="batch"),
    NetSpec(block="simba", norm="none"), NetSpec(block="bro", depth=0, norm="layer"),
    NetSpec(width=0),
])
def test_bad_specs_rejected(bad):
    with pytest.raises(ValueError):
        bad.check()


def test_profile_takes_a_network_description():
    cfg = get_profile("smoke", network="policy=simba-1x16-relu-layer-e2 value=mlp-2x16-tanh")
    assert cfg.network.policy.block == "simba" and cfg.to_dict()["network"]["value"]["width"] == 16


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_default_matches_brax_exactly(name):
    """The default description is T15's network: brax's MLP, same param tree and init values."""
    cfg = get_profile(name)
    key = jax.random.PRNGKey(3)
    ours = make_ppo_networks(OBS, ACT, spec=cfg.network)
    pol = tuple(cfg.network.policy.width for _ in range(cfg.network.policy.depth))
    val = tuple(cfg.network.value.width for _ in range(cfg.network.value.depth))
    brax = ppo_networks.make_ppo_networks(OBS, ACT, policy_hidden_layer_sizes=pol,
                                          value_hidden_layer_sizes=val)
    for a, b in ((ours.policy_network, brax.policy_network),
                 (ours.value_network, brax.value_network)):
        pa, pb = a.init(key), b.init(key)
        assert jax.tree_util.tree_structure(pa) == jax.tree_util.tree_structure(pb)
        jax.tree_util.tree_map(np.testing.assert_array_equal, pa, pb)
        obs = jax.random.normal(key, (5, OBS))
        np.testing.assert_allclose(a.apply(None, pa, obs), b.apply(None, pb, obs), atol=1e-6)


@pytest.mark.parametrize("name,spec", list(candidates(get_profile("smoke").network).items()))
def test_every_candidate_builds_and_runs(name, spec):
    nets = make_ppo_networks(OBS, ACT, spec=spec)
    key = jax.random.PRNGKey(0)
    obs = jax.random.normal(key, (7, OBS))
    p, v = nets.policy_network.init(key), nets.value_network.init(key)
    assert nets.policy_network.apply(None, p, obs).shape == (7, 2 * ACT)
    assert nets.value_network.apply(None, v, obs).shape == (7,)
    sizes = tower_sizes(spec, OBS, ACT)
    assert sizes["policy"] == sum(x.size for x in jax.tree_util.tree_leaves(p))


def test_tower_size_of_plain_mlp():
    spec = NetworkSpec(NetSpec(depth=2, width=8), NetSpec(depth=1, width=4))
    # policy: 11*8+8 + 8*8+8 + 8*8+8 (head: 2*ACT outputs); value: 11*4+4 + 4*1+1
    assert tower_sizes(spec, OBS, ACT) == {"policy": 96 + 72 + 72, "value": 48 + 5}


def test_policy_folder_records_description(tmp_path):
    spec = NetworkSpec.parse("policy=bro-1x16-relu-layer value=simba-1x16-relu-layer-e2")
    cfg = get_profile("smoke", network=spec)
    nets = make_ppo_networks(OBS, ACT, spec=spec)
    params = ({}, nets.policy_network.init(jax.random.PRNGKey(0)))
    save_policy(tmp_path, params, {"ppo": cfg.to_dict(), "observation_size": OBS,
                                   "action_size": ACT})
    meta = json.loads((tmp_path / META_FILE).read_text())
    assert meta["network_str"] == str(spec)
    assert load_policy(tmp_path).network == spec


def test_v1_meta_reads_as_plain_mlp():
    meta = {"ppo": {"policy_hidden": [128] * 4, "value_hidden": [256] * 5}}
    assert network_from_meta(meta) == NetworkSpec()
