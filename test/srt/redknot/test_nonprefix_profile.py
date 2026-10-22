"""The new benchmark must not silently drop a legacy full-prefix policy."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from nonprefix_profile import load_effective_profile


_HERE = Path(__file__).resolve().parent
_REAL = _HERE / "head_class/qwen3-32B_optimal_g15_lf_ret.json"


@pytest.fixture(scope="module")
def head_config_class():
    path = _HERE.parents[2] / "python/sglang/srt/layers/attention/redknot/head_config.py"
    name = "_nonprefix_profile_test_head_config"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module.HeadClassConfig
    finally:
        sys.modules.pop(name, None)


def _data(**updates):
    value = dict(
        num_layers=3, num_kv_heads=2,
        kv_head_classification=[["local_full", "global"], ["retrieval", "local_full"], ["local_full", "global"]],
        kv_head_max_distance=[[7, -1], [-1, 7], [7, -1]],
        dense_prefix_layers=1,
    )
    value.update(updates)
    return value


def _write(tmp_path, data):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("override,front,local", [(None, 48, 51), (0, 0, 435), (32, 32, 179)])
def test_real_legacy_profile_reports_effective_counts_not_inconsistent_summary(head_config_class, override, front, local):
    original = _REAL.read_bytes()
    profile, report = load_effective_profile(_REAL, head_config_class, override)
    assert profile.dense_prefix_layers == front
    assert report["effective_global_prefix_layers"] == front
    assert report["effective_local_head_count"] == local
    assert report["effective_local_head_ratio"] == pytest.approx(local / 512)
    assert report["raw_local_head_count"] == 435
    assert report["raw_local_head_ratio"] == pytest.approx(435 / 512)
    assert sum(sum(row) for row in report["raw_local_head_mask"]) == 435
    assert report["retrieval_heads_promoted_global"] == 29
    assert report["effective_counts"]["retrieval"] == 0
    assert report["effective_counts"]["dense"] == front * 8
    assert report["effective_full_recompute_heads"] == 512 - local
    assert report["declared_front_full_layers"] == 48
    assert report["declared_summary"]["local"] == 128
    assert report["declared_summary_matches_effective"] is False
    assert {entry["field"] for entry in report["summary_mismatches"]} >= {"local", "global", "local_pct", "global_pct"}
    assert report["explicit_global_prefix_override"] is (override is not None)
    assert report["prefix_policy_source"] == ("front_full_layers" if override is None else "explicit_override")
    assert _REAL.read_bytes() == original
    json.dumps(report, allow_nan=False)


def test_default_falls_back_to_existing_dense_prefix_layers(tmp_path, head_config_class):
    profile, report = load_effective_profile(_write(tmp_path, _data()), head_config_class)
    assert profile.dense_prefix_layers == 1
    assert report["prefix_policy_source"] == "dense_prefix_layers"
    assert report["effective_counts"] == {"local": 2, "global": 2, "dense": 2, "retrieval": 0, "total": 6}
    assert report["declared_summary_matches_effective"] is None


def test_missing_both_prefix_fields_preserves_zero_default(tmp_path, head_config_class):
    data = _data()
    del data["dense_prefix_layers"]
    profile, report = load_effective_profile(_write(tmp_path, data), head_config_class)
    assert profile.dense_prefix_layers == 0
    assert report["effective_local_head_count"] == 3


def test_front_field_precedes_dense_field_and_reports_conflict(tmp_path, head_config_class):
    profile, report = load_effective_profile(_write(tmp_path, _data(front_full_layers=2)), head_config_class)
    assert profile.dense_prefix_layers == 2
    assert report["effective_local_head_count"] == 1
    assert report["metadata_warnings"]


def test_explicit_zero_override_is_not_treated_as_missing(tmp_path, head_config_class):
    profile, report = load_effective_profile(_write(tmp_path, _data(front_full_layers=2)), head_config_class, 0)
    assert profile.dense_prefix_layers == 0
    assert report["effective_local_head_count"] == 3
    assert report["explicit_global_prefix_override"]


def test_all_global_override_and_summary_are_independent_of_raw_matrix(tmp_path, head_config_class):
    data = _data(summary={"local": 0, "global": 6, "total": 6, "local_pct": "0%", "global_pct": "100%"})
    profile, report = load_effective_profile(_write(tmp_path, data), head_config_class, 3)
    assert report["raw_local_head_count"] == 3
    assert report["effective_local_head_count"] == 0
    assert report["declared_summary_matches_effective"] is True
    assert all(profile.get_strategy(layer, head).head_type == "dense" for layer in range(3) for head in range(2))


@pytest.mark.parametrize("override", [-1, 4, True, 1.0, "2"])
def test_invalid_explicit_overrides_fail_closed(tmp_path, head_config_class, override):
    with pytest.raises(ValueError, match="global_prefix_layers"):
        load_effective_profile(_write(tmp_path, _data()), head_config_class, override)


@pytest.mark.parametrize("field", ["front_full_layers", "dense_prefix_layers"])
@pytest.mark.parametrize("value", [-1, 4, True, 1.0, "2", None])
def test_invalid_declared_prefix_fields_fail_closed(tmp_path, head_config_class, field, value):
    with pytest.raises(ValueError, match=field):
        load_effective_profile(_write(tmp_path, _data(**{field: value})), head_config_class)


@pytest.mark.parametrize("data", [[], _data(num_layers=True), _data(num_kv_heads=0),
                                  _data(kv_head_classification=[["local"]])])
def test_invalid_topology_fails_before_model_load(tmp_path, head_config_class, data):
    with pytest.raises(ValueError):
        load_effective_profile(_write(tmp_path, data), head_config_class)


def test_repeated_loads_are_fresh_and_do_not_mutate_profile_file(head_config_class):
    first, _ = load_effective_profile(_REAL, head_config_class, 0)
    first.head_class[0][0] = "global"
    first.dense_prefix_layers = 64
    second, report = load_effective_profile(_REAL, head_config_class)
    assert second.head_class[0][0] == "local"
    assert second.dense_prefix_layers == 48
    assert report["effective_local_head_count"] == 51


def test_existing_tensor_cache_is_invalidated_after_prefix_override(tmp_path, head_config_class):
    import torch

    class CachedLoader:
        @staticmethod
        def from_json(path):
            profile = head_config_class.from_json(path)
            profile.as_tensors(torch.device("cpu"))
            return profile

    profile, _ = load_effective_profile(_write(tmp_path, _data(front_full_layers=2)), CachedLoader)
    tensor = profile.as_tensors(torch.device("cpu"))["head_type"]
    assert tensor[:2].eq(profile.TYPE_DENSE).all()


def test_concurrent_profile_edit_is_not_silently_accepted(tmp_path, head_config_class):
    path = _write(tmp_path, _data())

    class ChangingLoader:
        @staticmethod
        def from_json(path):
            profile = head_config_class.from_json(path)
            Path(path).write_text(json.dumps(_data(front_full_layers=2)))
            return profile

    with pytest.raises(RuntimeError, match="changed while being loaded"):
        load_effective_profile(path, ChangingLoader)
