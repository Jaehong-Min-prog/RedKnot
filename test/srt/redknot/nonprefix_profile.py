"""Explicit legacy-profile interpretation for the new non-prefix benchmark.

This helper does not change the shared HeadClassConfig loader or profile files.
Profile summaries are descriptive metadata, never execution policy. In
particular, a legacy ``front_full_layers`` must not silently disappear when
the underlying loader only understands ``dense_prefix_layers``.
"""

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path


def _layer_count(value, name, total):
    if type(value) is not int or not 0 <= value <= total:
        raise ValueError(f"{name} must be an integer in [0, {total}]")
    return value


def load_effective_profile(path, HeadClassConfig, global_prefix_layers=None):
    """Return (fresh executable profile, JSON-serializable diagnostics).

    An explicit integer override, including zero, wins. Otherwise honor
    ``front_full_layers`` when present, then the loader's existing
    ``dense_prefix_layers``. Retrieval promotion is explicit and counted.
    """
    path = Path(path)
    original = path.read_bytes()
    data = json.loads(original)
    if not isinstance(data, dict):
        raise ValueError("head profile must be a JSON object")
    layers, heads = data.get("num_layers"), data.get("num_kv_heads")
    if type(layers) is not int or layers <= 0 or type(heads) is not int or heads <= 0:
        raise ValueError("profile topology must contain positive integer layer/head counts")
    matrix = data.get("kv_head_classification")
    if (not isinstance(matrix, list) or len(matrix) != layers
            or any(not isinstance(row, list) or len(row) != heads for row in matrix)):
        raise ValueError("head classification matrix does not match profile topology")
    for field in ("front_full_layers", "dense_prefix_layers"):
        if field in data:
            _layer_count(data[field], field, layers)
    if global_prefix_layers is not None:
        chosen = _layer_count(global_prefix_layers, "global_prefix_layers", layers)
        origin = "explicit_override"
    elif "front_full_layers" in data:
        chosen, origin = data["front_full_layers"], "front_full_layers"
    else:
        chosen, origin = data.get("dense_prefix_layers", 0), "dense_prefix_layers"

    profile = HeadClassConfig.from_json(str(path))
    # The legacy loader rereads the path. Never combine diagnostics from one
    # version with an executable profile loaded from a concurrently edited one.
    if path.read_bytes() != original:
        raise RuntimeError("head profile changed while being loaded")
    if (profile.num_layers, profile.num_kv_heads) != (layers, heads):
        raise ValueError("loaded profile topology differs from declared topology")
    normalized_counts = Counter(head for row in profile.head_class for head in row)
    local_mask = [[head == "local" for head in row] for row in profile.head_class]
    converted = profile.merge_retrieval_to_global()
    profile.dense_prefix_layers = chosen
    # as_tensors may cache effective strategies; do not retain a loader cache
    # that was built before applying this benchmark's explicit prefix policy.
    if hasattr(profile, "_cached_tensors"):
        profile._cached_tensors.clear()
    counts = Counter(profile.get_strategy(layer, head).head_type
                     for layer in range(layers) for head in range(heads))
    total = layers * heads
    effective = {name: counts.get(name, 0) for name in ("local", "global", "dense", "retrieval")}
    effective["total"] = total
    effective_full = effective["global"] + effective["dense"]
    declared_summary = copy.deepcopy(data.get("summary"))
    mismatches, warnings = [], []
    if ("front_full_layers" in data and "dense_prefix_layers" in data
            and data["front_full_layers"] != data["dense_prefix_layers"]):
        warnings.append("front_full_layers and dense_prefix_layers disagree; precedence is explicit_override > front_full_layers > dense_prefix_layers")
    if declared_summary is not None and not isinstance(declared_summary, dict):
        warnings.append("summary is not an object and is not used as execution policy")
    if isinstance(declared_summary, dict):
        # Old two-class summaries call all full-recompute heads 'global'. A
        # summary that explicitly distinguishes dense gets exact-type counts.
        summary_actual = dict(effective)
        if "dense" not in declared_summary:
            summary_actual["global"] = effective_full
        for field, actual in summary_actual.items():
            if field in declared_summary and declared_summary[field] != actual:
                mismatches.append(dict(field=field, declared=declared_summary[field], actual=actual))
        for field in ("local_pct", "global_pct"):
            if field not in declared_summary:
                continue
            declared = declared_summary[field]
            try:
                value = float(declared.removesuffix("%") if isinstance(declared, str) else declared)
            except (TypeError, ValueError):
                value = float("nan")
            actual = 100 * summary_actual[field.removesuffix("_pct")] / total
            if not math.isfinite(value) or not math.isclose(value, actual, abs_tol=0.005, rel_tol=0):
                mismatches.append(dict(field=field, declared=declared, actual=actual))
    diagnostics = dict(
        schema="redknot-nonprefix-effective-profile-v1", profile_path=str(path),
        profile_file_sha256=hashlib.sha256(original).hexdigest(),
        declared_front_full_layers=data.get("front_full_layers"),
        declared_dense_prefix_layers=data.get("dense_prefix_layers"),
        declared_summary=declared_summary,
        explicit_global_prefix_override=global_prefix_layers is not None,
        requested_global_prefix_layers=global_prefix_layers,
        effective_global_prefix_layers=chosen, prefix_policy_source=origin,
        retrieval_heads_promoted_global=converted,
        raw_matrix_counts=dict(Counter(head for row in matrix for head in row)),
        normalized_matrix_counts_before_promotion=dict(normalized_counts),
        raw_local_head_mask=local_mask, raw_local_head_count=normalized_counts.get("local", 0),
        raw_local_head_ratio=normalized_counts.get("local", 0)/total,
        effective_counts=effective, effective_full_recompute_heads=effective_full,
        effective_local_head_count=effective["local"], effective_local_head_ratio=effective["local"]/total,
        declared_summary_matches_effective=not mismatches if isinstance(declared_summary, dict) else None,
        summary_mismatches=mismatches, metadata_warnings=warnings,
        summary_global_semantics="global+dense unless summary explicitly distinguishes dense",
    )
    return profile, diagnostics
