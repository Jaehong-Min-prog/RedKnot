#!/usr/bin/env python3
"""Render only completed, checksum-verified SegPaged suite evidence.

No extrapolated points are inserted. Figures are optional and require
matplotlib; JSON and Markdown generation use the Python standard library.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def load_verified(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "segpaged-suite-v1" or manifest.get("status") != "complete":
        raise ValueError("Only a complete segpaged-suite-v1 manifest can be reported")
    reports = {}
    seen_paths = set()
    if not manifest.get("jobs"):
        raise ValueError("A suite needs at least one completed job")
    for job in manifest["jobs"]:
        if job.get("status") != "complete" or job.get("returncode") != 0:
            raise ValueError("Incomplete/failed jobs cannot be included")
        path = (directory / job["result"]).resolve()
        if path.parent != directory.resolve():
            raise ValueError("Result must belong to the suite directory")
        if path in seen_paths:
            raise ValueError("The same result cannot count as two different components")
        seen_paths.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != job["result_sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {path.name}")
        report = json.loads(path.read_text())
        if report.get("status") != "complete":
            raise ValueError(f"Incomplete child report: {path.name}")
        if job["component"] in reports:
            raise ValueError("Duplicate component evidence")
        reports[job["component"]] = report
    return manifest, reports


def table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def model_rows(report):
    rows = []
    for case in report.get("cases", []):
        runs = [run for run in case.get("runs", []) if not run.get("warmup", False)]
        arms = sorted({run["arm"] for run in runs})
        for arm in arms:
            selected = [run for run in runs if run["arm"] == arm]
            rows.append([case["dataset"], case["source_index"], arm, len(selected),
                         f"{statistics.median(r['ttft_seconds'] for r in selected):.4f}",
                         f"{statistics.median(r['request_seconds'] for r in selected):.4f}"])
    return rows


def make_plots(directory, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, yellow = "#2563a6", "#e9ab22"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    paths = []

    def save(fig, stem):
        fig.savefig(directory / (stem + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(directory / (stem + ".svg"), bbox_inches="tight")
        plt.close(fig)
        paths.append(stem + ".png")

    for name in ("operator", "public"):
        if name not in reports:
            continue
        cases = reports[name]["cases"]
        ratios = sorted({c["local_head_ratio"] for c in cases})
        queries = sorted({c["queries_per_head"] for c in cases})
        fig, axes = plt.subplots(1, len(queries), figsize=(6 * len(queries), 4.4), squeeze=False)
        for ax, nq in zip(axes[0], queries):
            for ratio in ratios:
                selected = sorted((c for c in cases if c["local_head_ratio"] == ratio and c["queries_per_head"] == nq),
                                  key=lambda c: c["length_tokens"])
                ax.plot([c["length_tokens"] / 1024 for c in selected],
                        [c["wall_median_speedup"] for c in selected], "o-", label=f"{ratio:.0%} local KV heads")
            ax.axhline(1, color="gray", linewidth=1, linestyle="--")
            ax.set(xlabel="Context (Ki tokens)", ylabel="Synchronized wall-time speedup (x)",
                   title=f"{nq} query token(s) per head")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"SegPaged {name}: visibility skipping / matched full traversal\nComponent latency, not model TTFT", fontsize=12)
        fig.tight_layout()
        save(fig, name + "_visibility")

    for index, case in enumerate(reports.get("cache", {}).get("cases", [])):
        rows = case["summary"]
        counts = sorted({r["live_requests"] for r in rows})
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
        panels = [("attach_wall_ms_p50", "Warm request binding", "Median wall time (ms)", 1),
                  ("active_bytes_after_repair", "Active physical pages after repair", "MiB (source cache included)", 1 / 2**20),
                  ("repair_wall_ms_p50", "Copy-on-write repair", "Median wall time (ms)", 1),
                  ("descriptor_wall_ms_p50", "Page descriptor construction", "Median batch wall time (ms)", 1)]
        for ax, (field, title, ylabel, scale) in zip(axes, panels):
            for offset, backend, color, label in [(-.18, "independent_pages", blue, "Independent pages"),
                                                  (.18, "shared_pages", yellow, "Shared document pages")]:
                lookup = {r["live_requests"]: r for r in rows if r["backend"] == backend}
                ax.bar([i + offset for i in range(len(counts))],
                       [lookup[n][field] * scale for n in counts], width=.36, color=color, label=label)
            ax.set_xticks(range(len(counts)), counts)
            ax.set(xlabel="Simultaneously live requests", ylabel=ylabel, title=title)
            ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Persistent non-prefix cache — {case['context_document_tokens']:,} document tokens\nSynthetic KV, not model TTFT or concurrent serving QPS", fontsize=12)
        fig.tight_layout()
        save(fig, f"cache_{index}_{case['context_document_tokens']}")
    if "head-sharing" in reports:
        cases = reports["head-sharing"]["cases"]
        sizes = sorted({c["document_tokens"] for c in cases})
        fig, axes = plt.subplots(2, len(sizes), figsize=(max(7, 5 * len(sizes)), 8), squeeze=False)
        for ax, length in zip(axes[0], sizes):
            selected = [c for c in cases if c["document_tokens"] == length]
            for offset, key, color, label in [(-.18, "independent_admitted", blue, "Independent KV"),
                                               (.18, "head_shared_admitted", yellow, "Local-head sharing")]:
                values = [c["summary"][key] for c in selected]
                bars = ax.bar([i + offset for i in range(len(selected))], values, width=.36, color=color, label=label)
                backend = "independent" if key == "independent_admitted" else "head_shared"
                ax.bar_label(bars, labels=[str(v) + ("+" if next(a for a in c["arms"] if a["backend"] == backend)["admission_is_censored"] else "")
                                          for c, v in zip(selected, values)], padding=3)
            ax.set_xticks(range(len(selected)), [f"{c['config']['local_head_ratio_actual']:.0%}" for c in selected])
            ax.set(xlabel="Reusable local KV heads", ylabel="Actually admitted requests",
                   title=f"{length:,} tokens; {selected[0]['document_full_head_logical_gib']:.3f} GiB full KV")
            ax.set_ylim(bottom=0, top=max(max(c["summary"][k] for k in ("independent_admitted", "head_shared_admitted"))
                                        for c in selected) * 1.18 + 1)
            ax.grid(axis="y", alpha=.2)
        for ax, length in zip(axes[1], sizes):
            selected = [c for c in cases if c["document_tokens"] == length]
            for offset, backend, color in [(-.18, "independent", blue), (.18, "head_shared", yellow)]:
                values = []
                for c in selected:
                    arm = next(a for a in c["arms"] if a["backend"] == backend)
                    step = next((r for r in arm["admission_steps"] if r.get("admitted") and r["resident_requests"] == 4), None)
                    values.append(step["after"]["allocated_bytes"] / 2**30 if step else float("nan"))
                bars = ax.bar([i + offset for i in range(len(selected))], values, width=.36, color=color)
                ax.bar_label(bars, labels=[f"{v:.3f}" if v == v else "N/A" for v in values], padding=3)
            ax.set_xticks(range(len(selected)), [f"{c['config']['local_head_ratio_actual']:.0%}" for c in selected])
            ax.set(xlabel="Reusable local KV heads", ylabel="Active KV page capacity (GiB)",
                   title="Matched workload: 4 resident requests\nIncludes local source cache, global/private KV and COW")
            ax.set_ylim(bottom=0)
            ax.margins(y=.17); ax.grid(axis="y", alpha=.2)
        axes[0][0].legend(fontsize=8)
        fig.suptitle("Same bounded KV budget: local pages shared, global pages request-private\nResident capacity, not measured serving QPS; '+' means guard-limited", fontsize=12)
        fig.tight_layout()
        save(fig, "head_sharing_capacity")
    return paths


def render(directory, plot=False):
    directory = Path(directory).resolve()
    manifest, reports = load_verified(directory)
    lines = ["# SegPaged Attention benchmark", "",
             f"Run: `{manifest['created_utc']}`. Only completed, SHA-256-verified files are included.", "",
             "## Interpretation boundaries", "",
             "Operator/API timings, cache-manager timings, and model TTFT are separate experiments. "
             "Do not multiply their speedups. Synthetic KV equality is not task accuracy. "
             "Active-page savings are not whole-model FLOP savings or a decrease in the fixed CUDA slab.", "",
             "**Concurrent serving QPS: NOT MEASURED by the component suite.** "
             "Several live request roots with serial attention calls measure cache occupancy/batch completion, "
             "not an LLM scheduler. Network/RDMA throughput and MLA integration are also not claimed.", ""]
    for name in ("operator", "public"):
        if name not in reports:
            continue
        lines += [f"## {name.capitalize()} visibility skipping", "",
                  "Same KV, positions, local/global policy and partition geometry in both arms. "
                  "Warmup/compilation excluded; all raw ABBA/BAAB samples are retained. "
                  "The zero-local-head case is a negative control.", ""]
        rows = []
        for c in reports[name]["cases"]:
            ref, new = (c["variants"][a]["synchronized_wall_ms"] for a in ("full_traversal", "skip_invisible"))
            rows.append([c["length_tokens"], c["queries_per_head"], f"{c['local_head_ratio']:.0%}",
                         f"{ref['median']:.4f}", f"{new['median']:.4f}", f"{new['p95']:.4f}",
                         f"{c['wall_median_speedup']:.3f}x", f"{c['correctness_after']['max_absolute_error']:.4g}"])
        lines += table(["Tokens", "Queries/head", "Local", "Full p50 ms", "Skip p50 ms", "Skip p95 ms", "Wall ratio", "Max abs error"], rows)
    if "cache" in reports:
        lines += ["## Persistent non-prefix cache", "",
                  "Offline source pages are published once and retained after the publisher request is released. "
                  "Online requests attach reordered documents; shared and copied arms have the same bounded slab "
                  "and consume the same repair payloads. Both use immutable pages/COW. "
                  "The copy baseline is this manager's per-page copy path, not an optimized contiguous-copy "
                  "implementation or a production vLLM/SGLang cache baseline. The shared fraction is a "
                  "document fraction, not the operator experiment's local-head fraction. All stored KV "
                  "participates in this cache experiment's global attention; sharing does not reduce its FLOPs. "
                  "This is an explicit synthetic position-independent storage contract, not a proof that arbitrary "
                  "real-model non-prefix KV is exact. Cold publication below means KV admission, not model prefill.", ""]
        for c in reports["cache"]["cases"]:
            lines += [f"### {c['context_document_tokens']:,} document tokens", ""]
            for arm in c["arms"]:
                lines += [f"- `{arm['backend']}` offline publication: {arm['offline_publish']['wall_ms']:.3f} ms."]
            lines += [""]
            rows = []
            for r in c["summary"]:
                rows.append([r["live_requests"], r["backend"], f"{r['attach_wall_ms_p50']:.3f}",
                             f"{r['repair_wall_ms_p50']:.3f}", f"{r['descriptor_wall_ms_p50']:.3f}",
                             f"{r['attention_batch_wall_ms_p50']:.3f}",
                             f"{r['active_bytes_after_repair']/2**20:.3f}", f"{r['reserved_bytes']/2**20:.3f}",
                             f"{r['attention_max_abs_error']:.4g}"])
            lines += table(["Live requests", "Backend", "Attach p50 ms", "Repair p50 ms", "Descriptor p50 ms", "Serial attention p50 ms", "Active MiB", "Fixed slab MiB", "Oracle max abs"], rows)
            traffic = [[r["live_requests"], r["backend"], f"{r['attach_written_bytes']/2**20:.3f}",
                        f"{r['repair_copied_bytes']/2**20:.3f}", f"{r['repair_written_bytes']/2**20:.3f}"]
                       for r in c["summary"]]
            lines += ["Payload counters below are recorded writes/copies, not measured HBM traffic or model FLOPs.", ""]
            lines += table(["Live requests", "Backend", "Attach written MiB", "Repair copied MiB", "Repair written MiB"], traffic)
    if "head-sharing" in reports:
        lines += ["## Same text, shared local heads, private global heads", "",
                  "This is the multi-request memory-capacity experiment. Only reusable local-head pages "
                  "remain in the source cache. Each request has different synthetic global-head K/V and a "
                  "private prefix. Local repair uses COW, including its transient page reservations. "
                  "The independent baseline does not keep an unused source cache by default. "
                  "Each arm really writes/adopts requests until the same bounded slab rejects admission; "
                  "this is not a formula-derived capacity. Model weights, source fixtures and verification "
                  "temporaries are outside this KV budget. Numerical checks do not prove real-model quality.", "",
                  "**A resident-capacity multiplier is not a QPS multiplier.** Request arrival scheduling, "
                  "global-head recomputation, model decoding and a throughput/latency SLA are not measured here. "
                  "Guard-limited runs are censored; they do not establish the maximum request capacity.", ""]
        rows = []
        memory_rows = []
        for c in reports["head-sharing"]["cases"]:
            s, config = c["summary"], c["config"]
            ratio = s["observed_resident_request_ratio"]
            rows.append([c["document_tokens"], f"{c['document_full_head_logical_gib']:.3f}",
                         f"{config['local_head_ratio_actual']:.0%}", f"{config['reserved_bytes']/2**30:.4f}",
                         s["independent_admitted"], s["head_shared_admitted"],
                         f"{ratio:.3f}x" if ratio is not None else "N/A", s["any_censored"]])
            for arm in c["arms"]:
                at_four = next((r for r in arm["admission_steps"] if r.get("admitted") and r["resident_requests"] == 4), None)
                errors = [v["attention_max_abs_error"] for v in arm["correctness"]["verified_requests"]]
                memory_rows.append([c["document_tokens"], f"{config['local_head_ratio_actual']:.0%}", arm["backend"],
                                    f"{arm['source_cache']['allocated_bytes']/2**20:.3f}",
                                    f"{at_four['after']['allocated_bytes']/2**20:.3f}" if at_four else "Not admitted",
                                    arm["termination"], f"{max(errors):.4g}" if errors else "Not verified"])
        lines += table(["Tokens", "Full-head KV GiB", "Local heads", "Same slab GiB", "Independent capacity", "Shared capacity", "Capacity ratio (not QPS)", "Censored"], rows)
        lines += table(["Tokens", "Local heads", "Backend", "Source-cache MiB", "Active MiB at 4 requests", "Termination", "Oracle max abs"], memory_rows)
    if "workflow" in reports:
        lines += ["## Actual model workflow", "",
                  "TTFT ends at the emitted first token. Online conversion, metadata, synchronization and seed "
                  "forward remain timed. Offline preparation is separately recorded. This current optional "
                  "query-stage bridge is not a persistent scheduler integration and can be slower. "
                  "The legacy generated output does not consume the diagnostic query logits; matching text "
                  "alone does not qualify the new query operator's task accuracy.", ""]
        lines += table(["Dataset", "Source row", "Arm", "Repeats", "Actual TTFT p50 s", "Full request p50 s"], model_rows(reports["workflow"]))
    else:
        lines += ["## Model-serving qualification", "",
                  "Not run in this suite invocation. No new model TTFT, task-accuracy, or concurrent QPS result. "
                  "Use the explicit workflow component for the existing experimental bridge, and a real scheduler "
                  "load test before claiming serving throughput.", ""]
    paths = make_plots(directory, reports) if plot else []
    if paths:
        lines += ["## Figures", ""]
        for path in paths:
            lines += [f"![{Path(path).stem}]({path})", ""]
    lines += ["## Raw evidence", ""]
    for job in manifest["jobs"]:
        lines += [f"- [{job['component']} raw JSON]({job['result']}); [complete log]({job['log']}); "
                  f"SHA-256 `{job['result_sha256']}`."]
    lines += ["", "Commands and source hashes are recorded in [manifest.json](manifest.json).", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
    provenance = dict(rendered_utc=datetime.now(timezone.utc).isoformat(),
                      renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      evidence_manifest_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
                      plot_requested=plot, figures=paths,
                      note="Rendering may be rerun with a newer renderer; measurement evidence and its hashes are unchanged.")
    (directory / "REPORT.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return directory / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    print(render(args.directory, plot=args.plot))
