#!/usr/bin/env python3
"""Generate the Task 6 report assets from the solver's run exports.

Usage from the repository root:

    go run . -repeat 20 -construction -json results_task6.json
    python3 tools/make_report_task6.py results_task6.json

The script writes the Task 6 figures to ``report/figures_task6/`` and prints
the report's tables. It reads the solver JSON only; the objective values are
recomputed from the instance CSVs and compared with the reported ones.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from oeviz import (  # noqa: E402
    ACCENT,
    ACCENT_SOFT,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    _header,
    apply_theme,
    load_instance,
    load_results,
    plot_solutions,
    save_figure,
    summarize,
    verify_runs,
)

BASELINE = "RandomStartSteepestEdges"
LM = "RandomStartSteepestEdgesLM"
MSLS = "MSLS"
ILS = "ILS"

CONSTRUCTION = [
    "ExtendEnd", "ExtendAny", "GreedyCycle",
    "RegretOnlyExtendAny", "RegretOnlyGreedyCycle",
    "RegretWeightedExtendAny", "RegretWeightedGreedyCycle",
]
SEARCH = ["RandomSolution", BASELINE, LM, MSLS, ILS]


def wide_table(stats: dict, instances: list[str], methods: list[str], kind: str) -> str:
    head = "| Method | " + " | ".join(f"{inst} av (min–max)" for inst in instances) + " |"
    sep = "|---|" + "|".join("---:" for _ in instances) + "|"
    lines = [head, sep]
    for m in methods:
        cells = []
        for inst in instances:
            s = stats.get((m, inst))
            if s is None:
                cells.append("—")
            elif kind == "objective":
                cells.append(f"{s['avg']:,.1f} ({s['min']:,} – {s['max']:,})")
            else:
                cells.append(f"{s['t_avg']:,.2f} ms ({s['t_min']:,.2f} – {s['t_max']:,.2f})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def add_component_averages(stats: dict, runs: list[dict]) -> None:
    for r in runs:
        s = stats[(r["strategy"], r["instance"])]
        s.setdefault("tour_sum", 0)
        s.setdefault("cost_sum", 0)
        s["tour_sum"] += r["tour_len"]
        s["cost_sum"] += r["node_cost"]
    for s in stats.values():
        s["tour_avg"] = s["tour_sum"] / s["runs"]
        s["cost_avg"] = s["cost_sum"] / s["runs"]


def components_table(stats: dict, instances: list[str], methods: list[str]) -> str:
    rows = [
        "| Instance | Method | Avg tour length | Avg node cost | Avg objective |",
        "|---|---|---:|---:|---:|",
    ]
    for inst in instances:
        for m in methods:
            s = stats.get((m, inst))
            if s is None:
                continue
            rows.append(
                f"| {inst} | {m} | {s['tour_avg']:,.1f} | {s['cost_avg']:,.1f} | "
                f"{s['avg']:,.1f} |"
            )
    return "\n".join(rows)


def comparison_table(runs: list[dict], instances: list[str]) -> str:
    """MSLS against ILS on the same instance, side by side."""
    index = defaultdict(list)
    for r in runs:
        index[(r["instance"], r["strategy"])].append(r)

    rows = [
        "| Instance | Method | Avg objective | Best | Worst | Avg time |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for inst in instances:
        for m in (MSLS, ILS):
            got = index[(inst, m)]
            objs = [r["objective"] for r in got]
            times = [r["elapsed_ms"] for r in got]
            rows.append(
                f"| {inst} | {m} | {sum(objs) / len(objs):,.1f} | {min(objs):,} | "
                f"{max(objs):,} | {sum(times) / len(times):,.2f} ms |"
            )
    return "\n".join(rows)


def window_table(runs: list[dict], instances: list[str]) -> str:
    """How each ILS run compares with the whole MSLS distribution."""
    index = defaultdict(list)
    for r in runs:
        index[(r["instance"], r["strategy"])].append(r)

    rows = [
        "| Instance | ILS better than all 20 MSLS runs | ILS better than MSLS average | "
        "ILS worse than MSLS best | Avg gain vs MSLS average |",
        "|---|---:|---:|---:|---:|",
    ]
    for inst in instances:
        msls = [r["objective"] for r in index[(inst, MSLS)]]
        ils = [r["objective"] for r in index[(inst, ILS)]]
        msls_avg = sum(msls) / len(msls)
        rows.append(
            f"| {inst} | {sum(o < min(msls) for o in ils)}/{len(ils)} | "
            f"{sum(o < msls_avg for o in ils)}/{len(ils)} | "
            f"{sum(o > min(msls) for o in ils)}/{len(ils)} | "
            f"{msls_avg - sum(ils) / len(ils):+,.1f} |"
        )
    return "\n".join(rows)


def ls_runs_table(runs: list[dict], instances: list[str]) -> str:
    rows = [
        "| Instance | Runs | Min | Max | Average | Total local searches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for inst in instances:
        counts = [r["ls_runs"] for r in runs if r["instance"] == inst and r["strategy"] == ILS]
        if not counts:
            continue
        rows.append(
            f"| {inst} | {len(counts)} | {min(counts)} | {max(counts)} | "
            f"{sum(counts) / len(counts):,.1f} | {sum(counts):,} |"
        )
    return "\n".join(rows)


def best_table(stats: dict, instances: list[str], methods: list[str]) -> str:
    rows = [
        "| Instance | Method | Best objective | Run | Tour length | Node cost |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for inst in instances:
        for m in methods:
            s = stats.get((m, inst))
            if s is None:
                continue
            best = s["best"]
            rows.append(
                f"| {inst} | {m} | {best['objective']:,} | {best['iteration']} | "
                f"{best['tour_len']:,} | {best['node_cost']:,} |"
            )
    return "\n".join(rows)


def best_tour_lists(stats: dict, instances: list[str], methods: list[str]) -> str:
    blocks = []
    for inst in instances:
        for m in methods:
            s = stats.get((m, inst))
            if s is None:
                continue
            best = s["best"]
            tour = ", ".join(str(v) for v in best["tour"])
            blocks.append(
                f"**{inst} — {m} — objective {best['objective']:,}, run {best['iteration']}**\n\n"
                f"```\n{tour}\n```"
            )
    return "\n\n".join(blocks)


def best_improvement(inst, tour: list[int]) -> int:
    """Most negative delta in the full neighbourhood (inter + 2-opt), or 0."""
    m = len(tour)
    t = np.asarray(tour, dtype=np.int64)
    selected = np.zeros(inst.n, dtype=bool)
    selected[t] = True
    unselected = np.flatnonzero(~selected)
    best = 0

    for i in range(m):
        old, prev, nxt = t[i], t[(i - 1) % m], t[(i + 1) % m]
        base = -inst.costs[old] - inst.dist[prev, old] - inst.dist[old, nxt]
        if len(unselected):
            deltas = (
                base
                + inst.costs[unselected]
                + inst.dist[prev, unselected]
                + inst.dist[unselected, nxt]
            )
            best = min(best, int(deltas.min()))

    for i in range(m):
        for j in range(i + 2, m):
            if i == 0 and j == m - 1:
                continue
            a, b = t[i], t[(i + 1) % m]
            c, d = t[j], t[(j + 1) % m]
            delta = int(
                inst.dist[a, c] + inst.dist[b, d]
                - inst.dist[a, b] - inst.dist[c, d]
            )
            if delta < best:
                best = delta
    return best


def local_optima_table(runs: list[dict], instances: dict, methods: list[str]) -> str:
    rows = [
        "| Instance | Method | Runs checked | Full local optima |",
        "|---|---|---:|---:|",
    ]
    for name, inst in instances.items():
        for m in methods:
            subset = [r for r in runs if r["instance"] == name and r["strategy"] == m]
            if not subset:
                continue
            ok = sum(best_improvement(inst, r["tour"]) >= 0 for r in subset)
            rows.append(f"| {name} | {m} | {len(subset)} | {ok}/{len(subset)} |")
    return "\n".join(rows)


def plot_objective_ranges(instance_name: str, rows: list[dict], path: Path) -> None:
    """One min–max range panel per instance (the method set is small)."""
    apply_theme()
    fig_h = 0.62 * len(rows) + 1.55
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.18, top=1 - 0.62 / fig_h)

    labels = [r["method"] for r in rows]
    ypos = list(range(len(rows)))[::-1]
    for y, r in zip(ypos, rows):
        dark = r["method"] in (MSLS, ILS)
        color = ACCENT if dark else ACCENT_SOFT
        ax.plot(
            [r["min"], r["max"]], [y, y],
            color=color, linewidth=4.5, solid_capstyle="round", zorder=2,
        )
        ax.scatter(
            [r["avg"]], [y], s=46, color=color if dark else ACCENT,
            edgecolors=SURFACE, linewidths=1.2, zorder=3,
        )
        ax.annotate(
            f"{r['avg']:,.0f}", (r["avg"], y),
            textcoords="offset points", xytext=(0, 11),
            ha="center", fontsize=8.5, color=INK_SECONDARY, zorder=4,
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=INK_PRIMARY, fontsize=9.5)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel("objective (lower is better)", fontsize=8.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    lo = min(r["min"] for r in rows)
    hi = max(r["max"] for r in rows)
    span = max(hi - lo, 1)
    ax.set_xlim(lo - span * 0.08, hi + span * 0.22)

    _header(
        fig, fig_h, 0.62,
        f"{instance_name} — objective across 20 runs per method",
        "bar = min–max range · dot = average · dark rows = the two new methods",
        tight=False,
    )
    save_figure(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("report/figures_task6"))
    args = parser.parse_args()

    runs = load_results(args.results)
    instances_list = list(dict.fromkeys(r["instance"] for r in runs))
    instances = {
        name: load_instance(Path(name)) for name in instances_list
    }

    stats = summarize(runs)
    add_component_averages(stats, runs)

    checks = verify_runs(runs, instances)
    bad = [c for c in checks if not (c["valid"] and c["objective_matches"])]
    print(f"# verification: {len(checks) - len(bad)}/{len(checks)} runs valid and matching")

    methods = [m for m in SEARCH if any(k[0] == m for k in stats)]
    construct = [m for m in CONSTRUCTION if any(k[0] == m for k in stats)]
    print("\n## Construction heuristics\n")
    print(wide_table(stats, instances_list, construct, "objective"))
    print("\n## Objectives\n")
    print(wide_table(stats, instances_list, methods, "objective"))
    print("\n## Times\n")
    print(wide_table(stats, instances_list, methods, "time"))
    print("\n## MSLS vs ILS\n")
    print(comparison_table(runs, instances_list))
    print("\n## ILS against the MSLS distribution\n")
    print(window_table(runs, instances_list))
    print("\n## ILS local search runs\n")
    print(ls_runs_table(runs, instances_list))
    print("\n## Components\n")
    print(components_table(stats, instances_list, methods))
    print("\n## Local optima\n")
    print(local_optima_table(runs, instances, [BASELINE, LM, MSLS, ILS]))
    print("\n## Best runs\n")
    print(best_table(stats, instances_list, methods))
    print("\n## Best tours\n")
    print(best_tour_lists(stats, instances_list, [MSLS, ILS]))

    for name, inst in instances.items():
        solutions = {
            MSLS: stats[(MSLS, name)]["best"]["tour"],
            ILS: stats[(ILS, name)]["best"]["tour"],
        }
        fig = plot_solutions(
            inst, solutions,
            title=f"{name} — best solution, MSLS vs ILS",
            subtitle="node colour = cost · line = tour · lower objective is better",
        )
        save_figure(fig, args.out / f"best_solutions_{name.removesuffix('.csv')}.png")

        rows = []
        for m in (BASELINE, LM, MSLS, ILS):
            s = stats.get((m, name))
            if s is not None:
                rows.append({
                    "method": m,
                    "min": s["min"],
                    "max": s["max"],
                    "avg": s["avg"],
                })
        plot_objective_ranges(
            name, rows,
            args.out / f"objective_ranges_{name.removesuffix('.csv')}.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
