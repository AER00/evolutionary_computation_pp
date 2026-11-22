#!/usr/bin/env python3
"""Generate the Task 7 report assets from the solver's run exports.

Usage from the repository root:

    go run . -repeat 20 -construction -json results_task7.json
    python3 tools/make_report_task7.py results_task7.json

The script writes the Task 7 figures to ``report/figures_task7/`` and prints
the report's tables. It reads the solver JSON only; the objective values are
recomputed from the instance CSVs and compared with the reported ones.
"""

from __future__ import annotations

import argparse
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

RANDOM = "RandomSolution"
BASELINE = "RandomStartSteepestEdges"
LM = "RandomStartSteepestEdgesLM"
MSLS = "MSLS"
ILS = "ILS"

LNS_NOLS = ["LNS-RandomNodes", "LNS-SingleSubpath", "LNS-MultiSubpath"]
LNS_LS = [m + "+LS" for m in LNS_NOLS]
LNS = LNS_NOLS + LNS_LS

CONSTRUCTION = [
    "ExtendEnd", "ExtendAny", "GreedyCycle",
    "RegretOnlyExtendAny", "RegretOnlyGreedyCycle",
    "RegretWeightedExtendAny", "RegretWeightedGreedyCycle",
]
SEARCH = [RANDOM, BASELINE, LM, MSLS, ILS] + LNS


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


def best_construction_table(stats: dict, instances: list[str]) -> str:
    rows = [
        "| Instance | Best greedy heuristic | Avg objective | Best run |",
        "|---|---|---:|---:|",
    ]
    for inst in instances:
        candidates = [
            (stats[(m, inst)]["avg"], m, stats[(m, inst)])
            for m in CONSTRUCTION if (m, inst) in stats
        ]
        avg, m, s = min(candidates)
        rows.append(
            f"| {inst} | {m} | {avg:,.1f} | {s['best']['objective']:,} "
            f"(run {s['best']['iteration']}) |"
        )
    return "\n".join(rows)


def best_local_search_table(stats: dict, instances: list[str]) -> str:
    rows = [
        "| Instance | Best local search | Avg objective | Avg time |",
        "|---|---|---:|---:|",
    ]
    for inst in instances:
        candidates = [
            (stats[(m, inst)]["avg"], m, stats[(m, inst)])
            for m in (BASELINE, LM) if (m, inst) in stats
        ]
        avg, m, s = min(candidates)
        rows.append(f"| {inst} | {m} | {avg:,.1f} | {s['t_avg']:,.2f} ms |")
    return "\n".join(rows)


def comparison_table(runs: list[dict], instances: list[str], methods: list[str]) -> str:
    """Average / best / worst plus the gain against MSLS and ILS."""
    index = defaultdict(list)
    for r in runs:
        index[(r["instance"], r["strategy"])].append(r)

    rows = [
        "| Instance | Method | Avg objective | Best | Worst | Gain vs MSLS | Gain vs ILS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for inst in instances:
        refs = {
            m: sum(r["objective"] for r in index[(inst, m)]) / len(index[(inst, m)])
            for m in (MSLS, ILS)
        }
        for m in methods:
            got = index[(inst, m)]
            objs = [r["objective"] for r in got]
            avg = sum(objs) / len(objs)
            rows.append(
                f"| {inst} | {m} | {avg:,.1f} | {min(objs):,} | {max(objs):,} | "
                f"{refs[MSLS] - avg:+,.1f} | {refs[ILS] - avg:+,.1f} |"
            )
    return "\n".join(rows)


def iterations_table(runs: list[dict], instances: list[str]) -> str:
    rows = [
        "| Instance | Method | Runs | Min | Max | Average | Total |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for inst in instances:
        for m in [ILS] + LNS:
            counts = [
                r["ls_runs"] for r in runs
                if r["instance"] == inst and r["strategy"] == m and r.get("ls_runs")
            ]
            if not counts:
                continue
            rows.append(
                f"| {inst} | {m} | {len(counts)} | {min(counts):,} | {max(counts):,} | "
                f"{sum(counts) / len(counts):,.1f} | {sum(counts):,} |"
            )
    return "\n".join(rows)


def destroy_summary_table(stats: dict, instances: list[str]) -> str:
    rows = [
        "| Destroy operator | " + " | ".join(
            f"{inst} no LS | {inst} +LS | {inst} gain" for inst in instances
        ) + " |",
        "|---|" + "|".join("---:" for _ in range(3 * len(instances))) + "|",
    ]
    for base in LNS_NOLS:
        cells = []
        for inst in instances:
            no = stats[(base, inst)]["avg"]
            with_ls = stats[(base + "+LS", inst)]["avg"]
            cells += [f"{no:,.1f}", f"{with_ls:,.1f}", f"{no - with_ls:+,.1f}"]
        rows.append(f"| {base.removeprefix('LNS-')} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def ranking_table(stats: dict, instances: list[str]) -> str:
    rows = [
        "| Rank | TSPA.csv method | Avg | TSPB.csv method | Avg |",
        "|---:|---|---:|---|---:|",
    ]
    ranked = {}
    for inst in instances:
        ranked[inst] = sorted(
            (stats[(m, inst)]["avg"], m) for m in [ILS] + LNS
        )
    for i in range(len(ranked[instances[0]])):
        cells = []
        for inst in instances:
            avg, m = ranked[inst][i]
            cells += [m, f"{avg:,.1f}"]
        rows.append(f"| {i + 1} | " + " | ".join(cells) + " |")
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


def local_optima_counts(runs: list[dict], instances: dict, methods: list[str]) -> dict:
    out = {}
    for name, inst in instances.items():
        for m in methods:
            subset = [r for r in runs if r["instance"] == name and r["strategy"] == m]
            if subset:
                out[(name, m)] = (
                    sum(best_improvement(inst, r["tour"]) >= 0 for r in subset),
                    len(subset),
                )
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_objective_ranges_combined(
    instances: list[str], stats: dict, path: Path
) -> None:
    """Stacked min–max panels, one per instance; dark rows are the LNS variants."""
    apply_theme()
    fig_h = 4.35 * len(instances) + 0.75
    fig, axes = plt.subplots(
        len(instances), 1, figsize=(7.4, fig_h), squeeze=False
    )
    fig.subplots_adjust(
        left=0.36, right=0.97, bottom=0.055,
        top=1 - 1.05 / fig_h, hspace=0.26,
    )

    for ax, instance_name in zip(axes[:, 0], instances):
        rows = []
        for m in (BASELINE, LM, MSLS, ILS) + tuple(LNS):
            s = stats.get((m, instance_name))
            if s is not None:
                rows.append({
                    "method": m, "min": s["min"], "max": s["max"], "avg": s["avg"],
                })

        ypos = list(range(len(rows)))[::-1]
        for y, r in zip(ypos, rows):
            dark = r["method"] in LNS
            color = ACCENT if dark else ACCENT_SOFT
            ax.plot(
                [r["min"], r["max"]], [y, y],
                color=color, linewidth=4.0, solid_capstyle="round", zorder=2,
            )
            ax.scatter(
                [r["avg"]], [y], s=38, color=color if dark else ACCENT,
                edgecolors=SURFACE, linewidths=1.1, zorder=3,
            )
            ax.annotate(
                f"{r['avg']:,.0f}", (r["avg"], y),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7.8, color=INK_SECONDARY, zorder=4,
            )

        ax.set_yticks(ypos)
        ax.set_yticklabels([r["method"] for r in rows], color=INK_PRIMARY, fontsize=8.4)
        ax.set_ylim(-0.7, len(rows) - 0.1)
        ax.set_xlabel("objective (lower is better)", fontsize=8.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        lo = min(r["min"] for r in rows)
        hi = max(r["max"] for r in rows)
        span = max(hi - lo, 1)
        ax.set_xlim(lo - span * 0.08, hi + span * 0.22)
        ax.set_title(instance_name, loc="left", pad=8, fontsize=10)

    _header(
        fig, fig_h, 0.55,
        "Objective across 20 runs per method",
        "bar = min–max range · dot = average · dark rows = LNS variants",
        tight=False,
    )
    save_figure(fig, path)


def plot_iteration_ranges(instances: list[str], stats: dict, path: Path) -> None:
    """Min–max main-loop iterations of the six LNS variants, one panel per instance."""
    apply_theme()
    fig, axes = plt.subplots(1, len(instances), figsize=(7.6, 3.9), squeeze=False)
    header = 0.66

    for ax, inst in zip(axes[0], instances):
        ypos = np.arange(len(LNS))[::-1]
        for y, m in zip(ypos, LNS):
            s = stats[(m, inst)]
            dark = m.endswith("+LS")
            color = ACCENT if dark else ACCENT_SOFT
            lo, hi = s["iter_min"], s["iter_max"]
            ax.plot(
                [lo, hi], [y, y],
                color=color, linewidth=4.5, solid_capstyle="round", zorder=2,
            )
            ax.scatter(
                [s["iter_avg"]], [y], s=46,
                color=color if dark else ACCENT,
                edgecolors=SURFACE, linewidths=1.2, zorder=3,
            )
            ax.annotate(
                f"{s['iter_avg']:,.0f}", (s["iter_avg"], y),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8.5, color=INK_SECONDARY, zorder=4,
            )

        ax.set_yticks(ypos)
        ax.set_yticklabels([m.removeprefix("LNS-") for m in LNS], color=INK_PRIMARY, fontsize=9)
        ax.set_ylim(-0.7, len(LNS) - 0.1)
        ax.set_xlabel("destroy–repair iterations in one run", fontsize=8.5)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        lo = min(stats[(m, inst)]["iter_min"] for m in LNS)
        hi = max(stats[(m, inst)]["iter_max"] for m in LNS)
        span = max(hi - lo, 1)
        ax.set_xlim(lo - span * 0.10, hi + span * 0.16)
        ax.set_title(inst, loc="left", pad=10, fontsize=10.5)

    _header(
        fig, 3.9, header,
        "Number of main-loop iterations across the 20 runs",
        "bar = min–max range · dot = average · dark rows = LNS with local search",
    )
    save_figure(fig, path)


def plot_gains(instances: list[str], stats: dict, path: Path) -> None:
    """Average improvement over MSLS, per method, as a two-panel bar chart."""
    apply_theme()
    methods = [ILS] + LNS
    labels = ["ILS"] + [m.removeprefix("LNS-") for m in LNS]
    fig, axes = plt.subplots(1, len(instances), figsize=(7.6, 3.9), squeeze=False)
    header = 0.66

    for ax, inst in zip(axes[0], instances):
        msls = stats[(MSLS, inst)]["avg"]
        gains = [msls - stats[(m, inst)]["avg"] for m in methods]
        ypos = np.arange(len(methods))[::-1]
        colors = [ACCENT if g > 0 else ACCENT_SOFT for g in gains]
        ax.barh(ypos, gains, height=0.55, color=colors, zorder=2)
        for y, g in zip(ypos, gains):
            ax.annotate(
                f"{g:+,.0f}", (g, y),
                textcoords="offset points", xytext=(5, 0),
                va="center", ha="left", fontsize=8.5, color=INK_SECONDARY,
            )
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, color=INK_PRIMARY, fontsize=9)
        ax.set_ylim(-0.7, len(methods) - 0.1)
        ax.set_xlabel("average objective below MSLS (positive = better)", fontsize=8.5)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        lo = min(gains)
        hi = max(gains)
        span = max(hi - lo, 1)
        ax.set_xlim(lo - span * 0.14, hi + span * 0.22)
        ax.set_title(inst, loc="left", pad=10, fontsize=10.5)

    _header(
        fig, 3.9, header,
        "Average improvement over MSLS",
        "bar = MSLS average − method average · higher is better",
    )
    save_figure(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("report/figures_task7"))
    args = parser.parse_args()

    runs = load_results(args.results)
    instances_list = list(dict.fromkeys(r["instance"] for r in runs))
    instances = {
        name: load_instance(Path(name)) for name in instances_list
    }

    stats = summarize(runs)
    add_component_averages(stats, runs)

    # Iteration counters live in the JSON ``ls_runs`` field: local-search runs
    # for ILS and destroy–repair iterations for LNS.
    iter_index = defaultdict(list)
    for r in runs:
        if r.get("ls_runs") is not None:
            iter_index[(r["strategy"], r["instance"])].append(r["ls_runs"])
    for key, counts in iter_index.items():
        stats[key]["iter_min"] = min(counts)
        stats[key]["iter_max"] = max(counts)
        stats[key]["iter_avg"] = sum(counts) / len(counts)

    checks = verify_runs(runs, instances)
    bad = [c for c in checks if not (c["valid"] and c["objective_matches"])]
    print(f"# verification: {len(checks) - len(bad)}/{len(checks)} runs valid and matching")

    methods = [m for m in SEARCH if any(k[0] == m for k in stats)]
    construct = [m for m in CONSTRUCTION if any(k[0] == m for k in stats)]

    print("\n## Construction heuristics (reference)\n")
    print(wide_table(stats, instances_list, construct, "objective"))
    print("\n## Best greedy heuristic\n")
    print(best_construction_table(stats, instances_list))
    print("\n## Best local search\n")
    print(best_local_search_table(stats, instances_list))
    print("\n## Objectives\n")
    print(wide_table(stats, instances_list, methods, "objective"))
    print("\n## Times\n")
    print(wide_table(stats, instances_list, methods, "time"))
    print("\n## LNS compared with MSLS and ILS\n")
    print(comparison_table(runs, instances_list, [MSLS, ILS] + LNS))
    print("\n## Main-loop iterations\n")
    print(iterations_table(runs, instances_list))
    print("\n## Destroy operators\n")
    print(destroy_summary_table(stats, instances_list))
    print("\n## Ranking\n")
    print(ranking_table(stats, instances_list))
    print("\n## Components\n")
    print(components_table(stats, instances_list, [BASELINE, LM, MSLS, ILS] + LNS))
    print("\n## Local optima\n")
    print(local_optima_table(runs, instances, [BASELINE, LM, MSLS, ILS] + LNS_LS))
    print("\n## Local optima, LNS without local search\n")
    no_ls = local_optima_counts(runs, instances, LNS_NOLS)
    for (inst, m), (ok, total) in no_ls.items():
        print(f"{inst}\t{m}\t{ok}/{total}")
    print("\n## Best runs\n")
    print(best_table(stats, instances_list, [RANDOM, BASELINE, LM, MSLS, ILS] + LNS))
    print("\n## Best tours\n")
    for inst in instances_list:
        best_lns = min(LNS, key=lambda m: stats[(m, inst)]["best"]["objective"])
        print(f"best LNS run on {inst}: {best_lns}")
        best_avg = min(LNS, key=lambda m: stats[(m, inst)]["avg"])
        print(f"best LNS average on {inst}: {best_avg}")
    tour_methods = [MSLS, ILS]
    print(best_tour_lists(stats, instances_list, tour_methods))
    for inst in instances_list:
        best_lns = min(LNS, key=lambda m: stats[(m, inst)]["best"]["objective"])
        print(f"\n**{inst} — {best_lns} — objective "
              f"{stats[(best_lns, inst)]['best']['objective']:,}, "
              f"run {stats[(best_lns, inst)]['best']['iteration']}**\n")
        print("```")
        print(", ".join(str(v) for v in stats[(best_lns, inst)]["best"]["tour"]))
        print("```")

    for name, inst in instances.items():
        best_lns = min(LNS, key=lambda m: stats[(m, name)]["avg"])
        base = best_lns.removesuffix("+LS")

        solutions = {
            MSLS: stats[(MSLS, name)]["best"]["tour"],
            ILS: stats[(ILS, name)]["best"]["tour"],
            base: stats[(base, name)]["best"]["tour"],
            base + "+LS": stats[(base + "+LS", name)]["best"]["tour"],
        }
        fig = plot_solutions(
            inst, solutions,
            title=f"{name} — best solution: MSLS, ILS and the best LNS operator",
            subtitle="node colour = cost · line = tour · lower objective is better",
            ncols=2,
        )
        save_figure(fig, args.out / f"best_solutions_{name.removesuffix('.csv')}.png")

    plot_objective_ranges_combined(
        instances_list, stats, args.out / "objective_ranges.png"
    )
    plot_iteration_ranges(instances_list, stats, args.out / "iterations.png")
    plot_gains(instances_list, stats, args.out / "gains_vs_msls.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
