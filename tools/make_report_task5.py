#!/usr/bin/env python3
"""Generate the Task 5 report assets from the solver's run exports.

Usage from the repository root:

    go run . -repeat 200 -json results_task5.json
    python3 tools/check_with_excel.py results_task5.json \
        --out report/excel_check_task5.json
    python3 tools/make_report_task5.py results_task5.json \
        --excel-check report/excel_check_task5.json

The script writes the Task 5 figures to ``report/figures_task5/`` and prints the
report's tables. It reads the solver JSON only; the objective values are
recomputed by the workbook checker, not by this script.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from oeviz import (  # noqa: E402
    ACCENT,
    ACCENT_SOFT,
    AXIS,
    INK_MUTED,
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
)

BASELINE = "RandomStartSteepestEdges"
LM = "RandomStartSteepestEdgesLM"
CANDIDATE_PREFIX = "RandomStartSteepestEdges"
CANDIDATE_SUFFIX = "Candidates"


def candidate_k(name: str) -> int | None:
    if not (name.startswith(CANDIDATE_PREFIX) and name.endswith(CANDIDATE_SUFFIX)):
        return None
    middle = name[len(CANDIDATE_PREFIX):-len(CANDIDATE_SUFFIX)]
    try:
        return int(middle)
    except ValueError:
        return None


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
                cells.append(f"{s['t_avg']:.2f} ms ({s['t_min']:.2f} – {s['t_max']:.2f})")
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


def by_run(runs: list[dict]) -> dict[tuple[str, str], dict[int, dict]]:
    out: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for r in runs:
        out[(r["instance"], r["strategy"])][r["iteration"]] = r
    return out


def paired_table(runs: list[dict], instances: list[str]) -> str:
    index = by_run(runs)
    rows = [
        "| Instance | Comparison | Mean difference | Min – max difference | "
        "Better / worse / tie | Identical tours |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for inst in instances:
        newer = index[(inst, LM)]
        older = index[(inst, BASELINE)]
        common = sorted(set(newer) & set(older))
        diffs = [newer[i]["objective"] - older[i]["objective"] for i in common]
        same = sum(newer[i]["tour"] == older[i]["tour"] for i in common)
        rows.append(
            f"| {inst} | LM − full | {sum(diffs) / len(diffs):+,.1f} | "
            f"{min(diffs):+,} – {max(diffs):+,} | "
            f"{sum(d < 0 for d in diffs)} / {sum(d > 0 for d in diffs)} / "
            f"{sum(d == 0 for d in diffs)} | {same}/{len(common)} |"
        )
    return "\n".join(rows)


def speed_table(runs: list[dict], instances: list[str]) -> str:
    index = by_run(runs)
    rows = [
        "| Instance | Aggregate speedup | Median per-run speedup | "
        "Min – max per-run speedup | LM faster |",
        "|---|---:|---:|---:|---:|",
    ]
    for inst in instances:
        lm = index[(inst, LM)]
        full = index[(inst, BASELINE)]
        common = sorted(set(lm) & set(full))
        full_total = sum(full[i]["elapsed_ms"] for i in common)
        lm_total = sum(lm[i]["elapsed_ms"] for i in common)
        ratios = sorted(full[i]["elapsed_ms"] / lm[i]["elapsed_ms"] for i in common)
        median = (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]) / 2
        faster = sum(lm[i]["elapsed_ms"] < full[i]["elapsed_ms"] for i in common)
        rows.append(
            f"| {inst} | {full_total / lm_total:.2f}× | "
            f"{median:.2f}× | {ratios[0]:.2f}× – {ratios[-1]:.2f}× | "
            f"{faster}/{len(common)} |"
        )
    return "\n".join(rows)


def best_table(stats: dict, instances: list[str], methods: list[str]) -> str:
    rows = [
        "| Instance | Method | Best objective | Run | Tour length | Node cost |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for inst in instances:
        for m in methods:
            s = stats[(m, inst)]
            best = s["best"]
            rows.append(
                f"| {inst} | {m} | {best['objective']:,} | {best['iteration']} | "
                f"{best['tour_len']:,} | {best['node_cost']:,} |"
            )
    return "\n".join(rows)


def checker_table(excel_path: Path, methods: list[str], instances: list[str]) -> str:
    data = json.loads(excel_path.read_text())
    checks = data["runs"]
    rows = [
        "| Instance | Method | Runs checked | Feasible | Objective matches | Components match |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for inst in instances:
        for m in methods:
            subset = [c for c in checks if c["method"] == m and c["instance"] == inst]
            if not subset:
                continue
            n = len(subset)
            rows.append(
                f"| {inst} | {m} | {n} | {sum(c['valid'] for c in subset)}/{n} | "
                f"{sum(c['objective_matches'] for c in subset)}/{n} | "
                f"{sum(c['components_match'] for c in subset)}/{n} |"
            )
    return "\n".join(rows)


def best_improvement(inst, tour: list[int]) -> int:
    """Most negative delta in the full neighbourhood (inter + 2-opt), or 0."""
    m = len(tour)
    t = np.asarray(tour, dtype=np.int64)
    selected = np.zeros(inst.n, dtype=bool)
    selected[t] = True
    unselected = np.flatnonzero(~selected)
    best = 0

    # Inter-route: replace t[i] by every unselected node.
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

    # Intra-route: all 2-opt exchanges.
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


def local_optima_table(
    runs: list[dict], instances: dict, methods: list[str]
) -> str:
    rows = [
        "| Instance | Method | Runs checked | Full local optima |",
        "|---|---|---:|---:|",
    ]
    for name, inst in instances.items():
        for m in methods:
            subset = [r for r in runs if r["instance"] == name and r["strategy"] == m]
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
        color = ACCENT if r["method"] == LM else ACCENT_SOFT
        ax.plot(
            [r["min"], r["max"]], [y, y],
            color=color, linewidth=4.5, solid_capstyle="round", zorder=2,
        )
        ax.scatter(
            [r["avg"]], [y], s=46, color=color if r["method"] == LM else ACCENT,
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
        f"{instance_name} — objective across 200 runs per method",
        "bar = min–max range · dot = average · dark row = list of improving moves",
        tight=False,
    )
    save_figure(fig, path)


def plot_time_comparison(
    instances: list[str],
    runs: list[dict],
    methods: list[str],
    path: Path,
) -> None:
    """Per-run time distributions for the full search and the LM search."""
    show = [BASELINE, LM]
    colors = [ACCENT_SOFT, ACCENT]
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.16, top=0.78, wspace=0.28)

    for ax, inst in zip(axes, instances):
        data = [
            [r["elapsed_ms"] for r in runs
             if r["instance"] == inst and r["strategy"] == m]
            for m in show
        ]
        box = ax.boxplot(
            data, positions=[2, 1], orientation="horizontal", widths=0.5,
            patch_artist=True, showfliers=False, zorder=2,
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(SURFACE)
            patch.set_alpha(0.9)
        for element in ("whiskers", "caps", "medians"):
            for line in box[element]:
                line.set_color(INK_SECONDARY)
                line.set_linewidth(1.0)

        # Individual runs: light jitter so the distribution is visible.
        rng = np.random.default_rng(5)
        for y, values, color in zip((2, 1), data, colors):
            ax.scatter(
                values, y + rng.uniform(-0.12, 0.12, len(values)),
                s=5, color=color, alpha=0.30, linewidths=0, zorder=3,
            )

        full_avg = sum(data[0]) / len(data[0])
        lm_avg = sum(data[1]) / len(data[1])
        for y, values, avg in ((2, data[0], full_avg), (1, data[1], lm_avg)):
            ax.annotate(
                f"{avg:.2f} ms", (max(values) + 0.20, y),
                ha="left", va="center", fontsize=8.5, color=INK_SECONDARY,
            )
        ax.text(
            0.02, 0.05, f"average speedup {full_avg / lm_avg:.2f}×",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=INK_PRIMARY,
        )

        ax.set_yticks([2, 1])
        ax.set_yticklabels(["full search", "LM search"], color=INK_PRIMARY, fontsize=9.5)
        ax.set_ylim(0.4, 2.5)
        ax.set_xlim(0, max(max(v) for v in data) * 1.20)
        ax.set_xlabel("time per run (ms, lower is better)", fontsize=8.5)
        ax.set_title(inst, loc="left", pad=10, fontsize=10.5)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    fig.text(
        0.012, 1 - 0.12 / 3.9, "Running time — full steepest search vs LM",
        ha="left", va="top", fontsize=12.5, color=INK_PRIMARY,
    )
    fig.text(
        0.012, 1 - 0.34 / 3.9,
        "box = quartiles and median · dots = individual runs (200 per method) · "
        "same random starts in every pair",
        ha="left", va="top", fontsize=8.5, color=INK_MUTED,
    )
    save_figure(fig, path)


def plot_2opt_variants(path: Path) -> None:
    """Schematic of the two stored 2-opt variants for one pair of edges."""
    apply_theme()
    fig_h = 3.5
    fig, axes = plt.subplots(1, 2, figsize=(8.6, fig_h))
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.10, top=0.72, wspace=0.12)

    pos = {"a": (0.18, 0.80), "b": (0.82, 0.80),
           "c": (0.18, 0.20), "d": (0.82, 0.20)}
    panels = [
        ("same direction", [("a", "c"), ("b", "d")],
         "stored {a, b, c, d}: adds (a,c) and (b,d)"),
        ("inverted direction", [("a", "d"), ("b", "c")],
         "stored {a, b, d, c}: adds (a,d) and (b,c)"),
    ]

    for ax, (title, added, note) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Removed edges: the same two undirected pairs in both variants.
        for u, v in (("a", "b"), ("c", "d")):
            ax.plot(
                [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=INK_MUTED, linewidth=2.2, linestyle=(0, (4, 3)), zorder=1,
            )
        # Added edges of this variant.
        for u, v in added:
            ax.plot(
                [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=ACCENT, linewidth=2.8, zorder=2,
            )
        for name, (x, y) in pos.items():
            ax.scatter(
                [x], [y], s=430, color=SURFACE,
                edgecolors=INK_PRIMARY, linewidths=1.4, zorder=3,
            )
            ax.text(
                x, y, name, ha="center", va="center",
                fontsize=12.5, color=INK_PRIMARY, zorder=4,
            )
        ax.set_title(title, loc="left", pad=8, fontsize=10.5)
        ax.text(
            0.0, 0.02, note, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY,
        )

    _header(
        fig, fig_h, 0.55,
        "The two stored 2-opt variants for one pair of edges",
        "dashed = removed edges · solid = added edges · only one variant "
        "is applicable at a given orientation",
        tight=False,
    )
    save_figure(fig, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="results_task5.json")
    ap.add_argument("--instances", default="TSPA.csv,TSPB.csv")
    ap.add_argument("--figures", default="report/figures_task5")
    ap.add_argument("--excel-check", default="report/excel_check_task5.json")
    ap.add_argument("--local-optima", action="store_true",
                    help="also verify that baseline and LM tours are full local optima")
    args = ap.parse_args()

    runs = load_results(args.results)
    paths = [p.strip() for p in args.instances.split(",") if p.strip()]
    instances = {Path(p).name: load_instance(p) for p in paths}
    names = list(instances)

    stats = summarize(runs)
    add_component_averages(stats, runs)
    methods = list(dict.fromkeys(r["strategy"] for r in runs))
    if BASELINE not in methods or LM not in methods:
        raise SystemExit(f"missing {BASELINE} or {LM} in {args.results}")

    plot_methods = [m for m in methods if m != "RandomSolution"]

    apply_theme()
    figdir = Path(args.figures)

    for name, inst in instances.items():
        best = {m: stats[(m, name)]["best"]["tour"] for m in (BASELINE, LM)}
        fig = plot_solutions(
            inst, best, ncols=2,
            title=f"{name} — best solution: full steepest vs LM",
            subtitle=(
                f"{inst.n} nodes · each solution selects M = {inst.m} nodes · "
                "200 runs per method · colour = node cost · line = closed tour"
            ),
        )
        save_figure(fig, figdir / f"best_solutions_{Path(name).stem}.png")

        rows = [
            {"method": m, "min": stats[(m, name)]["min"],
             "max": stats[(m, name)]["max"], "avg": stats[(m, name)]["avg"]}
            for m in plot_methods
        ]
        plot_objective_ranges(
            name, rows, figdir / f"objective_ranges_{Path(name).stem}.png",
        )

    plot_time_comparison(
        names, runs, [BASELINE, LM], figdir / "time_comparison.png",
    )
    plot_2opt_variants(figdir / "2opt_variants.png")

    print(f"figures written to {figdir}/")
    print()
    print("## Objective values\n")
    print(wide_table(stats, names, methods, "objective"))
    print()
    print("## Running times\n")
    print(wide_table(stats, names, methods, "time"))
    print()
    print("## Paired comparison: LM against the full search\n")
    print(paired_table(runs, names))
    print()
    print("## Speedup\n")
    print(speed_table(runs, names))
    print()
    print("## Objective components\n")
    print(components_table(stats, names, methods))
    print()
    print("## Best runs\n")
    print(best_table(stats, names, methods))
    print()
    if args.local_optima:
        print("## Full-neighbourhood local-optimum check\n")
        print(local_optima_table(runs, instances, [BASELINE, LM]))
        print()
    if Path(args.excel_check).exists():
        print("## Solution checker\n")
        print(checker_table(Path(args.excel_check), methods, names))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
