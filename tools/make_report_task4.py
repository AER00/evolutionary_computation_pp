#!/usr/bin/env python3
"""Generate the Task 4 report assets from the solver's run exports.

Usage from the repository root:

    go run . -json results_task4.json
    go run . -candidates 5,8,10,13,20,30,50 -json results_task4_sweep.json
    python3 tools/check_with_excel.py results_task4.json \
        --out report/excel_check_task4.json
    python3 tools/make_report_task4.py results_task4.json \
        --sweep results_task4_sweep.json \
        --excel-check report/excel_check_task4.json

The script writes the Task 4 figures to ``report/figures_task4/`` and prints
the report's tables. It reads the solver JSON only; the objective values are
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from oeviz import (  # noqa: E402
    ACCENT,
    ACCENT_SOFT,
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
CANDIDATE_PREFIX = "RandomStartSteepestEdges"
CANDIDATE_SUFFIX = "Candidates"


def candidate_k(name: str) -> int | None:
    """Return k from ``RandomStartSteepestEdges<k>Candidates``, else None."""
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


def paired_table(runs: list[dict], instances: list[str], methods: list[str]) -> str:
    by_run: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for r in runs:
        by_run[(r["instance"], r["strategy"])][r["iteration"]] = r

    rows = [
        "| Instance | Comparison | Mean difference | Better / worse / tie (of 200) |",
        "|---|---|---:|---:|",
    ]
    pairs = [(m, BASELINE) for m in methods if m != BASELINE]
    pairs += [(methods[-1], methods[-2])] if len(methods) >= 3 else []
    for inst in instances:
        for newer, older in pairs:
            a = by_run[(inst, newer)]
            b = by_run[(inst, older)]
            common = sorted(set(a) & set(b))
            diffs = [a[i]["objective"] - b[i]["objective"] for i in common]
            mean = sum(diffs) / len(diffs)
            better = sum(d < 0 for d in diffs)
            worse = sum(d > 0 for d in diffs)
            tie = sum(d == 0 for d in diffs)
            rows.append(
                f"| {inst} | {newer} − {older} | {mean:+,.1f} | "
                f"{better} / {worse} / {tie} |"
            )
    return "\n".join(rows)


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
            # The averages of the two components are not stored by summarize,
            # so recover them from the run export's reported components.
            rows.append(
                f"| {inst} | {m} | {s['tour_avg']:,.1f} | {s['cost_avg']:,.1f} | "
                f"{s['avg']:,.1f} |"
            )
    return "\n".join(rows)


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


def sweep_table(sweep_stats: dict, instances: list[str], ks: list[int]) -> str:
    rows = [
        "| k | TSPA avg objective | TSPB avg objective | TSPA ms/run | TSPB ms/run |",
        "|---:|---:|---:|---:|---:|",
    ]
    rows.append(
        "| full | "
        + " | ".join(
            f"{sweep_stats[(BASELINE, inst)]['avg']:,.1f}" for inst in instances
        )
        + " | "
        + " | ".join(
            f"{sweep_stats[(BASELINE, inst)]['t_avg']:.2f}" for inst in instances
        )
        + " |"
    )
    for k in ks:
        name = f"{CANDIDATE_PREFIX}{k}{CANDIDATE_SUFFIX}"
        cells = []
        for inst in instances:
            s = sweep_stats.get((name, inst))
            cells.append(f"{s['avg']:,.1f}" if s else "—")
        times = []
        for inst in instances:
            s = sweep_stats.get((name, inst))
            times.append(f"{s['t_avg']:.2f}" if s else "—")
        rows.append(f"| {k} | " + " | ".join(cells) + " | " + " | ".join(times) + " |")
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


def plot_ranges(
    instance_name: str,
    rows: list[dict],
    path: Path,
) -> None:
    """One readable min–max range panel for a small method set."""
    apply_theme()
    fig_h = 0.62 * len(rows) + 1.55
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.18, top=1 - 0.62 / fig_h)

    labels = [r["method"] for r in rows]
    ypos = list(range(len(rows)))[::-1]
    for y, r in zip(ypos, rows):
        ax.plot(
            [r["min"], r["max"]], [y, y],
            color=ACCENT_SOFT, linewidth=4.5, solid_capstyle="round", zorder=2,
        )
        ax.scatter(
            [r["avg"]], [y], s=46, color=ACCENT,
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
        "bar = min–max range · dot = average",
        tight=False,
    )
    save_figure(fig, path)


def plot_candidate_sweep(
    instances: list[str],
    sweep_runs: list[dict],
    ks: list[int],
    path: Path,
) -> None:
    sweep_stats = summarize(sweep_runs)
    styles = {
        instances[0]: {"color": ACCENT, "marker": "o", "linestyle": "-"},
        instances[1]: {"color": INK_SECONDARY, "marker": "s", "linestyle": "--"},
    }

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.82, wspace=0.28)

    for inst in instances:
        name = [f"{CANDIDATE_PREFIX}{k}{CANDIDATE_SUFFIX}" for k in ks]
        objs = [sweep_stats[(n, inst)]["avg"] for n in name]
        times = [sweep_stats[(n, inst)]["t_avg"] for n in name]
        style = styles[inst]

        axes[0].plot(ks, objs, label=inst, markersize=5, linewidth=2, **style)
        axes[0].axhline(
            sweep_stats[(BASELINE, inst)]["avg"],
            color=style["color"], linewidth=1.0, linestyle=":",
        )
        axes[0].annotate(
            inst.replace(".csv", ""), (ks[-1], objs[-1]),
            textcoords="offset points", xytext=(8, 0),
            ha="left", va="center", fontsize=9, color=style["color"],
        )

        axes[1].plot(ks, times, label=inst, markersize=5, linewidth=2, **style)
        axes[1].axhline(
            sweep_stats[(BASELINE, inst)]["t_avg"],
            color=style["color"], linewidth=1.0, linestyle=":",
        )

    for ax, ylabel, title in (
        (axes[0], "average objective", "Average objective"),
        (axes[1], "average ms per run", "Average time per run"),
    ):
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("candidates per node (k)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, loc="left", pad=10, fontsize=10.5)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_xlim(ks[0] * 0.85, ks[-1] * 1.28)
    axes[1].set_xlim(ks[0] * 0.85, ks[-1] * 1.28)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))

    handles = [
        Line2D([0], [0], color=ACCENT, marker="o", linewidth=2,
               label=instances[0].replace(".csv", "")),
        Line2D([0], [0], color=INK_SECONDARY, marker="s", linestyle="--",
               linewidth=2, label=instances[1].replace(".csv", "")),
        Line2D([0], [0], color=INK_MUTED, linestyle=":", linewidth=1.0,
               label="full neighbourhood"),
    ]
    axes[1].legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)

    fig.text(
        0.012, 1 - 0.12 / 4.2, "Candidate-list size — quality and time",
        ha="left", va="top", fontsize=12.5, color=INK_PRIMARY,
    )
    fig.text(
        0.012, 1 - 0.34 / 4.2,
        "dots = candidate search; dotted horizontal line = full-neighbourhood search; "
        "200 runs per point and instance",
        ha="left", va="top", fontsize=8.5, color=INK_MUTED,
    )
    save_figure(fig, path)


def best_solution_cells(stats: dict, instances: list[str], methods: list[str]) -> str:
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


def tours_block(stats: dict, instances: list[str], methods: list[str]) -> str:
    out = []
    for inst in instances:
        out.append(f"**{inst}**")
        out.append("")
        for m in methods:
            best = stats[(m, inst)]["best"]
            out.append(f"`{m}` — objective {best['objective']:,} (run {best['iteration']}):")
            out.append("```")
            out.append(", ".join(str(v) for v in best["tour"]))
            out.append("```")
            out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="results_task4.json")
    ap.add_argument("--sweep", default="results_task4_sweep.json")
    ap.add_argument("--instances", default="TSPA.csv,TSPB.csv")
    ap.add_argument("--figures", default="report/figures_task4")
    ap.add_argument("--excel-check", default="report/excel_check_task4.json")
    args = ap.parse_args()

    runs = load_results(args.results)
    paths = [p.strip() for p in args.instances.split(",") if p.strip()]
    instances = {Path(p).name: load_instance(p) for p in paths}
    names = list(instances)

    stats = summarize(runs)
    add_component_averages(stats, runs)
    methods = list(dict.fromkeys(r["strategy"] for r in runs))
    if BASELINE not in methods:
        raise SystemExit(f"missing baseline {BASELINE}")

    apply_theme()
    figdir = Path(args.figures)

    for name, inst in instances.items():
        best = {m: stats[(m, name)]["best"]["tour"] for m in methods}
        fig = plot_solutions(
            inst, best, ncols=len(methods),
            title=f"{name} — best solution per method",
            subtitle=(
                f"{inst.n} nodes · each solution selects M = {inst.m} nodes · "
                "200 runs per method · colour = node cost · line = closed tour"
            ),
        )
        save_figure(fig, figdir / f"best_solutions_{Path(name).stem}.png")

        rows = [
            {"method": m, "min": stats[(m, name)]["min"],
             "max": stats[(m, name)]["max"], "avg": stats[(m, name)]["avg"]}
            for m in methods
        ]
        plot_ranges(
            name, rows, figdir / f"objective_ranges_{Path(name).stem}.png",
        )

    sweep_runs = load_results(args.sweep) if Path(args.sweep).exists() else []
    sweep_stats = summarize(sweep_runs) if sweep_runs else {}
    ks = sorted({
        k for r in sweep_runs
        if (k := candidate_k(r["strategy"])) is not None
    })
    if ks:
        plot_candidate_sweep(names, sweep_runs, ks, figdir / "candidate_sweep.png")

    print(f"figures written to {figdir}/")
    print()
    print("## Objective values\n")
    print(wide_table(stats, names, methods, "objective"))
    print()
    print("## Running times\n")
    print(wide_table(stats, names, methods, "time"))
    print()
    print("## Paired comparisons\n")
    print(paired_table(runs, names, methods))
    print()
    print("## Objective components\n")
    print(components_table(stats, names, methods))
    print()
    if ks:
        print("## Candidate-list size sweep\n")
        print(sweep_table(sweep_stats, names, ks))
        print()
    print("## Best runs\n")
    print(best_solution_cells(stats, names, methods))
    print()
    if Path(args.excel_check).exists():
        print("## Solution checker\n")
        print(checker_table(Path(args.excel_check), methods, names))
        print()
    print("## Best tours\n")
    print(tours_block(stats, names, [m for m in methods if candidate_k(m) is not None]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
