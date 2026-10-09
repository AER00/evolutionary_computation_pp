#!/usr/bin/env python3
"""Generate the report assets for task 1 from the solver's run export.

Pipeline:

    go run . -json results.json        # every run, every method, every instance
    .venv/bin/python tools/make_report.py results.json

The script writes the figures into ``report/figures/`` and prints the report's
tables as Markdown, so the numbers in REPORT.md always come from the raw solver
output rather than from a hand-written table. It also re-checks every run
against the instance CSV independently of the Go code.

Reusing this for a later task: keep ``oeviz`` for the charts and copy the
``main()`` driver, which only assumes the JSON export has the same shape.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oeviz import (  # noqa: E402
    apply_theme,
    load_instance,
    load_results,
    plot_objective_ranges,
    plot_solutions,
    save_figure,
    summarize,
    verify_runs,
)

GREEDY = ("ExtendEnd", "ExtendAny", "GreedyCycle")


def markdown_table(stats: dict, instances: list[str], methods: list[str]) -> str:
    lines = [
        "| Instance | Method | Runs | Min | Max | Average | Time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for inst in instances:
        for m in methods:
            s = stats.get((m, inst))
            if s is None:
                continue
            lines.append(
                f"| {inst} | {m} | {s['runs']} | {s['min']:,} | {s['max']:,} | "
                f"{s['avg']:,.1f} | {s['elapsed_ms']/1000:.2f} s |"
            )
    return "\n".join(lines)


def verification_table(checks: list[dict], methods: list[str], instances: list[str]) -> str:
    lines = [
        "| Instance | Method | Runs checked | Feasible | Objective matches | Components match |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for inst in instances:
        for m in methods:
            subset = [c for c in checks if c["method"] == m and c["instance"] == inst]
            if not subset:
                continue
            lines.append(
                f"| {inst} | {m} | {len(subset)} | "
                f"{sum(c['valid'] for c in subset)}/{len(subset)} | "
                f"{sum(c['objective_matches'] for c in subset)}/{len(subset)} | "
                f"{sum(c['components_match'] for c in subset)}/{len(subset)} |"
            )
    return "\n".join(lines)


def best_solution_table(stats: dict, instances: list[str], methods: list[str]) -> str:
    lines = [
        "| Instance | Method | Objective | Start node | Node indices (best solution) |",
        "|---|---|---:|---:|---|",
    ]
    for inst in instances:
        for m in methods:
            s = stats.get((m, inst))
            if s is None:
                continue
            best = s["best"]
            tour = ", ".join(str(v) for v in best["tour"])
            lines.append(
                f"| {inst} | {m} | {best['objective']:,} | {best['iteration']} | `{tour}` |"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="results.json")
    ap.add_argument("--instances", default="TSPA.csv,TSPB.csv")
    ap.add_argument("--figures", default="report/figures")
    args = ap.parse_args()

    runs = load_results(args.results)
    paths = [p.strip() for p in args.instances.split(",") if p.strip()]
    instances = {Path(p).name: load_instance(p) for p in paths}

    stats = summarize(runs)
    methods = list(dict.fromkeys(r["strategy"] for r in runs))
    names = list(instances)

    checks = verify_runs(runs, instances)
    bad = [c for c in checks if not (c["valid"] and c["objective_matches"] and c["components_match"])]

    apply_theme()
    figdir = Path(args.figures)

    for name, inst in instances.items():
        best = {
            m: stats[(m, name)]["best"]["tour"]
            for m in methods
            if (m, name) in stats
        }
        runs_per_method = Counter(r["strategy"] for r in runs if r["instance"] == name)
        fig = plot_solutions(
            inst, best,
            title=f"{name} — best solution per method",
            subtitle=(
                f"{inst.n} nodes · each solution selects M = {inst.m} nodes · "
                f"{next(iter(runs_per_method.values()))} runs per method · "
                "colour = node cost · line = closed tour"
            ),
        )
        save_figure(fig, figdir / f"best_solutions_{Path(name).stem}.png")

        rows = [
            {"method": m, "min": stats[(m, name)]["min"],
             "max": stats[(m, name)]["max"], "avg": stats[(m, name)]["avg"]}
            for m in methods if (m, name) in stats
        ]
        fig = plot_objective_ranges(
            name, rows, zoom_methods=GREEDY,
            title=(
                f"{name} — objective across "
                f"{runs_per_method[methods[0]]} runs per method"
            ),
            subtitle="bar = min–max range · dot = average",
        )
        save_figure(fig, figdir / f"objective_ranges_{Path(name).stem}.png")

    print(f"figures written to {figdir}/")
    print()
    print("## Computational experiment\n")
    print(markdown_table(stats, names, methods))
    print()
    print("## Solution checker\n")
    print(verification_table(checks, methods, names))
    print()
    print(f"runs checked: {len(checks)}, failing: {len(bad)}")
    if bad:
        print("FAILURES:", bad[:5])
    print()
    print("## Best solutions\n")
    print(best_solution_table(stats, names, methods))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
