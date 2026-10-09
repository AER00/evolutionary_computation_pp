"""Reusable plotting and verification helpers for the OE course tasks.

Deliberately task-agnostic: an *instance* is an ``x;y;cost`` CSV, a *solution*
is a list of node indices. Nothing here knows about a particular heuristic, so
later tasks can import this module unchanged:

    from oeviz import load_instance, plot_solutions, objective, save_figure

    inst = load_instance("TSPA.csv")
    fig = plot_solutions(inst, {"MyMethod": tour}, title="...")
    save_figure(fig, "report/figures/my_figure.png")

The chart colours implement the project data-viz palette (light mode): one
sequential blue ramp for magnitude (node cost), neutral greys for chrome, and a
single accent hue for every non-magnitude mark, so no chart depends on telling
similar hues apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # no display needed; we always write files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# --------------------------------------------------------------------------
# Palette and theme
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"  # chart surface
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"  # categorical slot 1 / accent for single-series marks
ACCENT_SOFT = "#9ec5f4"  # light step of the same hue, for range bars

# Sequential blue ramp, light -> dark (magnitude: node cost).
RAMP_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
COST_CMAP = LinearSegmentedColormap.from_list("oe_blue", RAMP_STEPS)

DPI = 200


def apply_theme() -> None:
    """Set matplotlib defaults so every figure matches the palette."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 10.5,
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


# --------------------------------------------------------------------------
# Instances, objectives and validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One problem instance: node coordinates, node costs, derived distances."""

    name: str
    xs: np.ndarray
    ys: np.ndarray
    costs: np.ndarray
    dist: np.ndarray

    @property
    def n(self) -> int:
        return len(self.costs)

    @property
    def m(self) -> int:
        """How many nodes a solution must select: ceil(n / 2)."""
        return (self.n + 1) // 2


def load_instance(path: str | Path) -> Instance:
    """Read an ``x;y;cost`` CSV and precompute the integer distance matrix.

    Rows that do not hold three parsable integers are skipped, and distances are
    rounded to the nearest integer exactly as the Go solver does
    (``int(sqrt(dx^2 + dy^2) + 0.5)``), so objectives computed here are directly
    comparable with the ones the solver reports.
    """
    path = Path(path)
    xs, ys, costs = [], [], []
    for line in path.read_text().splitlines():
        parts = line.strip().split(";")
        if len(parts) != 3:
            continue
        try:
            x, y, c = (int(p) for p in parts)
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)
        costs.append(c)

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.floor(np.sqrt(dx * dx + dy * dy) + 0.5).astype(np.int64)

    return Instance(path.name, xs, ys, np.array(costs, dtype=np.int64), dist)


def tour_length(inst: Instance, tour: Sequence[int]) -> int:
    """Length of the closed tour visiting ``tour`` in the given order."""
    t = np.asarray(tour, dtype=np.int64)
    return int(inst.dist[t, np.roll(t, -1)].sum())


def nodes_cost(inst: Instance, tour: Sequence[int]) -> int:
    """Summed cost of the selected nodes."""
    return int(inst.costs[np.asarray(tour, dtype=np.int64)].sum())


def objective(inst: Instance, tour: Sequence[int]) -> int:
    """The quantity being minimised: tour length + cost of the selected nodes."""
    return tour_length(inst, tour) + nodes_cost(inst, tour)


def is_valid(inst: Instance, tour: Sequence[int]) -> bool:
    """True when ``tour`` selects exactly M distinct in-range nodes."""
    t = list(tour)
    return (
        len(t) == inst.m
        and len(set(t)) == inst.m
        and all(0 <= v < inst.n for v in t)
    )


# --------------------------------------------------------------------------
# Results loading (the JSON written by `go run . -json results.json`)
# --------------------------------------------------------------------------


def load_results(path: str | Path) -> list[dict]:
    """Read the solver's run export."""
    with open(path) as fh:
        return json.load(fh)


def summarize(runs: Iterable[Mapping]) -> dict[tuple[str, str], dict]:
    """Group runs into ``{(method, instance): stats}``.

    Stats carry min / max / average objective, the best run itself, and the
    total measured time — the numbers the report's table needs.
    """
    stats: dict[tuple[str, str], dict] = {}
    for r in runs:
        key = (r["strategy"], r["instance"])
        s = stats.setdefault(key, {
            "method": r["strategy"],
            "instance": r["instance"],
            "runs": 0,
            "objectives": [],
            "best": r,
            "elapsed_ms": 0.0,
        })
        s["runs"] += 1
        s["objectives"].append(r["objective"])
        s["elapsed_ms"] += r["elapsed_ms"]
        if r["objective"] < s["best"]["objective"]:
            s["best"] = r

    for s in stats.values():
        vals = s["objectives"]
        s["min"] = min(vals)
        s["max"] = max(vals)
        s["avg"] = sum(vals) / len(vals)
    return stats


def verify_runs(runs: Iterable[Mapping], instances: Mapping[str, Instance]) -> list[dict]:
    """Independently re-check every run against the raw instance CSV.

    This is the report's "solution checker": feasibility (exactly M distinct
    in-range nodes) plus a from-scratch recomputation of the objective, its two
    components, and the reported run count per method.
    """
    checks: list[dict] = []
    for r in runs:
        inst = instances[r["instance"]]
        tour = r["tour"]
        recomputed = objective(inst, tour)
        checks.append({
            "method": r["strategy"],
            "instance": r["instance"],
            "iteration": r["iteration"],
            "valid": is_valid(inst, tour),
            "objective_matches": recomputed == r["objective"],
            "components_match": (
                tour_length(inst, tour) == r["tour_len"]
                and nodes_cost(inst, tour) == r["node_cost"]
            ),
            "recomputed": recomputed,
            "reported": r["objective"],
        })
    return checks


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _draw_instance(
    ax,
    inst: Instance,
    tour: Sequence[int] | None,
    norm: Normalize,
    *,
    idle_size: float = 9.0,
    node_size: float = 34.0,
    line_width: float = 1.8,
) -> None:
    """Draw one instance, optionally with a solution on top of it.

    Unselected nodes stay recessive grey; selected nodes carry magnitude (cost)
    through the sequential ramp, with a surface-coloured ring so overlapping
    markers stay separable.
    """
    ax.scatter(
        inst.xs, inst.ys,
        s=idle_size, c=AXIS, linewidths=0, alpha=0.85, zorder=1,
    )

    if tour is None or len(tour) == 0:
        return

    t = np.asarray(tour, dtype=np.int64)
    closed = np.append(t, t[0])  # the tour is scored as a closed loop
    ax.plot(
        inst.xs[closed], inst.ys[closed],
        color=INK_SECONDARY, linewidth=line_width, solid_capstyle="round",
        alpha=0.9, zorder=2,
    )
    ax.scatter(
        inst.xs[t], inst.ys[t],
        s=node_size, c=inst.costs[t], cmap=COST_CMAP, norm=norm,
        edgecolors=SURFACE, linewidths=0.8, zorder=3,
    )


def _panel_title(ax, title: str, subtitle: str | None = None) -> None:
    """Left-aligned panel title with an optional muted second line."""
    ax.set_title(title, loc="left", pad=15 if subtitle else 6)
    if subtitle:
        ax.text(
            0, 1.015, subtitle, transform=ax.transAxes,
            fontsize=8.5, color=INK_MUTED, va="bottom", ha="left",
        )


def _clean_map_axes(ax, inst: Instance) -> None:
    # adjustable="datalim" keeps the axes box where the layout put it (so an
    # equal-aspect map cannot shove a neighbouring colour bar aside) and pads
    # the data range instead.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    # Margins rather than explicit limits: with a fixed data aspect, setting
    # both limits makes matplotlib drop one of them with a warning.
    ax.margins(0.04)


def _header(fig, fig_h: float, header_in: float, title, subtitle,
            tight: bool = True) -> None:
    """Reserve a header band for the figure title, then write it.

    The offsets are in inches, so the title can never collide with the panels
    however tall the figure ends up. Set ``tight=False`` when the caller has
    already placed the axes itself (tight_layout would move a colorbar).
    """
    if tight:
        fig.tight_layout(rect=(0, 0, 1, 1 - header_in / fig_h))
    if title:
        fig.text(
            0.012, 1 - 0.12 / fig_h, title,
            ha="left", va="top", fontsize=12.5, color=INK_PRIMARY,
        )
    if subtitle:
        fig.text(
            0.012, 1 - 0.34 / fig_h, subtitle,
            ha="left", va="top", fontsize=8.5, color=INK_MUTED,
        )


def plot_solution(ax, inst: Instance, tour: Sequence[int], title: str = "") -> None:
    """Draw a single solution on an existing axes."""
    norm = Normalize(vmin=inst.costs.min(), vmax=inst.costs.max())
    _draw_instance(ax, inst, tour, norm)
    _clean_map_axes(ax, inst)
    if title:
        _panel_title(ax, title, f"objective {objective(inst, tour):,}")


def plot_solutions(
    inst: Instance,
    solutions: Mapping[str, Sequence[int]],
    *,
    title: str | None = None,
    subtitle: str | None = None,
    ncols: int = 2,
    panel_size: float = 3.4,
) -> plt.Figure:
    """Small multiples: one node-cost map per method, sharing one colour scale.

    ``solutions`` maps a method name to its tour. Panels are labelled directly,
    so the reader never has to match a colour to a method.
    """
    methods = list(solutions)
    nrows = int(np.ceil(len(methods) / ncols))

    # Size each grid cell to the instance's own aspect ratio, so the
    # equal-aspect map fills its panel instead of leaving a band of dead space.
    aspect = np.ptp(inst.xs) / np.ptp(inst.ys)
    cell_h = float(np.clip(panel_size / aspect, 1.5, panel_size * 1.3))

    # The header must clear the suptitle and subtitle *and* the panel titles,
    # which sit above their axes on a 15pt pad.
    header = 0.85 if subtitle else 0.60
    cbar_gutter = 0.85  # inches reserved on the right for the colour bar
    fig_w = panel_size * ncols + cbar_gutter
    fig_h = cell_h * nrows + header
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Place the axes ourselves instead of using tight_layout: matplotlib's
    # tight layout ignores colour-bar axes, which would leave the bar sitting
    # on top of the right-hand panels.
    fig.subplots_adjust(
        left=0.015,
        right=1 - cbar_gutter / fig_w,
        bottom=0.015,
        top=1 - header / fig_h,
        wspace=0.06,
        hspace=0.16,
    )

    # One shared colour scale across the panels: costs are compared between
    # methods, so a per-panel scale would be misleading.
    norm = Normalize(vmin=inst.costs.min(), vmax=inst.costs.max())
    for ax, method in zip(axes.ravel(), methods):
        tour = solutions[method]
        _draw_instance(ax, inst, tour, norm)
        _clean_map_axes(ax, inst)
        _panel_title(ax, method, f"objective {objective(inst, tour):,}")

    for ax in axes.ravel()[len(methods):]:
        ax.set_visible(False)

    boxes = [ax.get_position() for ax in axes.ravel()[:len(methods)]]
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    cax = fig.add_axes([
        max(b.x1 for b in boxes) + 0.30 / fig_w,
        y0 + 0.20 * (y1 - y0),
        0.11 / fig_w,
        0.60 * (y1 - y0),
    ])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=COST_CMAP), cax=cax)
    bar.set_label("node cost", color=INK_SECONDARY, fontsize=8.5)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, color=INK_MUTED)

    _header(fig, fig_h, header, title, subtitle, tight=False)
    return fig


def plot_objective_ranges(
    instance_name: str,
    rows: Sequence[Mapping],
    *,
    zoom_methods: Sequence[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> plt.Figure:
    """Min/max/average objective per method, as horizontal ranges.

    ``rows`` is a sequence of ``{"method", "min", "max", "avg"}``. The left panel
    shows every method on a scale from zero; the right panel repeats the subset
    in ``zoom_methods`` on a truncated scale so their small spread is readable.
    Each row is named on the axis, so identity never rests on colour, and the
    average is the only value labelled.
    """
    panels = [("all methods", list(rows))]
    if zoom_methods:
        zoomed = [r for r in rows if r["method"] in set(zoom_methods)]
        if zoomed:
            panels.append(("greedy methods — axis zoomed, does not start at 0", zoomed))

    header = 0.62 if subtitle else 0.42
    fig_h = 0.62 * len(rows) + 1.5 + header
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.4 * len(panels), fig_h), squeeze=False,
    )

    for ax, (panel_title, panel_rows) in zip(axes[0], panels):
        labels = [r["method"] for r in panel_rows]
        ypos = np.arange(len(panel_rows))[::-1]  # first method on top

        for y, r in zip(ypos, panel_rows):
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
        ax.set_ylim(-0.7, len(panel_rows) - 0.1)
        ax.set_xlabel("objective (lower is better)", fontsize=8.5)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:,.0f}k")
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        if panel_rows:
            lo = min(r["min"] for r in panel_rows)
            hi = max(r["max"] for r in panel_rows)
            if panel_title.startswith("all"):
                ax.set_xlim(0, hi * 1.06)
            else:
                span = max(hi - lo, 1)
                ax.set_xlim(lo - span * 0.08, hi + span * 0.30)

        ax.set_title(panel_title, loc="left", pad=10, fontsize=10.5)

    _header(fig, fig_h, header, title, subtitle)
    return fig


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
