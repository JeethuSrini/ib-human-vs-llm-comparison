"""Generate rate-distortion plots for the IB human-vs-LLM study.

Three figures:
  1. memory_rate_distortion.png    — distortion vs N, one line per system, memory task
  2. reasoning_rate_distortion.png — same for reasoning task
  3. memory_vs_reasoning.png       — scatter: memory distortion vs reasoning distortion
                                     per system per N, showing the compression gap
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── visual style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# Model families → colour palette
_FAMILY_COLORS = {
    "gemma":   "#4285F4",   # Google blue
    "llama":   "#FF6D00",   # Meta orange
    "qwen":    "#9C27B0",   # purple
    "mistral": "#00897B",   # teal
    "human":   "#E53935",   # red
}

_LINE_STYLES = {
    "4b":  "solid",
    "7b":  "solid",
    "1b":  "dotted",
    "3b":  "dashed",
    "8b":  "solid",
    "12b": "dashed",
    "14b": "dashed",
    "27b": "dashdot",
    "72b": "dashdot",
    "11b": "dashdot",
    "70b": "dotted",
}

_MARKERS = {
    "model": "o",
    "human": "*",
}


def _short_label(model: str, source: str) -> str:
    if source == "human":
        return "Humans (baseline)"
    parts = model.split("/")[-1]
    for suffix in ["-it", "-instruct", "-vision-instruct", "-2512"]:
        parts = parts.replace(suffix, "")
    return parts


def _color_and_style(model: str, source: str) -> tuple[str, str]:
    m = model.lower()
    if source == "human":
        return _FAMILY_COLORS["human"], "solid"
    for fam, color in _FAMILY_COLORS.items():
        if fam in m:
            # pick line style by size token
            for size_tok, ls in _LINE_STYLES.items():
                if size_tok in m:
                    return color, ls
            return color, "solid"
    return "#607D8B", "solid"  # fallback grey


def _group_by_system(ib_points: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for pt in ib_points:
        key = (pt["source"], pt["model"], pt["mode"])
        groups[key].append(pt)
    return {k: sorted(v, key=lambda x: x["n_examples"]) for k, v in groups.items()}


def plot_rate_distortion(
    ib_points: list[dict],
    output_dir: Path,
    mode_filter: str = "separate",
) -> None:
    """Two-panel figure: memory distortion (left) and reasoning distortion (right)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _group_by_system(ib_points)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Empirical Rate–Distortion Curves\n"
        "(N = number of facts in prompt; Distortion = 1 − Accuracy)",
        fontsize=13, y=1.01,
    )

    for task_idx, (task, dist_key, dist_ci_low, dist_ci_high) in enumerate([
        ("Memory",    "memory_distortion",    "memory_dist_ci_low",    "memory_dist_ci_high"),
        ("Reasoning", "reasoning_distortion", "reasoning_dist_ci_low", "reasoning_dist_ci_high"),
    ]):
        ax = axes[task_idx]
        legend_handles = []

        for (source, model, mode), pts in groups.items():
            if mode != mode_filter and source != "human":
                continue
            xs = [pt["n_examples"] for pt in pts]
            ys = [pt[dist_key] for pt in pts]
            ci_lo = [pt[dist_ci_low]  for pt in pts]
            ci_hi = [pt[dist_ci_high] for pt in pts]
            color, ls = _color_and_style(model, source)
            marker = "*" if source == "human" else "o"
            ms = 18 if source == "human" else 6
            lw = 3.0 if source == "human" else 1.8
            label = _short_label(model, source)

            line, = ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw,
                            marker=marker, markersize=ms, label=label, zorder=3)
            ax.fill_between(xs, ci_lo, ci_hi, color=color, alpha=0.12)
            legend_handles.append(line)

        ax.set_xlabel("N (number of facts in prompt)", fontsize=11)
        ax.set_ylabel("Distortion  (1 − Accuracy)", fontsize=11)
        ax.set_title(f"{task} Task", fontsize=12, fontweight="bold")
        ax.set_xticks([3, 8, 15])
        ax.set_ylim(-0.02, 1.05)
        ax.legend(handles=legend_handles, fontsize=8, ncol=2,
                  loc="upper left", framealpha=0.8)

    plt.tight_layout()
    out_path = output_dir / "rate_distortion.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_memory_vs_reasoning(
    ib_points: list[dict],
    output_dir: Path,
    mode_filter: str = "separate",
) -> None:
    """Scatter: memory distortion vs reasoning distortion per (system, N).

    Points higher up = worse reasoning; points right = worse memory.
    The IB hypothesis: humans cluster bottom-right (good reasoning, worse memory)
    while LLMs cluster top-left (good memory, worse reasoning).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _group_by_system(ib_points)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(
        "Memory vs Reasoning Distortion\n(each point = one system at one N)",
        fontsize=12,
    )

    n_markers = {3: "o", 8: "s", 15: "^"}
    n_sizes   = {3: 80,  8: 110, 15: 150}
    legend_handles = []
    seen_labels: set[str] = set()

    for (source, model, mode), pts in groups.items():
        if mode != mode_filter and source != "human":
            continue
        color, _ = _color_and_style(model, source)
        label = _short_label(model, source)

        for pt in pts:
            n  = pt["n_examples"]
            mx = pt["memory_distortion"]
            ry = pt["reasoning_distortion"]
            marker = n_markers.get(n, "o")
            size   = n_sizes.get(n, 80)
            ax.scatter(mx, ry, color=color, marker=marker, s=size,
                       zorder=3, edgecolors="white", linewidths=0.5,
                       alpha=0.85)
            fs = 8 if source == "human" else 7
            fw = "bold" if source == "human" else "normal"
            ax.annotate(f"{label}\nN={n}", (mx, ry),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=fs, color=color, alpha=0.9,
                        fontweight=fw)

        if label not in seen_labels:
            seen_labels.add(label)
            legend_handles.append(
                mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                              markersize=8, label=label)
            )

    # diagonal guide: equal distortion
    lims = [0, 1]
    ax.plot(lims, lims, "k--", alpha=0.2, linewidth=1, label="equal distortion")

    ax.set_xlabel("Memory Distortion  (1 − Memory Accuracy)", fontsize=11)
    ax.set_ylabel("Reasoning Distortion  (1 − Reasoning Accuracy)", fontsize=11)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    # N legend
    n_legend = [
        mlines.Line2D([], [], color="grey", marker=n_markers[n], linestyle="None",
                      markersize=8, label=f"N={n}")
        for n in [3, 8, 15]
    ]
    legend1 = ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
                        framealpha=0.8, title="System")
    ax.add_artist(legend1)
    ax.legend(handles=n_legend, fontsize=8, loc="upper left",
              framealpha=0.8, title="Prompt size")

    plt.tight_layout()
    out_path = output_dir / "memory_vs_reasoning.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_scaling(
    ib_points: list[dict],
    output_dir: Path,
    mode_filter: str = "separate",
) -> None:
    """Accuracy vs model size within each family — scaling behaviour."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map model id to rough parameter count (billions)
    _SIZE_MAP = {
        "llama-3.2-1b":  1,  "llama-3.2-3b":  3,
        "llama-3.2-11b": 11, "llama-3.3-70b": 70,
        "gemma-3-4b":  4,  "gemma-3-12b": 12, "gemma-3-27b": 27,
        "qwen-2.5-7b": 7,  "qwen-2.5-72b": 72,
        "ministral-14b": 14,
    }

    def _size(model: str) -> float | None:
        m = model.split("/")[-1].lower()
        for k, v in _SIZE_MAP.items():
            if k in m:
                return v
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Accuracy vs Model Size  (within family)", fontsize=13)

    for task_idx, (task, acc_key) in enumerate([
        ("Memory",    "memory_accuracy"),
        ("Reasoning", "reasoning_accuracy"),
    ]):
        ax = axes[task_idx]
        for (source, model, mode), pts in _group_by_system(ib_points).items():
            if source == "human" or (mode != mode_filter):
                continue
            sz = _size(model)
            if sz is None:
                continue
            color, _ = _color_and_style(model, source)
            for pt in pts:
                n = pt["n_examples"]
                ax.scatter(sz, pt[acc_key], color=color,
                           s=60 + n * 4, alpha=0.75,
                           marker={3:"o",8:"s",15:"^"}.get(n,"o"),
                           edgecolors="white", linewidths=0.4)
        ax.set_xlabel("Model size (B parameters)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{task} Task", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xscale("log")

    plt.tight_layout()
    out_path = output_dir / "scaling.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── family ordering for subplots ─────────────────────────────────────────────
_FAMILY_ORDER = ["gemma", "llama", "qwen", "mistral"]
_FAMILY_LABELS = {
    "gemma":   "Gemma 3  (Google)",
    "llama":   "Llama 3.x  (Meta)",
    "qwen":    "Qwen 2.5  (Alibaba)",
    "mistral": "Mistral  (Mistral AI)",
}


def _detect_family(model: str) -> str | None:
    m = model.lower()
    for fam in _FAMILY_ORDER:
        if fam in m:
            return fam
    return None


def plot_per_family(
    ib_points: list[dict],
    output_dir: Path,
    mode_filter: str = "separate",
) -> None:
    """4-panel figure: one subplot per model family.

    Each panel shows memory and reasoning distortion curves for all models
    in that family, plus the human baseline for reference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect human baseline points once
    human_pts = {
        pt["n_examples"]: pt
        for pt in ib_points
        if pt["source"] == "human"
    }

    # Group model points by family
    family_data: dict[str, dict[tuple, list[dict]]] = {f: {} for f in _FAMILY_ORDER}
    for (source, model, mode), pts in _group_by_system(ib_points).items():
        if source == "human":
            continue
        if mode != mode_filter:
            continue
        fam = _detect_family(model)
        if fam is None:
            continue
        family_data[fam][(source, model, mode)] = pts

    xs = [3, 8, 15]
    saved_paths = []

    for fam in _FAMILY_ORDER:
        if not family_data[fam]:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(
            f"{_FAMILY_LABELS.get(fam, fam)}\nRate–Distortion Curve"
            f"\n(solid = memory, dashed = reasoning  |  ★ = human baseline)",
            fontsize=12, y=1.02,
        )
        legend_handles = []

        # ── human baseline ────────────────────────────────────────────────────
        h_color = _FAMILY_COLORS["human"]
        xh       = [n for n in xs if n in human_pts]
        h_mem    = [human_pts[n]["memory_distortion"]      for n in xh]
        h_rea    = [human_pts[n]["reasoning_distortion"]   for n in xh]
        h_mem_lo = [human_pts[n]["memory_dist_ci_low"]     for n in xh]
        h_mem_hi = [human_pts[n]["memory_dist_ci_high"]    for n in xh]
        h_rea_lo = [human_pts[n]["reasoning_dist_ci_low"]  for n in xh]
        h_rea_hi = [human_pts[n]["reasoning_dist_ci_high"] for n in xh]

        ax.plot(xh, h_mem, color=h_color, linestyle="solid",  linewidth=2.8,
                marker="*", markersize=16, zorder=5)
        ax.plot(xh, h_rea, color=h_color, linestyle="dashed", linewidth=2.8,
                marker="*", markersize=16, zorder=5)
        ax.fill_between(xh, h_mem_lo, h_mem_hi, color=h_color, alpha=0.12)
        ax.fill_between(xh, h_rea_lo, h_rea_hi, color=h_color, alpha=0.12)
        legend_handles += [
            mlines.Line2D([], [], color=h_color, linestyle="solid",  marker="*",
                          markersize=12, linewidth=2.5, label="Humans — memory"),
            mlines.Line2D([], [], color=h_color, linestyle="dashed", marker="*",
                          markersize=12, linewidth=2.5, label="Humans — reasoning"),
        ]

        # ── model lines (each model = distinct shade within family colour) ────
        models_in_fam = sorted(family_data[fam].items())
        n_models = len(models_in_fam)
        base_color = _FAMILY_COLORS.get(fam, "#607D8B")
        # generate shades: darkest → lightest within the family hue
        import colorsys
        r, g, b = int(base_color[1:3],16)/255, int(base_color[3:5],16)/255, int(base_color[5:7],16)/255
        h_hue, s_sat, v_val = colorsys.rgb_to_hsv(r, g, b)
        shades = [
            "#{:02x}{:02x}{:02x}".format(*[int(c*255) for c in colorsys.hsv_to_rgb(
                h_hue,
                max(0.3, s_sat - 0.15 * i),
                min(1.0, v_val + 0.12 * i),
            )])
            for i in range(n_models)
        ]

        for idx, ((source, model, mode), pts) in enumerate(models_in_fam):
            color = shades[idx]
            label = _short_label(model, source)
            pt_map   = {p["n_examples"]: p for p in pts}
            xm       = [n for n in xs if n in pt_map]
            ys_mem   = [pt_map[n]["memory_distortion"]      for n in xm]
            ys_rea   = [pt_map[n]["reasoning_distortion"]   for n in xm]
            ci_mem_lo = [pt_map[n]["memory_dist_ci_low"]    for n in xm]
            ci_mem_hi = [pt_map[n]["memory_dist_ci_high"]   for n in xm]
            ci_rea_lo = [pt_map[n]["reasoning_dist_ci_low"] for n in xm]
            ci_rea_hi = [pt_map[n]["reasoning_dist_ci_high"]for n in xm]

            ax.plot(xm, ys_mem, color=color, linestyle="solid",  linewidth=2.0,
                    marker="o", markersize=7, label=f"{label} — memory")
            ax.plot(xm, ys_rea, color=color, linestyle="dashed", linewidth=2.0,
                    marker="o", markersize=7, label=f"{label} — reasoning")
            ax.fill_between(xm, ci_mem_lo, ci_mem_hi, color=color, alpha=0.10)
            ax.fill_between(xm, ci_rea_lo, ci_rea_hi, color=color, alpha=0.10)
            legend_handles += [
                mlines.Line2D([], [], color=color, linestyle="solid",  marker="o",
                              markersize=7, linewidth=2, label=f"{label} — memory"),
                mlines.Line2D([], [], color=color, linestyle="dashed", marker="o",
                              markersize=7, linewidth=2, label=f"{label} — reasoning"),
            ]

        ax.set_xlim(2, 16)
        ax.set_xticks(xs)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("N (number of facts in prompt)", fontsize=11)
        ax.set_ylabel("Distortion  (1 − Accuracy)", fontsize=11)
        ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9,
                  loc="upper left", ncol=1)

        plt.tight_layout()
        out_path = output_dir / f"family_{fam}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_path}")
        saved_paths.append(out_path)

    return saved_paths
