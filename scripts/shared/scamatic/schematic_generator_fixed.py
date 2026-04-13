from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


MetricType = Literal[
    "PC", "PC_pos", "PC_neg",
    "Z", "Z_pos", "Z_neg",
    "Strength_pos", "Strength_neg",
]
DirectionType = Literal["increase", "decrease"]


# style
BG = "white"
TEXT = "#222222"
LIGHT_TEXT = "#555555"
PANEL_BG = "#fafafa"
MODULE_FILL = "#ececec"
MODULE_EDGE = "#888888"
FOCAL_FILL = "#ffe3bf"
FOCAL_EDGE = "#cc7a00"

POS_COLOR = "#d95f02"
NEG_COLOR = "#1f78b4"

TITLE_SIZE = 18
SUBTITLE_SIZE = 11
PANEL_TITLE_SIZE = 13
BODY_SIZE = 11
SMALL_SIZE = 9
NODE_RADIUS = 0.34


# module layout / names
CANONICAL_MODULE_NAMES = {
    "M1": "M1 Somatomotor",
    "M2": "M2 Visual-A",
    "M3": "M3 Limbic",
    "M4": "M4 Frontoparietal",
    "M5": "M5 Ventral Attention",
    "M6": "M6 Visual-B",
    "M7": "M7 Default Mode",
    "M8": "M8 Dorsal Attention",
}

MODULE_POSITIONS = {
    "M1": (1.55, -0.10),
    "M2": (0.85, 1.45),
    "M3": (-0.70, 1.80),
    "M4": (-1.95, 0.75),
    "M5": (-1.95, -0.85),
    "M6": (-0.70, -1.85),
    "M7": (0.85, -1.45),
    "M8": (2.45, 1.10),
}

DEFAULT_LABEL_OFFSETS = {
    "M1": (0.00, -0.62, "center", "top"),
    "M2": (0.00, -0.62, "center", "top"),
    "M3": (0.00, -0.60, "center", "top"),
    "M4": (0.00, -0.68, "center", "top"),
    "M5": (0.00, -0.68, "center", "top"),
    "M6": (0.00, -0.68, "center", "top"),
    "M7": (0.00, -0.68, "center", "top"),
    "M8": (0.00, -0.62, "center", "top"),
}

FOCAL_LABEL_OFFSETS = {
    "M1": (0.00, -0.72, "center", "top"),
    "M2": (0.00, -0.72, "center", "top"),
    "M3": (0.00, -0.70, "center", "top"),
    "M4": (0.00, -0.72, "center", "top"),
    "M5": (0.00, -0.72, "center", "top"),
    "M6": (0.00, -0.72, "center", "top"),
    "M7": (0.00, -0.72, "center", "top"),
    "M8": (0.00, -0.70, "center", "top"),
}

# link ordering for PC/Strength drawings so the pattern stays visually stable.
LINK_PRIORITY = ["M2", "M3", "M4", "M5", "M6", "M7", "M8", "M1"]


# text / normalization helpers
def wrap_text(text: Optional[str], width: int) -> Optional[str]:
    if not text:
        return None
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def prettify_free_text(text: str) -> str:
    text = normalize_spaces(text)
    replacements = {
        "DefaultMode": "Default Mode",
        "DorsalAttention": "Dorsal Attention",
        "VentralAttention": "Ventral Attention",
        "Visual-A": "Visual-A",
        "Visual-B": "Visual-B",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def parse_module_id(module_label: str) -> Optional[str]:
    match = re.search(r"\bM([1-8])\b", module_label)
    if match:
        return f"M{match.group(1)}"
    return None


def canonical_module_label(module_label: str) -> str:
    module_label = prettify_free_text(module_label)
    module_id = parse_module_id(module_label)
    if module_id and module_id in CANONICAL_MODULE_NAMES:
        return CANONICAL_MODULE_NAMES[module_id]
    return module_label


# metric helpers
def metric_family(metric: str) -> str:
    if metric.startswith("PC"):
        return "PC"
    if metric.startswith("Z"):
        return "Z"
    if metric.startswith("Strength"):
        return "Strength"
    raise ValueError(f"Unsupported metric: {metric}")


def metric_sign(metric: str) -> str:
    if metric.endswith("_pos"):
        return "pos"
    if metric.endswith("_neg"):
        return "neg"
    return "all"


def metric_line_style(metric: str):
    sign = metric_sign(metric)
    if sign == "neg":
        return NEG_COLOR, "--"
    return POS_COLOR, "-"


def pretty_metric_name(metric: str) -> str:
    mapping = {
        "PC": "Participation coefficient",
        "PC_pos": "Positive participation coefficient",
        "PC_neg": "Negative participation coefficient",
        "Z": "Within-module degree z-score",
        "Z_pos": "Positive within-module degree z-score",
        "Z_neg": "Negative within-module degree z-score",
        "Strength_pos": "Positive strength",
        "Strength_neg": "Negative strength",
    }
    return mapping.get(metric, metric)


def interpret_metric(metric: str) -> str:
    fam = metric_family(metric)
    sign = metric_sign(metric)

    if fam == "PC":
        base = "cross-module participation"
    elif fam == "Z":
        base = "within-module centrality"
    else:
        base = "total connection strength"

    if sign == "pos":
        base = "positive " + base
    elif sign == "neg":
        base = "negative " + base

    return base


def normalize_direction(direction: str, fallback: str = "increase") -> str:
    d = normalize_spaces(str(direction)).lower()
    if d == "increase":
        return "increase"
    if d == "decrease":
        return "decrease"
    if "increase" in d or "higher" in d:
        return "increase"
    if "decrease" in d or "lower" in d:
        return "decrease"
    if "weaker" in d or "absent" in d or "flat" in d:
        return "flat"
    if "sensitivity" in d:
        return fallback
    if "opposite" in d or "unique" in d:
        return fallback
    return fallback


def direction_text(metric: str, direction: str) -> str:
    direction = normalize_direction(direction)
    base = interpret_metric(metric)
    if direction == "increase":
        return f"{base} higher in ASD"
    if direction == "decrease":
        return f"{base} lower in ASD"
    return f"{base} changed in ASD"


def direction_from_d(d: float) -> DirectionType:
    # d is CTL - ASD
    # positive d => CTL > ASD => ASD lower => decrease
    # negative d => ASD > CTL => increase
    return "decrease" if d >= 0 else "increase"


# drawing primitives
def draw_panel_box(ax):
    ax.add_patch(
        Rectangle(
            (-3.0, -2.8),
            6.3,
            6.0,
            facecolor=PANEL_BG,
            edgecolor="#dddddd",
            linewidth=1.0,
            zorder=0,
        )
    )


def label_position_for_node(module_id: str, x: float, y: float, is_focal: bool = False) -> tuple[float, float, str, str]:
    offsets = FOCAL_LABEL_OFFSETS if is_focal else DEFAULT_LABEL_OFFSETS
    dx, dy, ha, va = offsets.get(module_id, (0.00, -0.62, "center", "top"))
    return x + dx, y + dy, ha, va


def label_style(is_focal: bool = False) -> dict:
    return {
        "fontsize": SMALL_SIZE,
        "color": LIGHT_TEXT,
        "zorder": 10,
        "fontweight": "medium" if is_focal else "normal",
        "bbox": {
            "boxstyle": "round,pad=0.14",
            "facecolor": PANEL_BG,
            "edgecolor": "none",
            "alpha": 0.96 if is_focal else 0.90,
        },
    }


def draw_modules(ax, focal_module_id: str, focal_module_label: str) -> dict[str, tuple[float, float]]:
    positions = dict(MODULE_POSITIONS)

    for module_id, (x, y) in positions.items():
        is_focal = module_id == focal_module_id
        fill = FOCAL_FILL if is_focal else MODULE_FILL
        edge = FOCAL_EDGE if is_focal else MODULE_EDGE
        lw = 1.6 if is_focal else 1.0

        ax.add_patch(
            Circle(
                (x, y),
                NODE_RADIUS,
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
                zorder=3,
            )
        )

        label = focal_module_label if is_focal else CANONICAL_MODULE_NAMES[module_id]
        tx, ty, ha, va = label_position_for_node(module_id, x, y, is_focal=is_focal)
        ax.text(tx, ty, label, ha=ha, va=va, **label_style(is_focal=is_focal))

    return positions


def rotate_point(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return x * c - y * s, x * s + y * c


def draw_internal_links(ax, focal_center, strength_level: float, color: str, linestyle: str):
    cx, cy = focal_center

    angle = math.radians(12)
    inner_nodes = []
    outer_nodes = []

    for deg in (0, 120, 240):
        x, y = rotate_point(
            0.18 * math.cos(math.radians(deg)),
            0.18 * math.sin(math.radians(deg)),
            angle,
        )
        inner_nodes.append((cx + x, cy + y))

    for deg in (20, 95, 190, 300):
        x, y = rotate_point(
            0.46 * math.cos(math.radians(deg)),
            0.46 * math.sin(math.radians(deg)),
            angle,
        )
        outer_nodes.append((cx + x, cy + y))

    lw = 1.0 + 3.2 * strength_level

    inner_pairs = [(0, 1), (1, 2), (2, 0)]
    outer_pairs = [(0, 0), (0, 1), (1, 2), (1, 3), (2, 0), (2, 3)]

    n_inner = 1 if strength_level < 0.35 else 2 if strength_level < 0.70 else 3
    n_outer = 2 if strength_level < 0.35 else 4 if strength_level < 0.70 else 6

    for i, j in inner_pairs[:n_inner]:
        xi, yi = inner_nodes[i]
        xj, yj = inner_nodes[j]
        ax.add_line(
            Line2D(
                [xi, xj],
                [yi, yj],
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=0.82,
                zorder=4,
            )
        )

    for i, j in outer_pairs[:n_outer]:
        xi, yi = inner_nodes[i]
        xj, yj = outer_nodes[j]
        ax.add_line(
            Line2D(
                [xi, xj],
                [yi, yj],
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=0.82,
                zorder=4,
            )
        )

    for x, y in inner_nodes + outer_nodes:
        ax.add_patch(
            Circle(
                (x, y),
                0.11,
                facecolor="white",
                edgecolor="#333333",
                linewidth=0.9,
                zorder=5,
            )
        )


def ordered_other_modules(focal_module_id: str) -> list[str]:
    return [m for m in LINK_PRIORITY if m != focal_module_id]


def clipped_segment(x0: float, y0: float, x1: float, y1: float, pad: float = NODE_RADIUS * 0.92):
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist == 0:
        return x0, y0, x1, y1
    ux = dx / dist
    uy = dy / dist
    return x0 + ux * pad, y0 + uy * pad, x1 - ux * pad, y1 - uy * pad


def draw_cross_module_links(
    ax,
    focal_module_id: str,
    positions: dict[str, tuple[float, float]],
    strength_level: float,
    color: str,
    linestyle: str,
    strength_mode: bool = False,
):
    fx, fy = positions[focal_module_id]
    ordered_ids = ordered_other_modules(focal_module_id)

    if strength_mode:
        chosen_ids = ordered_ids
        base_lw = 1.0 + 4.0 * strength_level
    else:
        n_links = max(1, round(strength_level * len(ordered_ids)))
        chosen_ids = ordered_ids[:n_links]
        base_lw = 1.2 + 2.0 * strength_level

    for idx, module_id in enumerate(chosen_ids):
        x, y = positions[module_id]
        x0, y0, x1, y1 = clipped_segment(fx, fy, x, y)
        lw = max(0.8, base_lw - 0.12 * idx)
        ax.add_line(
            Line2D(
                [x0, x1],
                [y0, y1],
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=0.82,
                zorder=2,
            )
        )


# left here unchanged, though now unused
def draw_strength_links(
    ax,
    focal_module_id: str,
    positions: dict[str, tuple[float, float]],
    strength_level: float,
    color: str,
    linestyle: str,
):
    """
    Strength should read as total incident connectivity, not pure participation.
    So we draw:
      - a local / within-module component
      - plus outward incident edges to the rest of the network
    """

    internal_level = min(1.0, 0.35 + 0.75 * strength_level)
    draw_internal_links(
        ax,
        positions[focal_module_id],
        internal_level,
        color,
        linestyle,
    )

    fx, fy = positions[focal_module_id]
    ordered_ids = ordered_other_modules(focal_module_id)

    edge_multipliers = [1.00, 0.95, 0.92, 0.88, 0.84, 0.80, 0.76]
    base_lw = 0.75 + 2.6 * strength_level
    alpha = 0.50 + 0.22 * strength_level

    for idx, module_id in enumerate(ordered_ids):
        x, y = positions[module_id]
        x0, y0, x1, y1 = clipped_segment(fx, fy, x, y)
        lw = max(0.7, base_lw * edge_multipliers[idx])

        ax.add_line(
            Line2D(
                [x0, x1],
                [y0, y1],
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=alpha,
                zorder=2,
            )
        )


def draw_strength_halo(ax, focal_center, strength_level, color, linestyle):
    """
    Strength should look local/module-centered, not like between-module participation.
    This draws one or two dashed halos around the focal module.
    """
    cx, cy = focal_center

    inner_r = NODE_RADIUS + 0.10
    outer_r = NODE_RADIUS + 0.22

    lw_inner = 0.9 + 2.0 * strength_level
    lw_outer = 0.7 + 1.5 * strength_level

    alpha_inner = 0.45 + 0.25 * strength_level
    alpha_outer = 0.30 + 0.20 * strength_level

    ax.add_patch(
        Circle(
            (cx, cy),
            inner_r,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=lw_inner,
            alpha=alpha_inner,
            zorder=4,
        )
    )

    if strength_level >= 0.45:
        ax.add_patch(
            Circle(
                (cx, cy),
                outer_r,
                fill=False,
                edgecolor=color,
                linestyle=linestyle,
                linewidth=lw_outer,
                alpha=alpha_outer,
                zorder=4,
            )
        )


def draw_strength_stubs(ax, focal_center, strength_level, color, linestyle):
    """
    Short local stubs just outside the focal module.
    These suggest increased total incident connectivity without implying
    a specific between-module / participation-style routing pattern.
    """
    cx, cy = focal_center

    angles = np.deg2rad([20, 75, 140, 210, 275, 330])

    n_stubs = 2 if strength_level < 0.35 else 4 if strength_level < 0.7 else 6
    stub_angles = angles[:n_stubs]

    r0 = NODE_RADIUS + 0.05
    r1 = NODE_RADIUS + 0.38 + 0.10 * strength_level
    lw = 0.9 + 1.8 * strength_level
    alpha = 0.40 + 0.20 * strength_level

    for ang in stub_angles:
        x0 = cx + r0 * np.cos(ang)
        y0 = cy + r0 * np.sin(ang)
        x1 = cx + r1 * np.cos(ang)
        y1 = cy + r1 * np.sin(ang)

        ax.add_line(
            Line2D(
                [x0, x1],
                [y0, y1],
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=alpha,
                zorder=3,
            )
        )


def draw_strength_links_local(ax, focal_center, strength_level, color, linestyle):
    """
    Strength = total incident connectivity.
    Show this as:
      - stronger internal/local organization
      - local halo around the focal module
      - short local stubs
    NOT as long between-module spokes.
    """
    internal_level = min(1.0, 0.35 + 0.75 * strength_level)

    draw_internal_links(ax, focal_center, internal_level, color, linestyle)
    draw_strength_halo(ax, focal_center, strength_level, color, linestyle)
    draw_strength_stubs(ax, focal_center, strength_level, color, linestyle)


def strength_level_for_panel(direction: str, asd: bool) -> float:
    direction = normalize_direction(direction)
    if direction == "increase":
        return 0.30 if not asd else 0.88
    if direction == "decrease":
        return 0.88 if not asd else 0.30
    return 0.55


def draw_metric_scene(
    ax,
    metric: MetricType,
    direction: DirectionType,
    panel_title: str,
    module_label: str,
):
    draw_panel_box(ax)

    focal_module_label = canonical_module_label(module_label)
    focal_module_id = parse_module_id(focal_module_label)
    if focal_module_id is None:
        raise ValueError(f"Could not parse module id from module_label={module_label!r}")

    positions = draw_modules(
        ax,
        focal_module_id=focal_module_id,
        focal_module_label=focal_module_label,
    )

    color, linestyle = metric_line_style(metric)
    fam = metric_family(metric)
    asd = "ASD" in panel_title.upper()
    level = strength_level_for_panel(direction, asd)

    if fam == "Z":
        draw_internal_links(ax, positions[focal_module_id], level, color, linestyle)

    elif fam == "PC":
        draw_cross_module_links(
            ax,
            focal_module_id,
            positions,
            level,
            color,
            linestyle,
            strength_mode=False,
        )

    elif fam == "Strength":
        draw_strength_links_local(
            ax,
            positions[focal_module_id],
            level,
            color,
            linestyle,
        )

    ax.text(
        0.15,
        2.72,
        panel_title,
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        weight="bold",
        color=TEXT,
    )
    ax.set_xlim(-3.1, 3.4)
    ax.set_ylim(-2.95, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")


# figure helpers
def add_center_arrow(fig, x0, y0, x1, y1):
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=1.8,
        color="#555555",
    )
    fig.patches.append(arrow)


def legend_handles(metric: str):
    color, linestyle = metric_line_style(metric)
    line = Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.5)
    focal = Line2D(
        [0],
        [0],
        marker="o",
        markersize=10,
        markerfacecolor=FOCAL_FILL,
        markeredgecolor=FOCAL_EDGE,
        linestyle="None",
    )
    return [line, focal], [pretty_metric_name(metric), "Focal module"]


# scematic builders
def make_dx_only_schematic(
    metric: str,
    direction: str,
    module_label: str,
    group_label: str,
    title: str,
    out_path: str | Path,
    footer_note: Optional[str] = None,
    stats_note: Optional[str] = None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    group_label = prettify_free_text(group_label)
    title = prettify_free_text(title)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.6), facecolor=BG)

    draw_metric_scene(
        axes[0],
        metric=metric,
        direction=direction,
        panel_title=f"{group_label} CTL",
        module_label=module_label,
    )
    draw_metric_scene(
        axes[1],
        metric=metric,
        direction=direction,
        panel_title=f"{group_label} ASD",
        module_label=module_label,
    )

    fig.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="center",
        fontsize=TITLE_SIZE,
        weight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.925,
        f"DX-only schematic | {pretty_metric_name(metric)}",
        ha="center",
        va="center",
        fontsize=SUBTITLE_SIZE,
        color=LIGHT_TEXT,
    )

    add_center_arrow(fig, 0.46, 0.50, 0.54, 0.50)

    fig.text(
        0.5,
        0.145,
        f"Control \u2192 Autism: {direction_text(metric, direction)}",
        ha="center",
        va="center",
        fontsize=BODY_SIZE,
        color=TEXT,
    )

    wrapped_stats = wrap_text(stats_note, 95)
    wrapped_footer = wrap_text(footer_note, 95)

    if wrapped_stats:
        fig.text(
            0.5,
            0.105,
            wrapped_stats,
            ha="center",
            va="center",
            fontsize=SMALL_SIZE,
            color=LIGHT_TEXT,
        )

    if wrapped_footer:
        fig.text(
            0.5,
            0.065,
            wrapped_footer,
            ha="center",
            va="center",
            fontsize=SMALL_SIZE,
            color=LIGHT_TEXT,
        )

    handles, labels = legend_handles(metric)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )

    fig.subplots_adjust(left=0.035, right=0.965, top=0.87, bottom=0.23, wspace=0.18)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_dxsex_schematic(
    metric: str,
    male_direction: str,
    female_direction: str,
    module_label: str,
    age_label: str,
    title: str,
    out_path: str | Path,
    footer_note: Optional[str] = None,
    stats_note: Optional[str] = None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    age_label = prettify_free_text(age_label)
    title = prettify_free_text(title)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.8), facecolor=BG)

    draw_metric_scene(axes[0, 0], metric, male_direction, "Male CTL", module_label)
    draw_metric_scene(axes[0, 1], metric, male_direction, "Male ASD", module_label)
    draw_metric_scene(axes[1, 0], metric, female_direction, "Female CTL", module_label)
    draw_metric_scene(axes[1, 1], metric, female_direction, "Female ASD", module_label)

    fig.text(
        0.5,
        0.975,
        title,
        ha="center",
        va="center",
        fontsize=TITLE_SIZE,
        weight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.94,
        f"DX×SEX schematic | {age_label} | {pretty_metric_name(metric)}",
        ha="center",
        va="center",
        fontsize=SUBTITLE_SIZE,
        color=LIGHT_TEXT,
    )

    add_center_arrow(fig, 0.46, 0.695, 0.54, 0.695)
    add_center_arrow(fig, 0.46, 0.315, 0.54, 0.315)

    male_text = f"Male: {direction_text(metric, male_direction)}"
    female_text = f"Female: {direction_text(metric, female_direction)}"

    fig.text(0.5, 0.148, male_text, ha="center", va="center", fontsize=BODY_SIZE, color=TEXT)
    fig.text(0.5, 0.120, female_text, ha="center", va="center", fontsize=BODY_SIZE, color=TEXT)

    male_dir_norm = normalize_direction(male_direction, fallback="increase")
    female_dir_norm = normalize_direction(female_direction, fallback="increase")
    if male_dir_norm != female_dir_norm:
        interaction_note = "Opposite-direction diagnosis effects by sex"
    else:
        interaction_note = "Same-direction diagnosis effects with different magnitude by sex"

    fig.text(
        0.5,
        0.091,
        interaction_note,
        ha="center",
        va="center",
        fontsize=BODY_SIZE,
        weight="bold",
        color=TEXT,
    )

    wrapped_stats = wrap_text(stats_note, 95)
    wrapped_footer = wrap_text(footer_note, 95)

    if wrapped_stats:
        fig.text(0.5, 0.055, wrapped_stats, ha="center", va="center", fontsize=SMALL_SIZE, color=LIGHT_TEXT)

    if wrapped_footer:
        fig.text(0.5, 0.028, wrapped_footer, ha="center", va="center", fontsize=SMALL_SIZE, color=LIGHT_TEXT)

    handles, labels = legend_handles(metric)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=2,
        frameon=False,
    )

    fig.subplots_adjust(left=0.035, right=0.965, top=0.90, bottom=0.18, hspace=0.18, wspace=0.18)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


FINDINGS = [
    # ==================== DX-only core families ====================

    {
        "kind": "dx_only",
        "metric": "Z",
        "direction": "decrease",
        "module_label": "M1 Somatomotor",
        "group_label": "Adult female",
        "title": "Adult female | Z | M1 Somatomotor",
        "stats_note": "Core family | FD < 0.3 significant in site and site+IQ | FD < 0.2 significant in site and site+IQ",
        "footer_note": "Interpretation: lower within-module degree z-score in ASD",
        "filename": "01_adult_female_Z_M1_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "Strength_neg",
        "direction": "increase",
        "module_label": "M2 Visual-A",
        "group_label": "Child female",
        "title": "Child female | Strength_neg | M2 Visual-A",
        "stats_note": "Core family | M2 is the more stable core result | FD < 0.3 significant in site, site+IQ, site+IQ+RH | FD < 0.2 site+IQ significant, near/significant trend across models",
        "footer_note": "Interpretation: higher negative strength in ASD",
        "filename": "02_child_female_strengthneg_M2_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "Strength_neg",
        "direction": "increase",
        "module_label": "M3 Limbic",
        "group_label": "Child female",
        "title": "Child female | Strength_neg | M3 Limbic",
        "stats_note": "Core family but noisier than M2 | FD < 0.3 significant mainly in site+IQ | FD < 0.2 only near-significant uncorrected trend",
        "footer_note": "Interpretation: supportive increase in negative strength in ASD",
        "filename": "03_child_female_strengthneg_M3_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "PC",
        "direction": "increase",
        "module_label": "M7 DefaultMode",
        "group_label": "Teen male",
        "title": "Teen male | PC | M7 DefaultMode",
        "stats_note": "Core family | FD < 0.3 significant across site, site+IQ, site+IQ+RH | No significant values at FD < 0.2",
        "footer_note": "Interpretation: higher cross-module participation in ASD",
        "filename": "04_teen_male_PC_M7_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "PC_pos",
        "direction": "increase",
        "module_label": "M4 Frontoparietal",
        "group_label": "Adult male",
        "title": "Adult male | PC_pos | M4 Frontoparietal",
        "stats_note": "Core family | Clearest and most stable adult-male shift | FD < 0.3 appeared in site, site+IQ, site+IQ+RH | FD < 0.2 near-significant corrected and uncorrected significant across models",
        "footer_note": "Interpretation: higher positive cross-module participation in ASD",
        "filename": "05_adult_male_PCpos_M4_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "PC_pos",
        "direction": "increase",
        "module_label": "M7 DefaultMode",
        "group_label": "Adult male",
        "title": "Adult male | PC_pos | M7 DefaultMode",
        "stats_note": "Core family | FD < 0.3 strongest in site+IQ+RH with near-significant site/site+IQ trends | FD < 0.2 weaker but supportive",
        "footer_note": "Interpretation: higher positive DMN participation in ASD",
        "filename": "06_adult_male_PCpos_M7_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "Strength_pos",
        "direction": "increase",
        "module_label": "M2 Visual-A",
        "group_label": "Adult male",
        "title": "Adult male | Strength_pos | M2 Visual-A",
        "stats_note": "Core family | FD < 0.3 appeared in site, site+IQ, site+IQ+RH | FD < 0.2 uncorrected significant across all three models",
        "footer_note": "Interpretation: higher positive strength in ASD",
        "filename": "07_adult_male_strengthpos_M2_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "Strength_pos",
        "direction": "increase",
        "module_label": "M4 Frontoparietal",
        "group_label": "Adult male",
        "title": "Adult male | Strength_pos | M4 Frontoparietal",
        "stats_note": "Core family | FD < 0.3 appeared in site and site+IQ | FD < 0.2 no strong significant trend",
        "footer_note": "Interpretation: higher positive frontoparietal strength in ASD",
        "filename": "08_adult_male_strengthpos_M4_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "Strength_pos",
        "direction": "increase",
        "module_label": "M6 Visual-B",
        "group_label": "Adult male",
        "title": "Adult male | Strength_pos | M6 Visual-B",
        "stats_note": "Core family | FD < 0.3 appeared in site, site+IQ, site+IQ+RH | FD < 0.2 near-significant corrected, uncorrected significant across all models",
        "footer_note": "Interpretation: higher positive strength in ASD",
        "filename": "09_adult_male_strengthpos_M6_dx_only.png",
    },

    # ==================== DX-only secondary families ====================

    {
        "kind": "dx_only",
        "metric": "PC",
        "direction": "increase",
        "module_label": "M6 Visual-B",
        "group_label": "Child male",
        "title": "Child male | PC | M6 Visual-B",
        "stats_note": "Secondary family | Strongest child-male PC effect | FD < 0.3 significant in site and site+IQ | FD < 0.2 site+IQ corrected significant, site near-significant",
        "footer_note": "Interpretation: slightly higher participation in ASD",
        "filename": "10_child_male_PC_M6_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "PC",
        "direction": "increase",
        "module_label": "M7 DefaultMode",
        "group_label": "Child male",
        "title": "Child male | PC | M7 DefaultMode",
        "stats_note": "Secondary family | FD < 0.3 site+IQ significant with site and site+IQ+RH near-significant | FD < 0.2 site+IQ corrected significant, site near-significant",
        "footer_note": "Interpretation: slightly higher participation in ASD",
        "filename": "11_child_male_PC_M7_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "PC",
        "direction": "increase",
        "module_label": "M8 DorsalAttention",
        "group_label": "Child male",
        "title": "Child male | PC | M8 DorsalAttention",
        "stats_note": "Secondary family | FD < 0.3 site+IQ significant and site near-significant | FD < 0.2 site+IQ near-significant",
        "footer_note": "Interpretation: slightly higher participation in ASD",
        "filename": "12_child_male_PC_M8_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "PC",
        "direction": "increase",
        "module_label": "M4 Frontoparietal",
        "group_label": "Child female",
        "title": "Child female | PC | M4 Frontoparietal",
        "stats_note": "Secondary family | FD < 0.3 significant for site and uncorrected site+IQ | No meaningful FD < 0.2 support",
        "footer_note": "Interpretation: higher participation in ASD",
        "filename": "13_child_female_PC_M4_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "PC",
        "direction": "increase",
        "module_label": "M7 DefaultMode",
        "group_label": "Child female",
        "title": "Child female | PC | M7 DefaultMode",
        "stats_note": "Secondary family | FD < 0.3 significant for site and uncorrected site+IQ | FD < 0.2 uncorrected site significant",
        "footer_note": "Interpretation: higher participation in ASD",
        "filename": "14_child_female_PC_M7_dx_only.png",
    },
    {
        "kind": "dx_only",
        "metric": "Strength_pos",
        "direction": "decrease",
        "module_label": "M2 Visual-A",
        "group_label": "Child male",
        "title": "Child male | Strength_pos | M2 Visual-A",
        "stats_note": "Secondary family | FD < 0.3 significant in site+IQ+RH and site+IQ uncorrected | FD < 0.2 same pattern",
        "footer_note": "Interpretation: slightly lower positive strength in ASD",
        "filename": "15_child_male_strengthpos_M2_dx_only.png",
    },

    # ==================== DX-only sensitivity-dependent ====================

    {
        "kind": "dx_only",
        "metric": "PC_neg",
        "direction": "sensitivity_dependent",
        "module_label": "M7 DefaultMode",
        "group_label": "Teen female",
        "title": "Teen female | PC_neg | Multiple modules",
        "stats_note": "Sensitivity-dependent family | Full sample showed apparent significance across several modules | Strongly influenced by subject 50127 | Signal weakened after excluding subject 50127",
        "footer_note": "Interpretation: do not treat as a stable core finding",
        "filename": "16_teen_female_PCneg_multimodule_dx_only.png",
    },

    # ==================== DX×SEX findings ====================

    {
        "kind": "dxsex",
        "metric": "Z",
        "male_direction": "increase_or_opposite_pattern",
        "female_direction": "decrease",
        "module_label": "M1 Somatomotor",
        "age_label": "Adult",
        "title": "Adult | DX×SEX | Z | M1 Somatomotor",
        "stats_note": "Significant diagnosis-by-sex interaction driven primarily by females | FD < 0.3 significant in site and site+IQ, uncorrected site+IQ+RH | FD < 0.2 significant in site and site+IQ, uncorrected site+IQ+RH",
        "footer_note": "Interpretation: females with ASD show lower within-module connectivity, opposite pattern to males",
        "filename": "17_adult_Z_M1_dxsex.png",
    },
    {
        "kind": "dxsex",
        "metric": "Z",
        "male_direction": "increase_or_opposite_pattern",
        "female_direction": "decrease",
        "module_label": "M6 Visual-B",
        "age_label": "Adult",
        "title": "Adult | DX×SEX | Z | M6 Visual-B",
        "stats_note": "Significant diagnosis-by-sex interaction driven primarily by females | FD < 0.3 significant in site and site+IQ | FD < 0.2 significant in site and site+IQ",
        "footer_note": "Interpretation: females with ASD show lower within-module connectivity in Visual-B, opposite pattern to males",
        "filename": "18_adult_Z_M6_dxsex.png",
    },
    {
        "kind": "dxsex",
        "metric": "Z_pos",
        "male_direction": "increase_or_opposite_pattern",
        "female_direction": "decrease",
        "module_label": "M1 Somatomotor",
        "age_label": "Adult",
        "title": "Adult | DX×SEX | Z_pos | M1 Somatomotor",
        "stats_note": "Significant diagnosis-by-sex interaction driven primarily by females | FD < 0.3 significant in site and site+IQ, uncorrected site+IQ+RH | FD < 0.2 significant in site and site+IQ, uncorrected site+IQ+RH",
        "footer_note": "Interpretation: females with ASD show lower positive within-module connectivity, opposite pattern to males",
        "filename": "19_adult_Zpos_M1_dxsex.png",
    },
    {
        "kind": "dxsex",
        "metric": "Strength_neg",
        "male_direction": "absent_or_much_weaker",
        "female_direction": "increase",
        "module_label": "M2 Visual-A",
        "age_label": "Child",
        "title": "Child | DX×SEX | Strength_neg | M2 Visual-A",
        "stats_note": "Autistic girls show a unique increase in negative strength that is absent in autistic boys | FD < 0.3 strongest in site+IQ+RH with uncorrected site/site+IQ support | FD < 0.2 significant in site, site+IQ, site+IQ+RH",
        "footer_note": "Interpretation: female-specific increase in negative visual strength",
        "filename": "20_child_strengthneg_M2_dxsex.png",
    },
    {
        "kind": "dxsex",
        "metric": "PC",
        "male_direction": "opposite_pattern",
        "female_direction": "unique_shift",
        "module_label": "M7 DefaultMode",
        "age_label": "Preteen",
        "title": "Preteen | DX×SEX | PC | M7 DefaultMode",
        "stats_note": "Less stable interaction result | Significant diagnosis-by-sex interaction in DMN at FD < 0.2 for site and site+IQ | No significant or near-significant values at FD < 0.3",
        "footer_note": "Interpretation: preteen DMN reorganization differs by sex, but stability is limited",
        "filename": "21_preteen_PC_M7_dxsex.png",
    },
]


# run
def main():
    out_dir = Path("poster_ready_schematics")
    out_dir.mkdir(exist_ok=True)

    for item in FINDINGS:
        out_path = out_dir / item["filename"]

        if item["kind"] == "dx_only":
            make_dx_only_schematic(
                metric=item["metric"],
                direction=item["direction"],
                module_label=item["module_label"],
                group_label=item["group_label"],
                title=item["title"],
                stats_note=item.get("stats_note"),
                footer_note=item.get("footer_note"),
                out_path=out_path,
            )

        elif item["kind"] == "dxsex":
            make_dxsex_schematic(
                metric=item["metric"],
                male_direction=item["male_direction"],
                female_direction=item["female_direction"],
                module_label=item["module_label"],
                age_label=item["age_label"],
                title=item["title"],
                stats_note=item.get("stats_note"),
                footer_note=item.get("footer_note"),
                out_path=out_path,
            )

        else:
            raise ValueError(f"Unknown finding kind: {item['kind']}")

    print(f"Saved schematics to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()