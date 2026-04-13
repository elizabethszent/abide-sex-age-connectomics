from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Patch


# config

# use ONLY fd < 0.3
DX_ROOT = Path(
    "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/results/hubs_organized/OVERALL_ageSexMatched_fd-0.3"
)

# aggregate ONLY across these models
KEEP_MODELS = {"site", "site_iq", "site_iq_rh"}

OUT_DIR = Path("poster_ready_summaries_by_sex_fd03_models")
OUT_FEMALE = OUT_DIR / "summary_table_female_dx_only_fd03_models.png"
OUT_MALE = OUT_DIR / "summary_table_male_dx_only_fd03_models.png"

ALPHA_FDR = 0.05
ALPHA_UNC = 0.05

AGE_ORDER = ["Child", "Preteen", "Teen", "Adult"]
METRIC_ORDER = ["Z", "Z+", "Z-", "PC", "PC+", "PC-", "Strength+", "Strength-"]

# manually remove unstable findings from the final summaries
EXCLUDED_CELLS = {
    # unstable due to one excluded subject
    ("female", "Teen", "PC-"),
}


# style
BG = "white"
TEXT = "#222222"
LIGHT_TEXT = "#555555"
GRID = "#cfcfcf"
EMPTY_FILL = "#ffffff"

INC_COLOR = "#d95f02"   # ASD > CTL
DEC_COLOR = "#1f78b4"   # ASD < CTL
MIXED_COLOR = "#bdbdbd"

TITLE_SIZE = 18
SUBTITLE_SIZE = 11
HEADER_SIZE = 11
LABEL_SIZE = 11
CELL_TEXT_SIZE = 10
CAPTION_SIZE = 9


# data
@dataclass
class ModuleEvidence:
    corrected_direction_votes: list[str] = field(default_factory=list)
    uncorrected_direction_votes: list[str] = field(default_factory=list)

    corrected_hits: int = 0
    uncorrected_hits: int = 0

    corrected_models: set[str] = field(default_factory=set)
    uncorrected_models: set[str] = field(default_factory=set)

    def has_corrected(self) -> bool:
        return self.corrected_hits > 0

    def has_uncorrected_only(self) -> bool:
        return self.corrected_hits == 0 and self.uncorrected_hits > 0

    def final_direction(self) -> str:
        votes = self.corrected_direction_votes if self.corrected_direction_votes else self.uncorrected_direction_votes
        if not votes:
            return "flat"

        inc = sum(1 for d in votes if d == "increase")
        dec = sum(1 for d in votes if d == "decrease")

        if inc > dec:
            return "increase"
        if dec > inc:
            return "decrease"
        return "mixed"

    def marker(self) -> str:
        if self.corrected_hits >= 2:
            return "★"
        if self.corrected_hits == 1:
            return "*"
        if self.uncorrected_hits >= 1:
            return "•"
        return ""

    def support_strength(self) -> tuple[int, int]:
        return (self.corrected_hits, self.uncorrected_hits)


# helpers
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def parse_float(value: str) -> float | None:
    v = normalize_spaces(value)
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_module_id(value: str) -> str | None:
    v = normalize_spaces(value)
    m = re.search(r"\bM?([1-8])\b", v)
    if not m:
        return None
    return f"M{m.group(1)}"


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            clean = {normalize_spaces(k): normalize_spaces(v) for k, v in row.items() if k is not None}
            if any(v for v in clean.values()):
                rows.append(clean)
        return rows


def prettify_metric(metric_dir: str, sign_dir: str) -> str:
    md = normalize_spaces(metric_dir).lower()
    sd = normalize_spaces(sign_dir).lower()

    if md in {"z", "withinmoduledegree", "within_module_degree", "withinmoduledegreez"}:
        base = "Z"
    elif md in {"pc", "participationcoefficient", "participation_coefficient"}:
        base = "PC"
    elif md in {"strength", "node_strength"}:
        base = "Strength"
    else:
        base = metric_dir

    if sd in {"pos", "positive"}:
        return f"{base}+"
    if sd in {"neg", "negative"}:
        return f"{base}-"
    return base


def prettify_age(age_dir: str) -> str:
    mapping = {
        "child_0_9": "Child",
        "preteen_10_12": "Preteen",
        "teen_13_17": "Teen",
        "adult_18_plus": "Adult",
    }
    return mapping.get(age_dir, age_dir.replace("_", " ").title())


def direction_from_beta(beta_ctl_minus_asd: float | None, mean_ctl: float | None, mean_asd: float | None) -> str:
    # beta_CTL_minus_ASD > 0 => CTL > ASD => ASD < CTL => decrease
    # beta_CTL_minus_ASD < 0 => ASD > CTL => increase
    if beta_ctl_minus_asd is not None:
        if beta_ctl_minus_asd > 0:
            return "decrease"
        if beta_ctl_minus_asd < 0:
            return "increase"
        return "flat"

    if mean_ctl is not None and mean_asd is not None:
        if mean_asd > mean_ctl:
            return "increase"
        if mean_asd < mean_ctl:
            return "decrease"
    return "flat"


# path collection
def collect_dx_table_paths(dx_root: Path) -> list[Path]:
    if not dx_root.exists():
        raise FileNotFoundError(f"DX root does not exist: {dx_root}")

    paths: list[Path] = []
    for csv_path in dx_root.glob("*/*/*/*/*/table.csv"):
        parts = csv_path.relative_to(dx_root).parts
        # expected: metric/sign/age/sex/model/table.csv
        if len(parts) != 6:
            continue
        metric_dir, sign_dir, age_dir, sex_dir, model_dir, _ = parts
        if model_dir not in KEEP_MODELS:
            continue
        paths.append(csv_path)

    if not paths:
        raise FileNotFoundError(
            f"No matching table.csv files found under {dx_root} for models {sorted(KEEP_MODELS)}"
        )
    return paths


# build results
def build_results(dx_root: Path):
    """
    results[sex][age][metric][module] = ModuleEvidence
    using ONLY fd<0.3 and ONLY site/site_iq/site_iq_rh
    """
    results: dict[str, dict[str, dict[str, dict[str, ModuleEvidence]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    table_paths = collect_dx_table_paths(dx_root)

    for csv_path in table_paths:
        rel_parts = csv_path.relative_to(dx_root).parts
        metric_dir, sign_dir, age_dir, sex_dir, model_dir, _ = rel_parts

        metric = prettify_metric(metric_dir, sign_dir)
        age = prettify_age(age_dir)
        sex = normalize_spaces(sex_dir).lower()

        rows = load_csv_rows(csv_path)

        for row in rows:
            module_id = parse_module_id(row.get("module", ""))
            if module_id is None:
                continue

            p_dx = parse_float(row.get("p_DX", ""))
            p_dx_fdr = parse_float(row.get("p_DX_FDR", ""))
            beta = parse_float(row.get("beta_CTL_minus_ASD", ""))
            mean_ctl = parse_float(row.get("mean_CTL", ""))
            mean_asd = parse_float(row.get("mean_ASD", ""))

            is_fdr = p_dx_fdr is not None and p_dx_fdr < ALPHA_FDR
            is_unc = (p_dx is not None and p_dx < ALPHA_UNC and not is_fdr)

            if not is_fdr and not is_unc:
                continue

            direction = direction_from_beta(beta, mean_ctl, mean_asd)

            cell = results[sex][age][metric]
            if module_id not in cell:
                cell[module_id] = ModuleEvidence()

            ev = cell[module_id]
            if is_fdr:
                ev.corrected_hits += 1
                ev.corrected_models.add(model_dir)
                if direction != "flat":
                    ev.corrected_direction_votes.append(direction)
            else:
                ev.uncorrected_hits += 1
                ev.uncorrected_models.add(model_dir)
                if direction != "flat":
                    ev.uncorrected_direction_votes.append(direction)

    return results


def apply_manual_exclusions(results):
    """
    Remove cells that should be blanked out from the final summary.
    """
    for sex, age, metric in EXCLUDED_CELLS:
        if sex in results and age in results[sex] and metric in results[sex][age]:
            del results[sex][age][metric]


# cell rendering
def choose_modules_for_cell(module_map: dict[str, ModuleEvidence]) -> dict[str, ModuleEvidence]:
    """
    Corrected-first logic:
    - if any corrected module exists in the cell, show ONLY corrected modules
    - otherwise show uncorrected-only modules
    """
    corrected = {m: ev for m, ev in module_map.items() if ev.has_corrected()}
    if corrected:
        return corrected
    return {m: ev for m, ev in module_map.items() if ev.has_uncorrected_only()}


def cell_background(module_map: dict[str, ModuleEvidence]) -> tuple[str, float, str | None]:
    shown = choose_modules_for_cell(module_map)
    if not shown:
        return EMPTY_FILL, 1.0, None

    directions = {ev.final_direction() for ev in shown.values()}

    if directions == {"increase"}:
        color = INC_COLOR
    elif directions == {"decrease"}:
        color = DEC_COLOR
    else:
        color = MIXED_COLOR

    only_uncorrected = all(ev.has_uncorrected_only() for ev in shown.values())
    strongest_corrected = max((ev.corrected_hits for ev in shown.values()), default=0)

    if strongest_corrected >= 2:
        alpha = 0.28
    elif strongest_corrected == 1:
        alpha = 0.18
    else:
        alpha = 0.10

    hatch = "//" if only_uncorrected else None
    return color, alpha, hatch


def format_module_lines(module_map: dict[str, ModuleEvidence], max_lines: int = 5) -> list[str]:
    shown = choose_modules_for_cell(module_map)
    if not shown:
        return []

    def mod_key(item):
        module_id, ev = item
        module_num = int(module_id[1:])
        return (-ev.support_strength()[0], -ev.support_strength()[1], module_num)

    ordered = sorted(shown.items(), key=mod_key)

    lines = []
    for module_id, ev in ordered[:max_lines]:
        direction = ev.final_direction()
        arrow = "↑" if direction == "increase" else "↓" if direction == "decrease" else "↕"
        marker = ev.marker()
        lines.append(f"{module_id}{arrow}{marker}")

    remaining = len(ordered) - max_lines
    if remaining > 0:
        lines.append(f"+{remaining} more")

    return lines


# draw table
def draw_sex_table(
    sex: str,
    sex_results: dict[str, dict[str, dict[str, ModuleEvidence]]],
    out_path: Path,
    metric_columns: list[str],
):
    ages = [a for a in AGE_ORDER if a in sex_results]
    if not ages:
        print(f"[warn] no rows for sex={sex}")
        return

    n_rows = len(ages)
    n_cols = len(metric_columns)

    left_margin = 1.8
    right_margin = 0.4
    top_margin = 1.6
    bottom_margin = 1.4
    cell_w = 1.55
    cell_h = 1.55

    total_w = left_margin + n_cols * cell_w + right_margin
    total_h = bottom_margin + n_rows * cell_h + top_margin

    fig_w = max(11, 1.2 + n_cols * 1.55)
    fig_h = max(6.5, 2.0 + n_rows * 1.55)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    title_sex = sex.capitalize()

    ax.text(
        total_w / 2,
        total_h - 0.35,
        f"{title_sex} | DX-only summary",
        ha="center",
        va="center",
        fontsize=TITLE_SIZE,
        weight="bold",
        color=TEXT,
    )
    ax.text(
        total_w / 2,
        total_h - 0.78,
        "Rows = age group, columns = measure, aggregated across site, site+IQ, and site+IQ+RH at FD < 0.3",
        ha="center",
        va="center",
        fontsize=SUBTITLE_SIZE,
        color=LIGHT_TEXT,
    )

    y_header = bottom_margin + n_rows * cell_h + 0.28
    for j, metric in enumerate(metric_columns):
        x = left_margin + j * cell_w
        ax.text(
            x + 0.5 * cell_w,
            y_header,
            metric,
            ha="center",
            va="center",
            fontsize=HEADER_SIZE,
            weight="bold",
            color=TEXT,
        )

    for i, age in enumerate(ages):
        y = bottom_margin + (n_rows - 1 - i) * cell_h

        ax.text(
            left_margin - 0.15,
            y + 0.5 * cell_h,
            age,
            ha="right",
            va="center",
            fontsize=LABEL_SIZE,
            weight="bold",
            color=TEXT,
        )

        for j, metric in enumerate(metric_columns):
            x = left_margin + j * cell_w
            module_map = sex_results.get(age, {}).get(metric, {})

            bg_color, bg_alpha, hatch = cell_background(module_map)

            rect = Rectangle(
                (x, y),
                cell_w,
                cell_h,
                facecolor=to_rgba(bg_color, bg_alpha) if bg_color != EMPTY_FILL else EMPTY_FILL,
                edgecolor=GRID,
                linewidth=1.0,
                hatch=hatch,
            )
            ax.add_patch(rect)

            lines = format_module_lines(module_map, max_lines=5)
            if lines:
                ax.text(
                    x + 0.5 * cell_w,
                    y + 0.5 * cell_h,
                    "\n".join(lines),
                    ha="center",
                    va="center",
                    fontsize=CELL_TEXT_SIZE,
                    color=TEXT,
                    linespacing=1.15,
                )

    ax.add_patch(
        Rectangle(
            (left_margin, bottom_margin),
            n_cols * cell_w,
            n_rows * cell_h,
            fill=False,
            edgecolor="#888888",
            linewidth=1.2,
        )
    )

    handles = [
        Patch(facecolor=to_rgba(INC_COLOR, 0.22), edgecolor="#666666", label="ASD > CTL"),
        Patch(facecolor=to_rgba(DEC_COLOR, 0.22), edgecolor="#666666", label="ASD < CTL"),
        Patch(facecolor=to_rgba(MIXED_COLOR, 0.18), edgecolor="#666666", label="Mixed directions"),
        Patch(facecolor=to_rgba("#999999", 0.10), edgecolor="#666666", hatch="//", label="Only uncorrected"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    fig.text(
        0.5,
        0.03,
        "Cell text format: M#↑★ = ASD > CTL with repeated corrected support across models; "
        "M#↓* = ASD < CTL with corrected support; M#• = only uncorrected support. "
        "Uncorrected modules are shown only when that cell has no corrected modules.",
        ha="center",
        va="center",
        fontsize=CAPTION_SIZE,
        color=LIGHT_TEXT,
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# main
def main():
    results = build_results(DX_ROOT)
    apply_manual_exclusions(results)

    observed_metrics = set()
    for sex in results:
        for age in results[sex]:
            observed_metrics.update(results[sex][age].keys())

    metric_columns = [m for m in METRIC_ORDER if m in observed_metrics]

    if not metric_columns:
        raise RuntimeError("No significant dx findings were found in the configured tables.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    draw_sex_table(
        sex="female",
        sex_results=results.get("female", {}),
        out_path=OUT_FEMALE,
        metric_columns=metric_columns,
    )

    draw_sex_table(
        sex="male",
        sex_results=results.get("male", {}),
        out_path=OUT_MALE,
        metric_columns=metric_columns,
    )

    print(f"Saved female table to: {OUT_FEMALE.resolve()}")
    print(f"Saved male table to:   {OUT_MALE.resolve()}")


if __name__ == "__main__":
    main()