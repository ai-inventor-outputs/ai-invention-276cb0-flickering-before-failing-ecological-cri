#!/usr/bin/env python3
"""Generate 8 publication-quality figures for CSD-LLM NeurIPS paper.

Loads data from 5 experiment dependencies and produces:
  Fig 1: Conceptual overview (ecological + LLM analog)
  Fig 2: Accuracy profiles (arithmetic + graph coloring, 2x3 grid)
  Fig 3: CSD indicator dashboard (5 stacked panels)
  Fig 4: UMAP flickering visualization (2x3 grid)
  Fig 5: Classifier comparison bar chart (CSD vs SPUQ)
  Fig 6: Cusp/mixture/fold/DDM model fits
  Fig 7: Temperature dose-response (variance + disagreement)
  Fig 8: Prospective protocol schematic

Gradual scaling: mini (Figs 2-3) -> medium (add 1,4,5,6) -> full (all 8).
"""

import json
import sys
import gc
import os
import math
import resource
import time
from pathlib import Path

# ──────────────────────────────────────────────
# 0. Hardware detection and memory limits
# ──────────────────────────────────────────────

def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.70 * 1e9)  # 70% of container
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ──────────────────────────────────────────────
# 1. Imports (after resource limits set)
# ──────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from scipy import stats
from loguru import logger

# ──────────────────────────────────────────────
# 2. Logging
# ──────────────────────────────────────────────

WORKSPACE = Path(
    "/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop"
    "/iter_5/gen_art/eval_id2_it5__opus"
)
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ──────────────────────────────────────────────
# 3. Paths and constants
# ──────────────────────────────────────────────

OUTPUT_DIR = WORKSPACE / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEP_BASE = Path("/ai-inventor/aii_pipeline/runs/neurips-open/3_invention_loop")
EXP1_IT2 = DEP_BASE / "iter_2/gen_art/exp_id1_it2__opus/full_method_out.json"
EXP3_IT2 = DEP_BASE / "iter_2/gen_art/exp_id3_it2__opus/full_method_out.json"
EXP3_IT3 = DEP_BASE / "iter_3/gen_art/exp_id3_it3__opus/full_method_out.json"
EXP2_IT4 = DEP_BASE / "iter_4/gen_art/exp_id2_it4__opus/full_method_out.json"
EXP3_IT4 = DEP_BASE / "iter_4/gen_art/exp_id3_it4__opus/full_method_out.json"

SINGLE_COL = 3.25   # inches
DOUBLE_COL = 6.75
DPI = 300
PALETTE = sns.color_palette("colorblind")

MODEL_SHORT = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "google/gemini-2.0-flash-001": "Gemini Flash",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "google/gemini-2.0-flash-lite-001": "Gemini Flash Lite",
}

FIGURE_FILENAMES = {
    "fig1": "fig1_conceptual_overview",
    "fig2": "fig2_accuracy_profiles",
    "fig3": "fig3_csd_dashboard",
    "fig4": "fig4_umap_flickering",
    "fig5": "fig5_classifier_comparison",
    "fig6": "fig6_model_fits",
    "fig7": "fig7_temperature_effect",
    "fig8": "fig8_prospective_protocol",
}

FIGURE_CONFIGS = {
    "fig1": {"title": "Conceptual Overview", "type": "conceptual",
             "column": "double", "stage": "medium", "target_w": DOUBLE_COL},
    "fig2": {"title": "Accuracy Profiles", "type": "empirical",
             "column": "double", "stage": "mini", "target_w": DOUBLE_COL},
    "fig3": {"title": "CSD Indicator Dashboard", "type": "empirical",
             "column": "single", "stage": "mini", "target_w": SINGLE_COL},
    "fig4": {"title": "UMAP Flickering Visualization", "type": "visualization",
             "column": "double", "stage": "medium", "target_w": DOUBLE_COL},
    "fig5": {"title": "Classifier Comparison", "type": "empirical",
             "column": "single", "stage": "medium", "target_w": SINGLE_COL},
    "fig6": {"title": "Model Fits", "type": "empirical",
             "column": "single", "stage": "medium", "target_w": SINGLE_COL},
    "fig7": {"title": "Temperature Effect", "type": "empirical",
             "column": "single", "stage": "full", "target_w": SINGLE_COL},
    "fig8": {"title": "Prospective Protocol Schematic", "type": "conceptual",
             "column": "single", "stage": "full", "target_w": SINGLE_COL},
}

# ──────────────────────────────────────────────
# 4. NeurIPS style
# ──────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ──────────────────────────────────────────────
# 5. Utility functions
# ──────────────────────────────────────────────


def load_json(path: Path) -> dict:
    """Load a JSON file, logging size."""
    sz_mb = path.stat().st_size / 1e6
    logger.info(f"Loading {path.name} ({sz_mb:.1f} MB)")
    data = json.loads(path.read_text())
    logger.info(f"Loaded {path.name} successfully")
    return data


def get_dataset(data: dict, name: str) -> list:
    """Get a dataset by name from the datasets list."""
    for ds in data.get("datasets", []):
        if ds["dataset"] == name:
            return ds["examples"]
    raise KeyError(f"Dataset '{name}' not found in {[d['dataset'] for d in data.get('datasets', [])]}")


def safe_float(val, default: float = 0.0) -> float:
    """Convert a value to float, handling 'nan', None, etc."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def save_figure(fig: plt.Figure, name: str) -> dict:
    """Save figure as PNG and PDF, return metrics dict."""
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(str(png_path), dpi=DPI)
    fig.savefig(str(pdf_path))
    w, h = fig.get_size_inches()
    plt.close(fig)

    png_kb = png_path.stat().st_size / 1024
    pdf_kb = pdf_path.stat().st_size / 1024

    logger.info(f"  Saved {name}: PNG={png_kb:.1f}KB, PDF={pdf_kb:.1f}KB, "
                f"size={w:.2f}x{h:.2f}in")
    return {
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "png_kb": round(png_kb, 2),
        "pdf_kb": round(pdf_kb, 2),
        "width": round(w, 4),
        "height": round(h, 4),
    }


# ──────────────────────────────────────────────
# 6. Figure functions
# ──────────────────────────────────────────────


def make_fig1() -> tuple[plt.Figure, int]:
    """Figure 1: Conceptual Overview — Ecological + LLM analog."""
    logger.info("Generating Figure 1: Conceptual Overview")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    # --- Panel (a): Ecological Analog ---
    x = np.linspace(0, 10, 500)
    np.random.seed(42)
    noise = np.random.normal(0, 1, len(x))

    # Piecewise clarity curve with flickering zone
    clarity = np.piecewise(
        x,
        [x < 3.5, (x >= 3.5) & (x < 6.5), x >= 6.5],
        [
            lambda d: 0.88 + 0.02 * np.sin(d * 3),
            lambda d: 0.55 + 0.20 * np.sin(d * 6) * np.exp(-((d - 5.0) ** 2) / 3.0),
            lambda d: 0.15 + 0.02 * np.sin(d * 3),
        ],
    )

    ax1.axvspan(0, 3.5, alpha=0.12, color="steelblue", zorder=0)
    ax1.axvspan(3.5, 6.5, alpha=0.12, color="gold", zorder=0)
    ax1.axvspan(6.5, 10, alpha=0.12, color="gray", zorder=0)

    ax1.plot(x, clarity, color="black", linewidth=0.9)
    ax1.set_xlabel("Nutrient Loading (Control Parameter)")
    ax1.set_ylabel("Water Clarity (State)")
    ax1.text(1.75, 0.05, "Clear", fontsize=7, ha="center", fontstyle="italic",
             color="steelblue")
    ax1.text(5.0, 0.05, "Flickering", fontsize=7, ha="center", fontstyle="italic",
             color="goldenrod")
    ax1.text(8.25, 0.05, "Turbid", fontsize=7, ha="center", fontstyle="italic",
             color="gray")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("(a) Ecological CSD Analog", fontsize=8)
    ax1.tick_params(labelbottom=False, labelleft=False)

    # --- Panel (b): LLM Analog ---
    positions = [2, 5, 8]
    labels = ["Unimodal\nCorrect", "Flickering\n(Bimodal)", "Unimodal\nIncorrect"]
    colors_dist = [PALETTE[2], PALETTE[1], PALETTE[3]]

    for pos, label, color in zip(positions, labels, colors_dist):
        y_range = np.linspace(0, 1, 200)
        if pos == 2:
            dist = stats.norm.pdf(y_range, 0.85, 0.06)
        elif pos == 5:
            dist = (0.5 * stats.norm.pdf(y_range, 0.30, 0.08)
                    + 0.5 * stats.norm.pdf(y_range, 0.75, 0.08))
        else:
            dist = stats.norm.pdf(y_range, 0.15, 0.06)

        dist_scaled = dist / dist.max() * 0.9
        ax2.fill_betweenx(y_range, pos - dist_scaled, pos + dist_scaled,
                          alpha=0.45, color=color)
        ax2.plot(pos - dist_scaled, y_range, color=color, linewidth=0.7)
        ax2.plot(pos + dist_scaled, y_range, color=color, linewidth=0.7)
        ax2.text(pos, -0.13, label, fontsize=5.5, ha="center", va="top")

    ax2.axvspan(0, 3.5, alpha=0.06, color="steelblue", zorder=0)
    ax2.axvspan(3.5, 6.5, alpha=0.06, color="gold", zorder=0)
    ax2.axvspan(6.5, 10, alpha=0.06, color="gray", zorder=0)
    ax2.set_xlabel("Task Difficulty (Control Parameter)")
    ax2.set_ylabel("Response Quality Distribution")
    ax2.set_title("(b) LLM Analog", fontsize=8)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.18, 1.12)
    ax2.tick_params(labelbottom=False, labelleft=False)

    fig.tight_layout()
    return fig, 0


def make_fig2(arith_data: dict, gc_data: dict) -> tuple[plt.Figure, int]:
    """Figure 2: Accuracy Profiles — 2x3 grid (arithmetic + graph coloring)."""
    logger.info("Generating Figure 2: Accuracy Profiles")

    # --- Extract arithmetic per-level accuracy ---
    arith_models: dict[str, dict] = {}
    for ds in arith_data["datasets"]:
        exs = ds["examples"]
        model = exs[0]["metadata_model"]
        d_star = exs[0]["metadata_d_star"]
        levels = sorted(set(ex["metadata_difficulty_level"] for ex in exs))
        level_acc = {ex["metadata_difficulty_level"]: safe_float(ex["predict_accuracy"])
                     for ex in exs}
        arith_models[model] = {
            "levels": np.array(sorted(level_acc.keys())),
            "accs": np.array([level_acc[l] for l in sorted(level_acc.keys())]),
            "d_star": d_star,
        }

    # --- Extract graph-coloring per-level accuracy ---
    gc_meta = gc_data.get("metadata", {}).get("analysis", {}).get("models", [])
    d_star_map = {m["model"]: m["d_star"] for m in gc_meta}

    gc_models: dict[str, dict] = {}
    for ds in gc_data["datasets"]:
        exs = ds["examples"]
        model = exs[0]["metadata_model"]
        d_star = d_star_map.get(model, 10)
        # Per-level accuracy (same for every response at that level)
        level_acc: dict[int, float] = {}
        for ex in exs:
            lev = ex["metadata_difficulty_level"]
            if lev not in level_acc:
                level_acc[lev] = float(ex["metadata_csd_accuracy"])
        gc_models[model] = {
            "levels": np.array(sorted(level_acc.keys())),
            "accs": np.array([level_acc[l] for l in sorted(level_acc.keys())]),
            "d_star": d_star,
        }

    # --- Build 2x3 figure ---
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 4.0))
    n_total = 0

    arith_order = [
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.1-8b-instruct",
        "openai/gpt-4o-mini",
    ]
    gc_order = [
        "google/gemini-2.0-flash-001",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-lite-001",
    ]

    for row, (order, models) in enumerate(
        [(arith_order, arith_models), (gc_order, gc_models)]
    ):
        for col, model in enumerate(order):
            ax = axes[row, col]
            md = models[model]
            lvls = md["levels"]
            accs = md["accs"]
            n_resp = 50
            ci = 1.96 * np.sqrt(np.clip(accs * (1 - accs), 0, None) / n_resp)

            ax.plot(lvls, accs, color=PALETTE[col], marker="o", markersize=2,
                    linewidth=1.2)
            ax.fill_between(lvls, np.clip(accs - ci, 0, 1),
                            np.clip(accs + ci, 0, 1),
                            alpha=0.2, color=PALETTE[col])
            ax.axvline(md["d_star"], color="red", linestyle="--", linewidth=0.8,
                       alpha=0.7)
            # d* label, placed to the right unless near right edge
            x_pos = md["d_star"] + 0.4
            ax.text(x_pos, 0.97, f"d*={md['d_star']}", color="red", fontsize=5.5,
                    transform=ax.get_xaxis_transform(), va="top",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                              pad=0.3))

            ax.set_title(MODEL_SHORT.get(model, model), fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            if col == 0:
                ax.set_ylabel("Accuracy")
            if row == 1:
                ax.set_xlabel("Difficulty Level")
            n_total += len(lvls)

    # Row labels
    axes[0, 0].annotate(
        "Arithmetic", xy=(-0.42, 0.5), xycoords="axes fraction",
        fontsize=8, fontweight="bold", rotation=90, va="center", ha="center",
    )
    axes[1, 0].annotate(
        "Graph Coloring", xy=(-0.42, 0.5), xycoords="axes fraction",
        fontsize=8, fontweight="bold", rotation=90, va="center", ha="center",
    )

    fig.tight_layout()
    return fig, n_total


def make_fig3(gc_data: dict) -> tuple[plt.Figure, int]:
    """Figure 3: CSD Indicator Dashboard — 5 stacked panels for Gemini Flash."""
    logger.info("Generating Figure 3: CSD Indicator Dashboard")

    exs = get_dataset(gc_data, "graph_coloring_csd_gemini-2.0-flash-001")

    # Extract unique per-level CSD values
    level_data: dict[int, dict] = {}
    for ex in exs:
        lev = ex["metadata_difficulty_level"]
        if lev not in level_data:
            level_data[lev] = {
                "accuracy": float(ex["metadata_csd_accuracy"]),
                "variance": float(ex["metadata_csd_embedding_variance"]),
                "dip_pvalue": float(ex["metadata_csd_dip_pvalue"]),
                "silhouette": float(ex["metadata_csd_silhouette_score"]),
                "disagreement": float(ex["metadata_csd_disagreement_rate"]),
            }

    levels = np.array(sorted(level_data.keys()))
    d_star = 14

    metrics_list = [
        ("Accuracy", [level_data[l]["accuracy"] for l in levels]),
        ("Embedding\nVariance", [level_data[l]["variance"] for l in levels]),
        ("1 - Dip p-value", [1.0 - level_data[l]["dip_pvalue"] for l in levels]),
        ("Silhouette\nScore", [level_data[l]["silhouette"] for l in levels]),
        ("Disagreement\nRate", [level_data[l]["disagreement"] for l in levels]),
    ]

    # Flickering zone: dip p < 0.05 AND accuracy > 0.5
    flicker = [l for l in levels
               if level_data[l]["dip_pvalue"] < 0.05 and level_data[l]["accuracy"] > 0.5]

    fig, axes = plt.subplots(5, 1, figsize=(SINGLE_COL, 5.5), sharex=True)

    for i, (name, values) in enumerate(metrics_list):
        ax = axes[i]
        ax.plot(levels, values, color=PALETTE[i], marker="o", markersize=2.5,
                linewidth=1.2)
        ax.axvline(d_star, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(name, fontsize=6.5)

        if len(flicker) > 0:
            ax.axvspan(min(flicker) - 0.5, max(flicker) + 0.5,
                       alpha=0.12, color="gold", zorder=0)

        if i < 4:
            ax.tick_params(labelbottom=False)

    axes[0].text(d_star + 0.4, 0.97, f"d*={d_star}", color="red", fontsize=5.5,
                 transform=axes[0].get_xaxis_transform(), va="top",
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.3))

    # Flickering zone label
    if len(flicker) > 0:
        mid_f = (min(flicker) + max(flicker)) / 2
        axes[0].text(mid_f, 0.45, "flickering\nzone", fontsize=5, ha="center",
                     color="goldenrod", fontstyle="italic", alpha=0.8)

    axes[-1].set_xlabel("Difficulty Level")
    fig.tight_layout()
    return fig, len(levels) * 5


def make_fig4(gc_data: dict) -> tuple[plt.Figure, int]:
    """Figure 4: UMAP Flickering Visualization — 2x3 grid."""
    logger.info("Generating Figure 4: UMAP Flickering Visualization")

    model_configs = [
        {
            "model": "google/gemini-2.0-flash-001",
            "dataset": "graph_coloring_csd_gemini-2.0-flash-001",
            "d_star": 14,
            "levels": [3, 14, 20],
            "labels": ["Easy (d=3)", "Boundary (d=14)", "Hard (d=20)"],
        },
        {
            "model": "openai/gpt-4o-mini",
            "dataset": "graph_coloring_csd_gpt-4o-mini",
            "d_star": 10,
            "levels": [2, 10, 18],
            "labels": ["Easy (d=2)", "Boundary (d=10)", "Hard (d=18)"],
        },
    ]

    # Choose dimensionality reduction method
    try:
        from umap import UMAP
        use_umap = True
        logger.info("Using UMAP for dimensionality reduction")
    except ImportError:
        use_umap = False
        logger.info("UMAP not available, falling back to PCA")

    # Load sentence-transformers
    import torch
    torch.set_num_threads(NUM_CPUS)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded SentenceTransformer all-MiniLM-L6-v2")

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 4.5))
    total_points = 0

    for row, config in enumerate(model_configs):
        exs = get_dataset(gc_data, config["dataset"])
        logger.info(f"  Processing {MODEL_SHORT.get(config['model'], config['model'])}: "
                    f"{len(exs)} responses")

        # Collect all responses
        all_texts = []
        all_levels = []
        all_correct = []
        for ex in exs:
            text = ex.get("predict_model_response", "")
            if not text:
                continue
            all_texts.append(text)
            all_levels.append(ex["metadata_difficulty_level"])
            correct_val = ex.get("predict_is_correct", "false")
            all_correct.append(str(correct_val).lower() == "true")

        all_levels_arr = np.array(all_levels)
        all_correct_arr = np.array(all_correct)

        # Embed all texts in batches
        logger.info(f"  Embedding {len(all_texts)} texts...")
        t0 = time.time()
        embeddings = embed_model.encode(all_texts, batch_size=256,
                                        show_progress_bar=False)
        logger.info(f"  Embedding done in {time.time() - t0:.1f}s")

        # Dimensionality reduction on ALL responses
        logger.info("  Running dimensionality reduction...")
        t0 = time.time()
        if use_umap:
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                           random_state=42, n_jobs=NUM_CPUS)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)

        coords_2d = reducer.fit_transform(embeddings)
        logger.info(f"  Reduction done in {time.time() - t0:.1f}s")

        # Plot 3 selected levels
        for col, (level, label) in enumerate(zip(config["levels"], config["labels"])):
            ax = axes[row, col]
            mask = all_levels_arr == level
            if mask.sum() == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")
                ax.set_title(label, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x = coords_2d[mask, 0]
            y = coords_2d[mask, 1]
            c_arr = all_correct_arr[mask]

            colors = np.array([PALETTE[2] if c else PALETTE[3] for c in c_arr])
            ax.scatter(x, y, c=colors, s=15, alpha=0.65, edgecolors="none")
            ax.set_title(label, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            total_points += int(mask.sum())

        # Row label
        axes[row, 0].set_ylabel(
            MODEL_SHORT.get(config["model"], config["model"]), fontsize=7
        )

        # Cleanup per-model
        del embeddings, coords_2d
        gc.collect()

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[2],
               markersize=5, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[3],
               markersize=5, label="Incorrect"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    # Free model
    del embed_model
    gc.collect()

    return fig, total_points


def make_fig5(clf_data: dict) -> tuple[plt.Figure, int]:
    """Figure 5: Classifier Comparison — grouped bars + cost inset."""
    logger.info("Generating Figure 5: Classifier Comparison")

    clf_comp = clf_data["metadata"]["classifier_comparison"]

    # Classifiers ordered from highest to lowest LOPO F1
    classifiers = [
        {"name": "CSD-full", "key": "csd_zt_reldist_rf"},
        {"name": "CSD-diff", "key": "csd_raw_diff_svm"},
        {"name": "SPUQ", "key": "spuq_accuracy_rf"},
        {"name": "Disagree.", "key": "disagreement_only_logreg"},
        {"name": "CSD-raw", "key": "csd_raw_logreg"},
        {"name": "Bimodal.", "key": "bimodality_only_logreg"},
    ]

    fig, (ax, ax_cost) = plt.subplots(1, 2, figsize=(SINGLE_COL, 3.5),
                                       gridspec_kw={"width_ratios": [5, 1.2],
                                                    "wspace": 0.45})

    x = np.arange(len(classifiers))
    width = 0.32

    lopo = [clf_comp[c["key"]]["lopo_f1"] for c in classifiers]
    loto = [clf_comp[c["key"]]["loto_f1"] for c in classifiers]

    bars1 = ax.bar(x - width / 2, lopo, width, label="LOPO F1",
                   color=PALETTE[0], alpha=0.85)
    bars2 = ax.bar(x + width / 2, loto, width, label="LOTO F1",
                   color=PALETTE[1], alpha=0.85)

    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels([c["name"] for c in classifiers], fontsize=5.5,
                       rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=5.5, loc="upper right")

    # Value labels on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=4.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=4.5)

    # Cost panel — separate subplot to avoid overlap
    cost_x = [0, 1]
    cost_calls = [0, 1520]
    bars_cost = ax_cost.bar(cost_x, cost_calls,
                            color=[PALETTE[0], PALETTE[4]], alpha=0.65, width=0.6)
    ax_cost.set_xticks(cost_x)
    ax_cost.set_xticklabels(["CSD", "SPUQ"], fontsize=5.5)
    ax_cost.set_ylabel("Extra API Calls", fontsize=6)
    ax_cost.tick_params(labelsize=5)
    ax_cost.set_title("Cost", fontsize=7)
    for b in bars_cost:
        ax_cost.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                     str(int(b.get_height())), ha="center", fontsize=5)

    fig.tight_layout()
    return fig, len(classifiers) * 2


def make_fig6(fit_data: dict) -> tuple[plt.Figure, int]:
    """Figure 6: Model Fits — cusp/mixture/fold/DDM for arith__gemini-flash."""
    logger.info("Generating Figure 6: Model Fits")

    # Get per-level fit data
    all_fits = get_dataset(fit_data, "per_level_fits")
    series = [ex for ex in all_fits if ex["metadata_series"] == "arith__gemini-flash"]
    series.sort(key=lambda x: x["metadata_difficulty_level"])

    levels = np.array([ex["metadata_difficulty_level"] for ex in series])
    observed = np.array([safe_float(ex["predict_observed_variance"]) for ex in series])
    cusp = np.array([safe_float(ex["predict_cusp_predicted"]) for ex in series])
    mixture = np.array([safe_float(ex["predict_mixture_predicted"]) for ex in series])
    fold = np.array([safe_float(ex["predict_fold_predicted"]) for ex in series])
    ddm = np.array([safe_float(ex["predict_ddm_predicted"]) for ex in series])

    # Get R^2 values from model_comparison_all_series
    model_comp = get_dataset(fit_data, "model_comparison_all_series")
    r2_vals = {"cusp": 0.0, "mixture": 0.0, "fold": 0.0, "ddm": 0.0}
    for ex in model_comp:
        if "arith__gemini-flash" in ex.get("input", ""):
            r2_vals["cusp"] = safe_float(ex.get("predict_cusp_R2_variance"))
            r2_vals["mixture"] = safe_float(ex.get("predict_mixture_R2"))
            r2_vals["fold"] = safe_float(ex.get("predict_fold_R2"))
            r2_vals["ddm"] = safe_float(ex.get("predict_ddm_R2_variance"))
            break

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

    ax.scatter(levels, observed, color="black", s=20, zorder=5, label="Observed",
               marker="o")
    ax.plot(levels, cusp, color=PALETTE[0], linewidth=1.2,
            label=f"Cusp (R\u00b2={r2_vals['cusp']:.2f})")
    ax.plot(levels, mixture, color=PALETTE[1], linewidth=1.2, linestyle="--",
            label=f"Mixture (R\u00b2={r2_vals['mixture']:.2f})")
    ax.plot(levels, fold, color=PALETTE[2], linewidth=1.2, linestyle=":",
            label=f"Fold (R\u00b2={r2_vals['fold']:.2f})")
    ax.plot(levels, ddm, color=PALETTE[3], linewidth=1.2, linestyle="-.",
            label=f"DDM (R\u00b2={r2_vals['ddm']:.2f})")

    ax.axvline(15, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(15.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.3,
            "d*=15", color="red", fontsize=5.5)

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Embedding Variance")
    ax.legend(fontsize=5.5, loc="best")

    fig.tight_layout()
    return fig, len(levels) * 5


def make_fig7(temp_data: dict) -> tuple[plt.Figure, int]:
    """Figure 7: Temperature Effect — variance + disagreement vs difficulty."""
    logger.info("Generating Figure 7: Temperature Effect")

    temps = [0.4, 0.7, 1.0, 1.3]
    temp_colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(temps)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL, 3.5), sharex=True)
    n_total = 0

    for i, temp in enumerate(temps):
        ds_name = f"csd_temp_T{temp}__gemini-2.0-flash-001"
        try:
            exs = get_dataset(temp_data, ds_name)
        except KeyError:
            logger.warning(f"  Dataset {ds_name} not found, skipping T={temp}")
            continue

        exs_sorted = sorted(exs, key=lambda x: x["metadata_difficulty_level"])
        levels = [ex["metadata_difficulty_level"] for ex in exs_sorted]
        variance = [safe_float(ex["predict_csd_variance"]) for ex in exs_sorted]
        disagree = [safe_float(ex["predict_disagreement_rate"]) for ex in exs_sorted]

        ax1.plot(levels, variance, color=temp_colors[i], linewidth=1.2,
                 marker="o", markersize=2, label=f"T={temp}")
        ax2.plot(levels, disagree, color=temp_colors[i], linewidth=1.2,
                 marker="o", markersize=2, label=f"T={temp}")
        n_total += len(levels) * 2

    ax1.set_ylabel("Embedding Variance")
    ax1.set_title("(a) Variance vs Difficulty", fontsize=8)
    ax1.legend(fontsize=5.5, ncol=2, loc="upper right")

    ax2.set_xlabel("Difficulty Level")
    ax2.set_ylabel("Disagreement Rate")
    ax2.set_title("(b) Disagreement vs Difficulty", fontsize=8)
    ax2.legend(fontsize=5.5, ncol=2, loc="upper right")

    fig.tight_layout()
    return fig, n_total


def make_fig8() -> tuple[plt.Figure, int]:
    """Figure 8: Prospective Protocol Schematic — horizontal flowchart."""
    logger.info("Generating Figure 8: Prospective Protocol Schematic")

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    ax.set_xlim(-0.2, 10.5)
    ax.set_ylim(-0.3, 3.3)
    ax.axis("off")

    # Boxes: (x, y, w, h, text, color)
    boxes = [
        (0.0, 1.0, 1.9, 1.0, "Sweep\nDifficulty", PALETTE[0]),
        (2.4, 1.0, 2.0, 1.0, "Generate N\nResponses", PALETTE[1]),
        (4.9, 1.0, 2.0, 1.0, "Compute CSD\nIndicators", PALETTE[2]),
        (7.5, 1.0, 1.6, 1.0, "Threshold\nTest", PALETTE[4]),
    ]

    for x, y, w, h, text, color in boxes:
        bbox = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=(*color[:3], 0.25), edgecolor=color, linewidth=1.3,
        )
        ax.add_patch(bbox)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=5.5, fontweight="bold")

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        y1 = boxes[i][1] + boxes[i][3] / 2
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + boxes[i + 1][3] / 2
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5),
        )

    # Decision output: Flag (red)
    flag_x, flag_y = 9.5, 2.5
    ax.text(flag_x, flag_y, "Flag:\nNear Boundary", fontsize=5, ha="center",
            color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffcccc",
                      edgecolor="red", alpha=0.6))
    ax.annotate(
        "", xy=(flag_x - 0.3, flag_y - 0.3),
        xytext=(boxes[-1][0] + boxes[-1][2] / 2, boxes[-1][1] + boxes[-1][3]),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.0),
    )

    # Decision output: Continue (green)
    cont_x, cont_y = 9.5, 0.3
    ax.text(cont_x, cont_y, "Continue", fontsize=5, ha="center",
            color="green", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ccffcc",
                      edgecolor="green", alpha=0.6))
    ax.annotate(
        "", xy=(cont_x - 0.3, cont_y + 0.2),
        xytext=(boxes[-1][0] + boxes[-1][2] / 2, boxes[-1][1]),
        arrowprops=dict(arrowstyle="-|>", color="green", lw=1.0),
    )

    # Annotations
    ax.text(5.9, 2.6, "variance  |  bimodality  |  dip", fontsize=4.5,
            ha="center", color="gray", fontstyle="italic",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

    fig.tight_layout()
    return fig, 0


# ──────────────────────────────────────────────
# 7. Eval output compilation
# ──────────────────────────────────────────────


def compile_eval_output(results: dict, stage: str) -> dict:
    """Build schema-compliant eval_out.json."""

    examples = []
    total_generated = 0
    all_pass_size = True
    all_neurips = True
    png_sizes: list[float] = []
    total_n_data_points = 0

    for fig_id, config in FIGURE_CONFIGS.items():
        fname = FIGURE_FILENAMES[fig_id]
        if fig_id in results:
            r = results[fig_id]
            generated = True
            total_generated += 1
            png_kb = r["png_kb"]
            pdf_kb = r["pdf_kb"]
            w = r["width"]
            h = r["height"]
            n_pts = r.get("n_data_points", 0)

            size_ok = 5 <= png_kb <= 2048
            if not size_ok:
                all_pass_size = False

            dim_ok = abs(w - config["target_w"]) < 0.5
            if not dim_ok:
                all_neurips = False

            png_sizes.append(png_kb)
            total_n_data_points += n_pts
        else:
            generated = False
            png_kb = pdf_kb = w = h = 0.0
            n_pts = 0

        example = {
            "input": f"Figure {fig_id[3:]}: {config['title']}",
            "output": "Generated successfully" if generated
                      else "Not generated (stage not reached)",
            "predict_figure_generated": str(generated).lower(),
            "predict_filesize_png_kb": f"{png_kb:.1f}",
            "predict_filesize_pdf_kb": f"{pdf_kb:.1f}",
            "predict_width_inches": f"{w:.2f}",
            "predict_height_inches": f"{h:.2f}",
            "predict_n_data_points": str(n_pts),
            "eval_figure_generated": 1 if generated else 0,
            "eval_filesize_png_kb": round(float(png_kb), 2),
            "eval_width_inches": round(float(w), 2),
            "eval_height_inches": round(float(h), 2),
            "eval_n_data_points": int(n_pts),
            "eval_size_check_pass": 1 if generated and 5 <= png_kb <= 2048 else 0,
            "eval_dimension_compliance": (
                1 if generated and abs(w - config["target_w"]) < 0.5 else 0
            ),
            "metadata_figure_id": fig_id,
            "metadata_figure_type": config["type"],
            "metadata_column_type": config["column"],
            "metadata_filepath_png": f"figures/{fname}.png" if generated else "",
            "metadata_filepath_pdf": f"figures/{fname}.pdf" if generated else "",
            "metadata_scaling_stage": config["stage"],
            "metadata_fold": "test",
        }
        examples.append(example)

    metrics_agg: dict[str, int | float] = {
        "total_figures_generated": total_generated,
        "total_figures_attempted": 8,
        "all_figures_pass_size_check": 1 if all_pass_size else 0,
        "neurips_dimension_compliance": 1 if all_neurips else 0,
        "scaling_stage_mini": 1 if stage in ("mini", "medium", "full") else 0,
        "scaling_stage_medium": 1 if stage in ("medium", "full") else 0,
        "scaling_stage_full": 1 if stage == "full" else 0,
        "total_data_points_plotted": total_n_data_points,
    }
    if png_sizes:
        metrics_agg["mean_filesize_png_kb"] = round(sum(png_sizes) / len(png_sizes), 2)
        metrics_agg["min_filesize_png_kb"] = round(min(png_sizes), 2)
        metrics_agg["max_filesize_png_kb"] = round(max(png_sizes), 2)

    return {
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "figure_generation_results",
                "examples": examples,
            }
        ],
        "metadata": {
            "evaluation_name": "CSD-LLM Publication Figures",
            "scaling_stage_reached": stage,
            "output_directory": "figures/",
            "style_config": {
                "dpi": DPI,
                "font_family": "serif",
                "palette": "seaborn colorblind",
                "single_col_width": SINGLE_COL,
                "double_col_width": DOUBLE_COL,
            },
        },
    }


def write_eval_out(results: dict, stage: str) -> None:
    """Write eval_out.json."""
    out = compile_eval_output(results, stage)
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Wrote {out_path} (stage={stage})")


# ──────────────────────────────────────────────
# 8. Main with gradual scaling
# ──────────────────────────────────────────────


@logger.catch
def main() -> None:
    start = time.time()
    logger.info(f"Starting figure generation: {NUM_CPUS} CPUs, "
                f"{TOTAL_RAM_GB:.1f} GB RAM")

    results: dict[str, dict] = {}

    # ── MINI STAGE (Figures 2, 3) ──────────────
    logger.info("=" * 60)
    logger.info("MINI STAGE: Generating Figures 2, 3")
    logger.info("=" * 60)

    arith_data = load_json(EXP1_IT2)
    gc_data = load_json(EXP3_IT2)

    try:
        fig, n = make_fig2(arith_data, gc_data)
        m = save_figure(fig, FIGURE_FILENAMES["fig2"])
        m["n_data_points"] = n
        results["fig2"] = m
    except Exception:
        logger.exception("Failed to generate Figure 2")

    try:
        fig, n = make_fig3(gc_data)
        m = save_figure(fig, FIGURE_FILENAMES["fig3"])
        m["n_data_points"] = n
        results["fig3"] = m
    except Exception:
        logger.exception("Failed to generate Figure 3")

    # Verify mini stage
    for fid in ["fig2", "fig3"]:
        if fid in results:
            r = results[fid]
            logger.info(f"  {fid}: OK  PNG={r['png_kb']:.0f}KB  "
                        f"{r['width']:.2f}x{r['height']:.2f}in  "
                        f"pts={r.get('n_data_points', 0)}")
        else:
            logger.error(f"  {fid}: FAILED")

    elapsed = time.time() - start
    logger.info(f"Mini stage done in {elapsed:.1f}s")
    write_eval_out(results, "mini")

    # ── MEDIUM STAGE (add Figures 1, 4, 5, 6) ─
    logger.info("=" * 60)
    logger.info("MEDIUM STAGE: Adding Figures 1, 4, 5, 6")
    logger.info("=" * 60)

    try:
        fig, n = make_fig1()
        m = save_figure(fig, FIGURE_FILENAMES["fig1"])
        m["n_data_points"] = n
        results["fig1"] = m
    except Exception:
        logger.exception("Failed to generate Figure 1")

    try:
        fig, n = make_fig4(gc_data)
        m = save_figure(fig, FIGURE_FILENAMES["fig4"])
        m["n_data_points"] = n
        results["fig4"] = m
    except Exception:
        logger.exception("Failed to generate Figure 4")

    # Free graph-coloring data after Fig 4
    del gc_data
    gc.collect()

    # Figure 5 — classifier comparison
    try:
        clf_data = load_json(EXP2_IT4)
        fig, n = make_fig5(clf_data)
        m = save_figure(fig, FIGURE_FILENAMES["fig5"])
        m["n_data_points"] = n
        results["fig5"] = m
        del clf_data
        gc.collect()
    except Exception:
        logger.exception("Failed to generate Figure 5")

    # Figure 6 — model fits
    try:
        fit_data = load_json(EXP3_IT4)
        fig, n = make_fig6(fit_data)
        m = save_figure(fig, FIGURE_FILENAMES["fig6"])
        m["n_data_points"] = n
        results["fig6"] = m
        del fit_data
        gc.collect()
    except Exception:
        logger.exception("Failed to generate Figure 6")

    for fid in ["fig1", "fig4", "fig5", "fig6"]:
        if fid in results:
            r = results[fid]
            logger.info(f"  {fid}: OK  PNG={r['png_kb']:.0f}KB  "
                        f"{r['width']:.2f}x{r['height']:.2f}in")
        else:
            logger.error(f"  {fid}: FAILED")

    elapsed = time.time() - start
    logger.info(f"Medium stage done in {elapsed:.1f}s")
    write_eval_out(results, "medium")

    # ── FULL STAGE (add Figures 7, 8) ──────────
    logger.info("=" * 60)
    logger.info("FULL STAGE: Adding Figures 7, 8")
    logger.info("=" * 60)

    try:
        temp_data = load_json(EXP3_IT3)
        fig, n = make_fig7(temp_data)
        m = save_figure(fig, FIGURE_FILENAMES["fig7"])
        m["n_data_points"] = n
        results["fig7"] = m
        del temp_data
        gc.collect()
    except Exception:
        logger.exception("Failed to generate Figure 7")

    try:
        fig, n = make_fig8()
        m = save_figure(fig, FIGURE_FILENAMES["fig8"])
        m["n_data_points"] = n
        results["fig8"] = m
    except Exception:
        logger.exception("Failed to generate Figure 8")

    # Free arithmetic data
    del arith_data
    gc.collect()

    for fid in ["fig7", "fig8"]:
        if fid in results:
            r = results[fid]
            logger.info(f"  {fid}: OK  PNG={r['png_kb']:.0f}KB  "
                        f"{r['width']:.2f}x{r['height']:.2f}in")
        else:
            logger.error(f"  {fid}: FAILED")

    # Final output
    write_eval_out(results, "full")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"ALL DONE in {elapsed:.1f}s")
    logger.info(f"Figures generated: {len(results)}/8")
    for fid in sorted(results.keys()):
        r = results[fid]
        logger.info(f"  {fid}: {r['png_kb']:.0f}KB PNG, "
                    f"{r.get('n_data_points', 0)} data points")
    logger.info(f"Output: {WORKSPACE / 'eval_out.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
