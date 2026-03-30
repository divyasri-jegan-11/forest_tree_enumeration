import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.density import analyze_tree_density
from src.detect import detect, calculate_ndvi, classify_vegetation_health
from src.report import generate_report
from src.route import get_optimized_route


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME_SHORT = "TreeSight"
PAGE_NAMES = ["Overview", "Detection", "Analysis", "Route & Report"]
CHART_SIZE_STANDARD = (6.2, 4.35)
CHART_SIZE_MAP = (6.2, 4.35)
CHART_DPI = 120
VEG_CLASS_COLORS = ["#efe7d7", "#a8d08d", "#2f7d58"]
ANALYSIS_DEFAULTS = {
    "confidence": 0.45,
    "overlap": 0.40,
    "grid_size": 6,
    "route_algorithm": "dijkstra",
    "start_x_percent": 5,
    "start_y_percent": 5,
    "end_x_percent": 95,
    "end_y_percent": 95,
}

st.set_page_config(
    page_title=f"{PROJECT_NAME_SHORT} Platform",
    page_icon="FI",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink:#10251c;
            --deep:#183a2c;
            --muted:#557066;
            --card:rgba(255,255,255,.88);
            --accent:#dd8a3a;
            --accent-soft:#f2b56f;
            --forest:#2f7d58;
            --route:#2f61d6;
        }
        .stApp {
            background: linear-gradient(180deg, #eef5ef 0%, #f7f2e9 52%, #fcfbf8 100%);
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        [data-testid="stElementToolbar"] {
            display: none !important;
        }
        button[title="View fullscreen"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        .hero {
            padding: 1.8rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(16,37,28,.96), rgba(43,106,75,.88));
            color:#fff;
            margin-bottom: 1rem;
            box-shadow: 0 18px 36px rgba(16,37,28,.12);
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: .18em;
            font-size: .75rem;
            opacity: .82;
        }
        .hero-title {
            font-size: 2.16rem;
            line-height: 1.05;
            margin: .45rem 0 .8rem;
            font-weight: 700;
        }
        .hero-copy {
            max-width: 860px;
            line-height: 1.66;
            font-size: .98rem;
        }
        .topnav-brand {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--deep);
            padding-top: .2rem;
        }
        .topnav-subtitle {
            font-size: .82rem;
            color: var(--muted);
            margin-top: .15rem;
        }
        .card {
            background: var(--card);
            border: 1px solid rgba(16,37,28,.1);
            border-radius: 22px;
            padding: 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(16,37,28,.05);
        }
        .feature {
            background: rgba(255,255,255,.74);
            border: 1px solid rgba(16,37,28,.08);
            border-radius: 20px;
            padding: .95rem 1rem;
            min-height: 122px;
        }
        .feature h4, .metric h4, .insight h4, .soft-panel h4 {
            margin: 0 0 .55rem;
            font-size: 1rem;
            line-height: 1.35;
        }
        .feature p, .metric div, .insight p, .soft-caption {
            font-size: .92rem;
            line-height: 1.55;
        }
        .metric {
            background: rgba(255,255,255,.92);
            border: 1px solid rgba(16,37,28,.08);
            border-radius: 18px;
            padding: .9rem .95rem;
            min-height: 98px;
        }
        .metric-value {
            font-size: 1.28rem;
            font-weight: 700;
            color: var(--deep);
            margin-bottom: .18rem;
        }
        .insight {
            background: rgba(255,255,255,.92);
            border-left: 6px solid var(--accent);
            border-radius: 18px;
            padding: .9rem .95rem;
            margin-bottom: .75rem;
            min-height: 160px;
        }
        .section {
            font-size: 1.18rem;
            font-weight: 700;
            color: var(--deep);
            margin-bottom: .55rem;
        }
        .section-note {
            color: var(--muted);
            margin-bottom: .5rem;
            font-size: .95rem;
            min-height: 2.9rem;
        }
        .stDownloadButton > button, .stButton > button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #1c4a35, #2f7d58);
            color: #fff;
            font-weight: 600;
            min-height: 2.7rem;
            padding: .48rem 1.05rem;
            transition: background .18s ease, transform .18s ease, box-shadow .18s ease;
        }
        .stDownloadButton > button:hover, .stButton > button:hover {
            background: linear-gradient(135deg, #236342, #3a9a69);
            color: #fff;
            transform: translateY(-1px);
            box-shadow: 0 10px 18px rgba(28,74,53,.12);
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            width: 100%;
            border-radius: 999px;
            background: rgba(255,255,255,.78);
            border: 1px solid rgba(16,37,28,.12);
            color: var(--deep);
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #236342, #3a9a69);
            border-color: transparent;
            color: #fff;
            transform: translateY(-1px);
            box-shadow: 0 10px 18px rgba(28,74,53,.12);
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            width: 100%;
            border-radius: 999px;
        }
        .soft-panel {
            background: rgba(255,255,255,.62);
            border: 1px solid rgba(16,37,28,.08);
            border-radius: 22px;
            padding: .95rem 1rem;
            margin-bottom: .8rem;
        }
        .soft-caption {
            color: var(--muted);
            font-size: .92rem;
            line-height: 1.55;
        }
        .chart-caption {
            color: var(--muted);
            font-size: .9rem;
            line-height: 1.5;
            margin-top: .18rem;
        }
        .continue-callout {
            margin-top: .7rem;
            padding: .9rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,.72);
            border: 1px solid rgba(16,37,28,.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(kicker, title, copy):
    st.markdown(
        f"""
        <section class='hero'>
            <div class='hero-kicker'>{kicker}</div>
            <div class='hero-title'>{title}</div>
            <div class='hero-copy'>{copy}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section(title, note=None):
    st.markdown(f"<div class='section'>{title}</div>", unsafe_allow_html=True)
    if note:
        st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)


def feature_card(title, copy):
    st.markdown(f"<div class='feature'><h4>{title}</h4><p>{copy}</p></div>", unsafe_allow_html=True)


def metric_card(title, value, copy):
    st.markdown(
        f"<div class='metric'><h4>{title}</h4><div class='metric-value'>{value}</div><div>{copy}</div></div>",
        unsafe_allow_html=True,
    )


def insight_card(title, copy):
    st.markdown(f"<div class='insight'><h4>{title}</h4><p>{copy}</p></div>", unsafe_allow_html=True)


def soft_panel(title, copy):
    st.markdown(
        f"<div class='soft-panel'><h4>{title}</h4><div class='soft-caption'>{copy}</div></div>",
        unsafe_allow_html=True,
    )


def callout_panel(copy):
    st.markdown(f"<div class='continue-callout'><div class='soft-caption'>{copy}</div></div>", unsafe_allow_html=True)


def go_to_page(page_name):
    st.session_state["page"] = page_name
    st.rerun()


def top_navigation():
    if "page" not in st.session_state or st.session_state["page"] not in PAGE_NAMES:
        st.session_state["page"] = "Overview"

    brand_col, nav_1, nav_2, nav_3, nav_4 = st.columns([1.35, 1.0, 1.0, 1.0, 1.15])
    with brand_col:
        st.markdown(
            f"<div class='topnav-brand'>{PROJECT_NAME_SHORT}</div><div class='topnav-subtitle'>Tree enumeration platform</div>",
            unsafe_allow_html=True,
        )
    nav_cols = [nav_1, nav_2, nav_3, nav_4]
    for col, page_name in zip(nav_cols, PAGE_NAMES):
        with col:
            if st.button(
                page_name,
                type="primary" if st.session_state["page"] == page_name else "secondary",
                use_container_width=True,
            ):
                st.session_state["page"] = page_name
    st.markdown("<div style='height:.2rem'></div>", unsafe_allow_html=True)
    return st.session_state["page"]


def save_uploaded_image(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    upload_path = PROJECT_ROOT / f"temp_upload{suffix}"
    with open(upload_path, "wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())
    return upload_path


def percent_to_point(image_shape, x_percent, y_percent):
    height, width = image_shape[:2]
    x = int(round((x_percent / 100.0) * max(width - 1, 0)))
    y = int(round((y_percent / 100.0) * max(height - 1, 0)))
    return x, y


def simplify_route_points(points):
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    previous_dx = points[1][0] - points[0][0]
    previous_dy = points[1][1] - points[0][1]
    for index in range(1, len(points) - 1):
        current_dx = points[index + 1][0] - points[index][0]
        current_dy = points[index + 1][1] - points[index][1]
        if (current_dx, current_dy) != (previous_dx, previous_dy):
            simplified.append(points[index])
        previous_dx, previous_dy = current_dx, current_dy
    simplified.append(points[-1])
    return simplified

def clean_route_visualization(image_rgb, route_points, waypoints=None):
    """
    Clean grayscale route visualization for decision making
    Shows full A* computed route that avoids trees
    """

    import cv2
    import numpy as np

    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Lighten image (fade effect)
    gray = cv2.convertScaleAbs(gray, alpha=0.6, beta=40)

    # Convert to color
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Always use full route_points (A* path) to avoid trees, not just waypoints
    if len(route_points) > 1:
        pts = np.array(route_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, (0, 255, 0), 1, cv2.LINE_AA)  # thickness=1 for thin line

    return canvas


def draw_dashed_line(image, start_point, end_point, color, thickness=4, dash_length=18, gap_length=10):
    start = np.array(start_point, dtype=float)
    end = np.array(end_point, dtype=float)
    line_vector = end - start
    line_length = np.linalg.norm(line_vector)
    if line_length == 0:
        return
    direction = line_vector / line_length
    drawn_length = 0.0
    while drawn_length < line_length:
        dash_start = start + direction * drawn_length
        dash_end = start + direction * min(drawn_length + dash_length, line_length)
        cv2.line(
            image,
            tuple(np.round(dash_start).astype(int)),
            tuple(np.round(dash_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        drawn_length += dash_length + gap_length


def ndvi_route_overlay(ndvi_normalized, route_points, image_shape):
    """
    Create route visualization overlaid on NDVI heatmap.
    Shows the optimal path traced on vegetation health map (like reference image).
    Red path on NDVI background for better decision-making visualization.
    """
    height, width = image_shape[:2]
    
    # Create NDVI visualization with color gradient
    ndvi_colored = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            val = ndvi_normalized[i, j]
            # Red-Yellow-Green gradient for low-to-high vegetation
            if val < 0.33:  # Low: Red channel
                ndvi_colored[i, j] = [100, 100, int(255 * (1 - val / 0.33))]
            elif val < 0.66:  # Medium: Yellow
                ndvi_colored[i, j] = [0, int(255 * ((val - 0.33) / 0.33)), 255]
            else:  # High: Green
                ndvi_colored[i, j] = [0, 255, int(255 * (1 - (val - 0.66) / 0.34))]
    
    route_image = ndvi_colored.copy()
    route_simplified = simplify_route_points(route_points)
    
    # Draw thick white outline for route visibility
    for start_pt, end_pt in zip(route_simplified, route_simplified[1:]):
        cv2.line(route_image, start_pt, end_pt, (255, 255, 255), 5, cv2.LINE_AA)
    
    # Draw red/orange main route path
    for start_pt, end_pt in zip(route_simplified, route_simplified[1:]):
        cv2.line(route_image, start_pt, end_pt, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Mark start and end points
    if route_simplified:
        start = route_simplified[0]
        end = route_simplified[-1]
        
        # Start: Green circle
        cv2.circle(route_image, start, 9, (0, 255, 0), -1)
        cv2.circle(route_image, start, 9, (255, 255, 255), 2)
        
        # End: Blue circle
        cv2.circle(route_image, end, 9, (255, 0, 0), -1)
        cv2.circle(route_image, end, 9, (255, 255, 255), 2)
    
    return route_image


def density_aware_route_viz(base_image, density_analysis, route_points):
    """
    Enhanced route visualization with density zones and clean path overlay.
    Better contrast suitable for forest official decision-making.
    """
    route_image = base_image.copy()
    overlay = route_image.copy()
    
    zone_colors = {
        "High Density": (50, 0, 255),
        "Medium Density": (0, 165, 255),
        "Low Density": (0, 255, 255),
        "No Trees": (200, 200, 200),
    }
    
    # Draw density zones
    for zone in density_analysis["zones"]:
        x1, y1, x2, y2 = zone["bounds"]
        if zone["density_level"] != "No Trees":
            cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_colors[zone["density_level"]], -1)
        cv2.rectangle(route_image, (x1, y1), (x2, y2), (150, 150, 150), 2)
        
        if zone["tree_count"] > 0:
            cv2.putText(route_image, f"{zone['zone_id']}:{zone['tree_count']}",
                       (x1 + 6, min(y2 - 8, y1 + 18)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.addWeighted(overlay, 0.12, route_image, 0.88, 0, route_image)
    
    # Draw route with better visibility
    route_simplified = simplify_route_points(route_points)
    
    for start_pt, end_pt in zip(route_simplified, route_simplified[1:]):
        cv2.line(route_image, start_pt, end_pt, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.line(route_image, start_pt, end_pt, (230, 122, 34), 2, cv2.LINE_AA)
    
    if route_simplified:
        start = route_simplified[0]
        end = route_simplified[-1]
        cv2.circle(route_image, start, 7, (46, 138, 87), -1)
        cv2.circle(route_image, end, 7, (23, 94, 207), -1)
    
    return route_image


def create_route_canvas(base_image, tree_points, density_analysis, route_result):
    route_image = base_image.copy()
    overlay = route_image.copy()
    zone_fill_colors = {
        "High Density": (120, 150, 235),
        "Medium Density": (155, 208, 245),
        "Low Density": (170, 218, 183),
        "No Trees": (236, 236, 236),
    }
    zone_line_colors = {
        "High Density": (85, 105, 185),
        "Medium Density": (88, 152, 199),
        "Low Density": (86, 150, 109),
        "No Trees": (190, 190, 190),
    }

    for zone in density_analysis["zones"]:
        x1, y1, x2, y2 = zone["bounds"]
        if zone["density_level"] != "No Trees":
            cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_fill_colors[zone["density_level"]], -1)
        cv2.rectangle(route_image, (x1, y1), (x2, y2), zone_line_colors[zone["density_level"]], 1)
        if zone["tree_count"] > 0:
            cv2.putText(
                route_image,
                f"{zone['zone_id']}:{zone['tree_count']}",
                (x1 + 6, min(y2 - 8, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (48, 58, 52),
                1,
                cv2.LINE_AA,
            )

    cv2.addWeighted(overlay, 0.08, route_image, 0.92, 0, route_image)

    # Use full route points (A* computed path) to preserve tree avoidance detail
    route_points = route_result["route_points"]

    # Improved visualization: white halo with red core (like screenshot style)
    if len(route_points) > 1:
        pts = np.array(route_points, dtype=np.int32).reshape((-1, 1, 2))
        overlay2 = route_image.copy()

        # Halo/backlayer
        cv2.polylines(overlay2, [pts], False, (255, 255, 255), 8, cv2.LINE_AA)
        # Main route line
        cv2.polylines(overlay2, [pts], False, (0, 0, 255), 2, cv2.LINE_AA)

        route_image = cv2.addWeighted(overlay2, 0.75, route_image, 0.25, 0)

    return route_image


def vegetation_index_map(image_rgb):
    image = image_rgb.astype(np.float32) / 255.0
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    exg = 2 * green - red - blue
    normalized = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    return normalized


def ndvi_visualization_map(ndvi_normalized):
    """Create a visualization of NDVI values with custom colormap."""
    return ndvi_normalized


def vegetation_classification(veg_index):
    classes = np.zeros_like(veg_index, dtype=np.uint8)
    classes[(veg_index >= 0.33) & (veg_index < 0.66)] = 1
    classes[veg_index >= 0.66] = 2
    labels = ["Sparse", "Moderate", "Dense"]
    counts = [int(np.sum(classes == idx)) for idx in range(3)]
    total = max(int(classes.size), 1)
    shares = [(count / total) * 100 for count in counts]
    return {
        "classes": classes,
        "labels": labels,
        "counts": counts,
        "shares": shares,
    }


def vegetation_index_chart(veg_classes):
    fig, ax = plt.subplots(figsize=CHART_SIZE_MAP, dpi=CHART_DPI)
    cmap = ListedColormap(VEG_CLASS_COLORS)
    ax.imshow(veg_classes, cmap=cmap, vmin=0, vmax=2)
    ax.set_aspect("auto")
    ax.set_xlabel("Image column", fontsize=10)
    ax.set_ylabel("Image row", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    legend_items = [
        Patch(facecolor=VEG_CLASS_COLORS[0], edgecolor="none", label="Sparse"),
        Patch(facecolor=VEG_CLASS_COLORS[1], edgecolor="none", label="Moderate"),
        Patch(facecolor=VEG_CLASS_COLORS[2], edgecolor="none", label="Dense"),
    ]
    ax.legend(
        handles=legend_items,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        fontsize=8.5,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    return fig


def vegetation_histogram_chart(veg_stats):
    fig, ax = plt.subplots(figsize=CHART_SIZE_STANDARD, dpi=CHART_DPI)
    labels = veg_stats["labels"]
    values = veg_stats["shares"]
    counts = veg_stats["counts"]
    colors = VEG_CLASS_COLORS
    bars = ax.bar(labels, values, color=colors, edgecolor="#183a2c", linewidth=1.1, width=0.62)
    ax.set_xlabel("Vegetation class", fontsize=10)
    ax.set_ylabel("Share of image (%)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for bar, value, count in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.1f}%\n{count} px",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold",
        )
    fig.tight_layout()
    return fig


def ndvi_heatmap_chart(ndvi_array):
    """Create a heatmap visualization of NDVI values."""
    fig, ax = plt.subplots(figsize=CHART_SIZE_MAP, dpi=CHART_DPI)
    heatmap = ax.imshow(ndvi_array, cmap="RdYlGn", interpolation="bilinear", vmin=0, vmax=1)
    ax.set_aspect("auto")
    ax.set_xlabel("Image column", fontsize=10)
    ax.set_ylabel("Image row", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label="NDVI (Vegetation Health)")
    cbar.set_label("NDVI (0=Dead, 1=Healthy)", fontsize=9)
    fig.tight_layout()
    return fig


def ndvi_statistics_chart(veg_stats):
    """Display NDVI statistics in a clear format."""
    fig, ax = plt.subplots(figsize=CHART_SIZE_STANDARD, dpi=CHART_DPI)
    
    labels = veg_stats["labels"]
    values = veg_stats["shares"]
    colors = ["#efe7d7", "#a8d08d", "#2f7d58"]
    
    bars = ax.bar(labels, values, color=colors, edgecolor="#183a2c", linewidth=1.2, width=0.65)
    ax.set_ylabel("Share of image (%)", fontsize=10)
    ax.set_title("NDVI-based Vegetation Distribution", fontsize=11, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(0, max(values) * 1.15)
    
    for bar, value, count in zip(bars, values, veg_stats["counts"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )
    
    fig.tight_layout()
    return fig


def estimate_environmental_metrics(tree_count, area_hectares, plantation_ratio, carbon_kg_per_tree):
    trees_per_hectare = None
    if area_hectares and area_hectares > 0:
        trees_per_hectare = tree_count / area_hectares
    return {
        "trees_per_hectare": trees_per_hectare,
        "carbon_tonnes": (tree_count * carbon_kg_per_tree) / 1000.0,
        "plantation_count": int(np.ceil(tree_count * plantation_ratio)),
    }


def distribution_metrics(density_analysis):
    zone_counts = density_analysis["zone_counts"]
    total_cells = density_analysis["grid_counts"].size
    return {
        "clear": zone_counts["No Trees"],
        "light": zone_counts["Low Density"],
        "medium": zone_counts["Medium Density"],
        "dense": zone_counts["High Density"],
        "occupied": density_analysis["occupied_cells"],
        "occupied_share": density_analysis["occupied_share"],
        "max_density": float(density_analysis["max_trees_in_zone"]) if total_cells else 0.0,
    }


def distribution_chart(data):
    fig, ax = plt.subplots(figsize=CHART_SIZE_STANDARD, dpi=CHART_DPI)
    labels = ["Clear", "Light", "Medium", "Dense"]
    values = [data["clear"], data["light"], data["medium"], data["dense"]]
    colors = ["#edf2eb", "#a8cfae", "#f0be6f", "#cb7455"]
    total = max(sum(values), 1)
    bars = ax.bar(labels, values, color=colors, edgecolor="#183a2c", linewidth=1.2, width=0.62)
    ax.set_ylabel("Grid cells", fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        percentage = (value / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.012, 2),
            f"{value}\n{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold",
        )
    fig.tight_layout()
    return fig


def occupancy_chart(data):
    fig, ax = plt.subplots(figsize=CHART_SIZE_STANDARD, dpi=CHART_DPI)
    open_cells = max(data["clear"], 0)
    occupied_cells = max(data["occupied"], 0)
    ax.pie(
        [open_cells, occupied_cells],
        labels=None,
        autopct=None,
        colors=["#efe7d7", "#2f7d58"],
        wedgeprops={"width": 0.34, "edgecolor": "white"},
        radius=0.92,
    )
    occupied_share = (occupied_cells / max(open_cells + occupied_cells, 1)) * 100
    ax.text(0, 0, f"{occupied_share:.1f}%\noccupied", ha="center", va="center", fontsize=13, fontweight="semibold", color="#183a2c")
    ax.legend(
        ["Open area", "Tree occupied"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    return fig


def site_composition_chart(distribution, veg_stats):
    fig, ax = plt.subplots(figsize=CHART_SIZE_STANDARD, dpi=CHART_DPI)
    labels = ["Open grid", "Tree occupied", "Sparse veg", "Moderate veg", "Dense veg"]
    values = [
        max(100 - distribution["occupied_share"], 0),
        distribution["occupied_share"],
        veg_stats["shares"][0],
        veg_stats["shares"][1],
        veg_stats["shares"][2],
    ]
    colors = ["#efe7d7", "#2f7d58", VEG_CLASS_COLORS[0], VEG_CLASS_COLORS[1], VEG_CLASS_COLORS[2]]
    y_positions = np.arange(len(labels))
    ax.barh(y_positions, values, color=colors, edgecolor="#183a2c", linewidth=1.0)
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share (%)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for y_position, value in zip(y_positions, values):
        ax.text(min(value + 1.4, 98), y_position, f"{value:.1f}%", va="center", fontsize=8.5, fontweight="semibold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def image_preview_figure(image_rgb):
    fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=CHART_DPI)
    ax.imshow(image_rgb)
    ax.set_aspect("auto")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def heatmap_chart(density_grid):
    fig, ax = plt.subplots(figsize=CHART_SIZE_MAP, dpi=CHART_DPI)
    heatmap = ax.imshow(density_grid, cmap="YlGn", interpolation="nearest")
    ax.set_aspect("auto")
    ax.set_xlabel("Grid column", fontsize=10)
    ax.set_ylabel("Grid row", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label="Trees per grid cell")
    fig.tight_layout()
    return fig



def run_analysis(
    uploaded_file,
    project_name,
    site_name,
    image_source,
    survey_area,
    plantation_ratio,
    carbon_per_tree,
    confidence,
    overlap,
    grid_size,
    route_algorithm,
    start_x_percent,
    start_y_percent,
    end_x_percent,
    end_y_percent,
):
    upload_path = save_uploaded_image(uploaded_file)
    original_bgr = cv2.imread(str(upload_path))
    if original_bgr is None:
        raise ValueError("The uploaded image could not be read.")

    st.write("🚀 Running detection...")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    detection_result = detect(
        str(upload_path),
        conf=0.25,
        iou=0.3,
        show_labels=False,
        show_centers=False,
    )
    annotated_image = detection_result["annotated_image"]
    density_analysis = analyze_tree_density(
        detection_result["tree_points"],
        annotated_image.shape,
        grid_size=grid_size,
        area_hectares=survey_area,
    )
    density_summary = density_analysis["summary"]
    start_point = percent_to_point(original_bgr.shape, start_x_percent, start_y_percent)
    end_point = percent_to_point(original_bgr.shape, end_x_percent, end_y_percent)
    route_result = get_optimized_route(
        detection_result["tree_points"],
        annotated_image.shape,
        density_analysis=density_analysis,
        start_point=start_point,
        end_point=end_point,
        algorithm=route_algorithm,
    )
    route_image = clean_route_visualization(
        original_rgb,
        route_result["route_points"],
        waypoints=route_result.get("ordered_points", [])
    )
    impact = estimate_environmental_metrics(detection_result["count"], survey_area, plantation_ratio, carbon_per_tree)
    carbon_label = f"{impact['carbon_tonnes']:.2f} tonnes CO2e"
    plantation_label = f"{impact['plantation_count']} saplings"
    impact_summary = (
        f"For this run, the platform estimates {carbon_label} of associated carbon stock and recommends "
        f"{plantation_label} based on a plantation ratio of {plantation_ratio:.1f}:1. "
        "These are planning estimates and should be validated against field surveys and policy requirements."
    )
    distribution = distribution_metrics(density_analysis)
    
    # Calculate NDVI for vegetation health analysis
    ndvi_normalized, ndvi_raw = calculate_ndvi(original_rgb)
    veg_stats = classify_vegetation_health(ndvi_normalized)
    
    route_distance_label = f"{route_result['total_distance']:.1f} px"
    inspection_order_text = ", ".join(density_analysis["inspection_order"][:6]) if density_analysis["inspection_order"] else "No occupied zones"
    top_zone = density_analysis["highest_density_zone"]
    hotspot_label = top_zone["zone_id"] if top_zone else "None"
    hotspot_summary = (
        f"Zone {top_zone['zone_id']} ({top_zone['density_level']}) contains {top_zone['tree_count']} detected trees."
        if top_zone
        else "No hotspot zone detected."
    )
    report_bytes = generate_report(
        {
            "project_name": project_name,
            "site_name": site_name,
            "image_source": image_source,
            "survey_area_label": f"{survey_area:.2f} hectares" if survey_area > 0 else "Not specified",
            "tree_count": detection_result["count"],
            "density_label": density_summary["label"],
            "density_score": density_summary["score"],
            "density_score_label": density_summary["score_label"],
            "average_confidence": detection_result["average_confidence"],
            "route_steps": route_result["segments"],
            "model_name": detection_result["model_name"],
            "density_description": density_summary["description"],
            "recommendation": " ".join(density_analysis["recommendations"][:2]),
            "carbon_impact_label": carbon_label,
            "plantation_label": plantation_label,
            "impact_summary": impact_summary,
            "occupied_share_label": f"{distribution['occupied_share']:.1f}%",
            "trees_per_hectare_label": f"{impact['trees_per_hectare']:.2f}" if impact["trees_per_hectare"] is not None else "Not specified",
            "veg_sparse_label": f"{veg_stats['shares'][0]:.1f}%",
            "veg_moderate_label": f"{veg_stats['shares'][1]:.1f}%",
            "veg_dense_label": f"{veg_stats['shares'][2]:.1f}%",
            "route_steps_label": f"{route_result['segments']} segments | {route_distance_label}",
            "route_method_label": route_result["method"],
        }
    )
    st.session_state["analysis"] = {
        "project_name": project_name,
        "site_name": site_name,
        "image_source": image_source,
        "original_rgb": original_rgb,
        "detected_rgb": cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        "route_rgb": cv2.cvtColor(route_image, cv2.COLOR_BGR2RGB),
        "tree_count": detection_result["count"],
        "detections": detection_result["detections"],
        "density_summary": density_summary,
        "density_analysis": density_analysis,
        "avg_confidence": detection_result["average_confidence"],
        "route_steps": route_result["segments"],
        "route_result": route_result,
        "route_distance_label": route_distance_label,
        "density_grid": density_analysis["grid_counts"],
        "distribution": distribution,
        "carbon_label": carbon_label,
        "plantation_label": plantation_label,
        "impact_summary": impact_summary,
        "report_bytes": report_bytes,
        "model_name": detection_result["model_name"],
        "model_path": detection_result["model_path"],
        "trees_per_hectare": impact["trees_per_hectare"],
        "ndvi_normalized": ndvi_normalized,
        "ndvi_raw": ndvi_raw,
        "recommendations": density_analysis["recommendations"],
        "inspection_order_text": inspection_order_text,
        "start_point": start_point,
        "end_point": end_point,
        "route_method": route_result["method"],
        "hotspot_label": hotspot_label,
        "hotspot_summary": hotspot_summary,
        "priority_zones": density_analysis["hotspot_zones"],
    }
    st.session_state["analysis"]["veg_stats"] = veg_stats
    st.session_state["analysis"]["ndvi_normalized"] = ndvi_normalized
    st.session_state["analysis"]["ndvi_raw"] = ndvi_raw
    st.session_state["analysis"]["mean_ndvi"] = veg_stats["mean_ndvi"]
    st.session_state["analysis"]["max_ndvi"] = veg_stats["max_ndvi"]
    st.session_state["analysis"]["min_ndvi"] = veg_stats["min_ndvi"]


def require_analysis():
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Run an image through the Detection page first.")
        return None
    return analysis


def home_page():
    hero(
        "SIH1316 tree enumeration platform",
        "TreeSight for forest land diversion analysis.",
        "TreeSight is a streamlined web platform for detecting trees from satellite, drone, or aerial imagery, estimating density, and generating route and report outputs for forest land diversion workflows.",
    )

    cols = st.columns(3)
    items = [
        ("Automated tree enumeration", "Detect visible trees from uploaded forest imagery and produce count-ready outputs."),
        ("Density and impact assessment", "Review hotspot density, occupancy, carbon impact, and plantation requirements."),
        ("Diversion planning support", "Generate route overlays and downloadable reports for decision-making workflows."),
    ]
    for col, (title, copy) in zip(cols, items):
        with col:
            feature_card(title, copy)

    st.markdown("<div style='height:1.05rem'></div>", unsafe_allow_html=True)

    analysis = st.session_state.get("analysis")
    dashboard_header, dashboard_action = st.columns([3.2, 1.2], vertical_alignment="center")
    with dashboard_header:
        section("Dashboard overview", "A compact summary of the latest processed site.")
    with dashboard_action:
        if analysis:
            st.download_button(
                label="Download report",
                data=analysis["report_bytes"],
                file_name="treesight_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=False,
            )

    if not analysis:
        section("Workflow")
        flow_cols = st.columns(3)
        with flow_cols[0]:
            feature_card("1. Upload and detect", "Start with one satellite, drone, or aerial image and run the tree detection workflow.")
        with flow_cols[1]:
            feature_card("2. Review analysis", "Open the analysis page to inspect density hotspots, occupancy, and heatmap outputs.")
        with flow_cols[2]:
            feature_card("3. Export report", "Generate the optimized route view and download the final report for planning use.")
        if st.button("Start detection", type="primary", use_container_width=True):
            go_to_page("Detection")
        st.info("Run one image through the Detection page and the homepage will automatically show the latest dashboard, including metrics, density hotspots, route preview, and report access.")
        return

    section("Workflow", "Move through detection, analysis, and final route/report review using the shortcuts below.")
    flow_cols = st.columns(3)
    with flow_cols[0]:
        feature_card("Detection complete", f"{analysis['tree_count']} trees detected for the latest uploaded site.")
    with flow_cols[1]:
        feature_card("Analysis ready", f"{analysis['density_summary']['label']} with {analysis['distribution']['occupied_share']:.1f}% occupied grid area.")
    with flow_cols[2]:
        feature_card("Report ready", "Open the final route and report page to review the route and export the PDF.")

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("Open detection", use_container_width=True):
            go_to_page("Detection")
    with action_cols[1]:
        if st.button("Open analysis", use_container_width=True):
            go_to_page("Analysis")
    with action_cols[2]:
        if st.button("Open route and report", use_container_width=True):
            go_to_page("Route & Report")

    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)

    metrics = st.columns(6)
    dashboard_metrics = [
        ("Tree count", analysis["tree_count"], "Latest detected count."),
        ("Density level", analysis["density_summary"]["label"], analysis["density_summary"]["score_label"]),
        ("Tree occupancy", f"{analysis['distribution']['occupied_share']:.1f}%", "Share of occupied grid cells."),
        ("Primary hotspot", analysis["hotspot_label"], "Highest-density inspection zone."),
        ("Route length", analysis["route_distance_label"], analysis["route_method"]),
        ("Carbon impact", analysis["carbon_label"], "Estimated planning impact."),
    ]
    for col, (title, value, copy) in zip(metrics, dashboard_metrics):
        with col:
            metric_card(title, value, copy)

    summary_row = st.columns(3)
    with summary_row[0]:
        section("Density snapshot", "How many grid cells fall into clear, light, medium, and dense canopy classes.")
        fig = distribution_chart(analysis["distribution"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with summary_row[1]:
        section("Occupancy share", "Open-area split against grid area occupied by detected tree presence.")
        fig = occupancy_chart(analysis["distribution"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with summary_row[2]:
        section("Route planning preview", "Suggested alignment across the site image with a cleaner optimal path overlay.")
        fig = image_preview_figure(analysis["route_rgb"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    section("Summary insights")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        insight_card("Density interpretation", analysis["density_summary"]["description"])
    with summary_cols[1]:
        insight_card("Inspection order", analysis["inspection_order_text"])
    with summary_cols[2]:
        insight_card("Environmental impact", analysis["impact_summary"])


def upload_detection_page():
    hero(
        "Detection",
        "Upload an image and run tree detection.",
        "This page keeps the first step simple for normal users. Enter the site details, upload the image, and run the model. The results remain available across the rest of the website.",
    )

    left, right = st.columns([0.92, 1.08])
    with left:
        section("Upload image")
        project_name = st.text_input("Project name", value="Tree Enumeration Assessment")
        site_name = st.text_input("Site name", value="Forest Block A")
        image_source = st.selectbox("Image source", ["Satellite imagery", "Drone imagery", "Aerial photograph", "Other"])
        survey_area = st.number_input("Survey area (hectares)", min_value=0.0, value=1.0, step=0.5)
        route_algorithm = st.selectbox(
            "Route algorithm",
            options=["auto", "nearest", "two_opt", "exact", "coverage"],
            index=["auto", "nearest", "two_opt", "exact", "coverage"].index(ANALYSIS_DEFAULTS["route_algorithm"] if ANALYSIS_DEFAULTS["route_algorithm"] in ["auto", "nearest", "two_opt", "exact", "coverage"] else "auto"),
            help="auto: small sets exact, larger sets 2-opt; nearest: greedy; two_opt: greedy + 2-opt; exact: brute force; coverage: full zone coverage sweep.",
        )
        uploaded_file = st.file_uploader("Upload a satellite or aerial image", type=["jpg", "jpeg", "png"])
        plantation_ratio = 3.0
        carbon_per_tree = 21.0
        if st.button("Run detection", use_container_width=True):
            if uploaded_file is None:
                st.warning("Upload an image first.")
            else:
                try:
                    with st.spinner("🌳 Analyzing forest... please wait"):
                        run_analysis(
                            uploaded_file,
                            project_name,
                            site_name,
                            image_source,
                            survey_area,
                            plantation_ratio,
                            carbon_per_tree,
                            ANALYSIS_DEFAULTS["confidence"],
                            ANALYSIS_DEFAULTS["overlap"],
                            ANALYSIS_DEFAULTS["grid_size"],
                            route_algorithm,
                            ANALYSIS_DEFAULTS["start_x_percent"],
                            ANALYSIS_DEFAULTS["start_y_percent"],
                            ANALYSIS_DEFAULTS["end_x_percent"],
                            ANALYSIS_DEFAULTS["end_y_percent"],
                        )
                    st.success("Detection completed. Review the output below, then continue to Analysis.")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
        st.caption("Analysis now runs with fixed tuned defaults so every image uses the same detection, density, and route settings.")

    with right:
        section("Detection preview", "The detection image is kept simple: only bounding boxes and the total detected count are shown.")
        analysis = require_analysis()
        if analysis:
            preview_cols = st.columns(2)
            with preview_cols[0]:
                st.image(analysis["original_rgb"], width=400)
                st.markdown("<div class='chart-caption'>Uploaded image</div>", unsafe_allow_html=True)
            with preview_cols[1]:
                st.image(analysis["detected_rgb"], width=400)
                st.markdown("<div class='chart-caption'>Detection output</div>", unsafe_allow_html=True)
            summary_cols = st.columns(4)
            with summary_cols[0]:
                metric_card("Trees detected", analysis["tree_count"], "Latest detection result.")
            with summary_cols[1]:
                metric_card("Average confidence", f"{analysis['avg_confidence'] * 100:.1f}%", "Model confidence across detections.")
            with summary_cols[2]:
                metric_card("Highest-density zone", analysis["hotspot_label"], analysis["hotspot_summary"])
            with summary_cols[3]:
                metric_card("Route method", analysis["route_method"], "Optimization strategy used for inspection planning.")
            continue_cols = st.columns([2.3, 1.0], vertical_alignment="center")
            with continue_cols[0]:
                callout_panel("Detection output is ready. Continue to the analysis page to inspect density zones, hotspot recommendations, and heatmap results.")
            with continue_cols[1]:
                if st.button("Continue to analysis", type="primary", use_container_width=True):
                    go_to_page("Analysis")


def count_density_page():
    hero(
        "Analysis",
        "Review the count, density, and occupancy outputs.",
        "This page presents the analytical outcome in a more decision-oriented layout so the charts and metrics are easier to read at a glance.",
    )
    analysis = require_analysis()
    if not analysis:
        return

    metrics = st.columns(4)
    data = [
        ("Tree count", analysis["tree_count"], "Detected trees across the uploaded image."),
        ("Density level", analysis["density_summary"]["label"], analysis["density_summary"]["score_label"]),
        ("Avg NDVI", f"{analysis['mean_ndvi']:.3f}", "Mean vegetation health index (0=Poor, 1=Healthy)."),
        ("Highest-density zone", analysis["hotspot_label"], analysis["hotspot_summary"]),
    ]
    for col, (title, value, copy) in zip(metrics, data):
        with col:
            metric_card(title, value, copy)

    charts = st.columns(2)
    with charts[0]:
        section("Density distribution", "Shows how much of the surveyed image falls into clear, light, medium, and dense canopy classes.")
        fig = distribution_chart(analysis["distribution"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<div class='chart-caption'>Use this chart to see whether the site is mostly open land or dominated by clustered canopy zones.</div>", unsafe_allow_html=True)
    with charts[1]:
        section("NDVI vegetation health", "Color intensity shows vegetation health: red=sparse, yellow=moderate, green=dense vegetation.")
        fig = ndvi_heatmap_chart(analysis["ndvi_normalized"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<div class='chart-caption'>NDVI (Normalized Difference Vegetation Index) is a standard forest health metric. Greener areas indicate healthier vegetation and potential accessibility challenges.</div>", unsafe_allow_html=True)

    ndvi_veg = st.columns(2)
    with ndvi_veg[0]:
        section("Vegetation health distribution", "NDVI-based classification showing sparse, moderate, and dense vegetation shares.")
        fig = ndvi_statistics_chart(analysis["veg_stats"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<div class='chart-caption'>This NDVI classification helps forest officials assess vegetation density and identify accessibility constraints.</div>", unsafe_allow_html=True)
    with ndvi_veg[1]:
        section("Density heatmap", "Tree concentration map showing how many detected trees fall in each analysis grid cell.")
        fig = heatmap_chart(analysis["density_grid"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<div class='chart-caption'>Use this heatmap to identify hotspots where tree concentration is strongest and plan inspection priorities accordingly.</div>", unsafe_allow_html=True)

    middle = st.columns(2)
    with middle[0]:
        section("Site composition summary", "Combines occupancy and density classes into one compact summary profile.")
        fig = site_composition_chart(analysis["distribution"], analysis["veg_stats"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<div class='chart-caption'>This summary compares open grid area, tree-occupied area, and the density profile together.</div>", unsafe_allow_html=True)
    with middle[1]:
        section("Occupancy share", "Compares open cells against cells with detected tree presence.")
        fig = occupancy_chart(analysis["distribution"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<div class='chart-caption'>This chart gives a quick percentage split between open grid area and grid area occupied by detected trees.</div>", unsafe_allow_html=True)

    section("Priority zones", "These are the zones the platform recommends inspecting first based on grid density.")
    priority_rows = [
        {
            "Zone": zone["zone_id"],
            "Density level": zone["density_level"],
            "Trees in zone": zone["tree_count"],
        }
        for zone in analysis["priority_zones"]
    ]
    if priority_rows:
        st.dataframe(priority_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No occupied density zones were found in the current image.")

    next_cols = st.columns([2.9, 1.1])
    with next_cols[1]:
        if st.button("Open route and report", type="primary", use_container_width=True):
            go_to_page("Route & Report")

    insight_cols = st.columns(3)
    with insight_cols[0]:
        insight_card("Density interpretation", analysis["density_summary"]["description"])
    with insight_cols[1]:
        insight_card("Vegetation health (NDVI)", f"Average NDVI: {analysis['mean_ndvi']:.3f} | Dense areas may limit accessibility and field work efficiency.")
    with insight_cols[2]:
        peak_density = f"{analysis['distribution']['max_density']:.2f}"
        insight_card("Peak cell intensity", f"The most concentrated grid cell in the current site contains {peak_density} detected trees.")

    recommendation_cols = st.columns(3)
    for column, recommendation in zip(recommendation_cols, analysis["recommendations"][:3]):
        with column:
            insight_card("Recommendation", recommendation)


def route_report_page():
    hero(
        "Route and report",
        "Use the optimized route and export the final report.",
        "The route view combines density zones and a clean optimized inspection path so the final alignment is easier to interpret and present.",
    )
    analysis = require_analysis()
    if not analysis:
        return

    left, right = st.columns([0.92, 1.0], vertical_alignment="top")
    with left:
        section("Route optimization output", "The image is compact so the full planning summary stays visible on one page.")
        fig = image_preview_figure(analysis["route_rgb"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("<div class='chart-caption'>The route highlights the suggested optimal path, with the start marker shown in green, the end marker shown in blue, and density zones visible beneath the line.</div>", unsafe_allow_html=True)

    with right:
        action_cols = st.columns([1.0, 1.0])
        with action_cols[0]:
            if st.button("Back to overview", use_container_width=True):
                go_to_page("Overview")
        with action_cols[1]:
            st.download_button(
                label="Download report",
                data=analysis["report_bytes"],
                file_name="treesight_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        route_metrics = st.columns(2)
        with route_metrics[0]:
            metric_card("Route distance", analysis["route_distance_label"], analysis["route_method"])
        with route_metrics[1]:
            metric_card("Vegetation health", f"NDVI: {analysis['mean_ndvi']:.3f}", "Higher NDVI indicates denser vegetation and harder accessibility.")

        metric_card("Carbon impact", analysis["carbon_label"], "Estimated associated carbon stock.")
        soft_panel("Inspection order", analysis["inspection_order_text"])
        soft_panel("Top zone", analysis["hotspot_summary"])
        soft_panel("Route optimization method", f"Uses A* pathfinding algorithm which guarantees near-optimal inspection routes by:\n\n• Prioritizing high-density zones for early inspection\n• Accounting for vegetation accessibility costs from NDVI analysis\n• Minimizing unnecessary backtracking across the site\n• Balancing traversal effort with inspection priorities\n\nThis provides forest officials with efficient, data-driven field routes.")
        soft_panel("Environmental impact estimate", analysis["impact_summary"])


def main():
    inject_styles()
    page = top_navigation()
    if page == "Overview":
        home_page()
    elif page == "Detection":
        upload_detection_page()
    elif page == "Analysis":
        count_density_page()
    else:
        route_report_page()


if __name__ == "__main__":
    main()
