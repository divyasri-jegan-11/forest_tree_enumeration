import math
from collections import Counter

import numpy as np


ZONE_PRIORITY = {
    "No Trees": 0,
    "Low Density": 1,
    "Medium Density": 2,
    "High Density": 3,
}


def calculate_density(count, image_shape, pixels_per_unit=1_000_000):
    height, width = image_shape[:2]
    normalized_area = max((height * width) / pixels_per_unit, 1e-6)
    return count / normalized_area


def classify_density(density_score):
    if density_score < 30:
        return "Low Density"
    if density_score < 75:
        return "Medium Density"
    return "High Density"


def summarize_density(count, image_shape):
    score = calculate_density(count, image_shape)
    label = classify_density(score)

    descriptions = {
        "Low Density": "Sparse tree presence with broad open areas and lower ecological friction.",
        "Medium Density": "Balanced vegetation cover with moderate canopy concentration across the scene.",
        "High Density": "Dense canopy presence with limited open corridors and stronger environmental sensitivity.",
    }

    recommendations = {
        "Low Density": "Use this as a low-impact corridor, but validate soil and access constraints before field work.",
        "Medium Density": "Prefer partial avoidance and route around local hotspots where tree clusters intensify.",
        "High Density": "Avoid direct traversal when possible and prioritize the optimized route or protected buffers.",
    }

    return {
        "score": score,
        "label": label,
        "description": descriptions[label],
        "recommendation": recommendations[label],
    }


def resolve_grid_size(image_shape, grid_size):
    height, width = image_shape[:2]
    max_rows = max(4, min(12, height // 40 if height >= 40 else 4))
    max_cols = max(4, min(12, width // 40 if width >= 40 else 4))
    if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
        rows = int(grid_size[0])
        cols = int(grid_size[1])
    else:
        rows = int(grid_size)
        cols = int(grid_size)
    rows = max(2, min(rows, max_rows))
    cols = max(2, min(cols, max_cols))
    return rows, cols


def point_to_grid_cell(point, image_shape, grid_shape):
    height, width = image_shape[:2]
    rows, cols = grid_shape
    cell_width = max(width / cols, 1.0)
    cell_height = max(height / rows, 1.0)
    x, y = point
    col = min(int(x / cell_width), cols - 1)
    row = min(int(y / cell_height), rows - 1)
    return row, col


def zone_name(row_index, col_index):
    return f"{chr(65 + row_index)}{col_index + 1}"


def build_density_grid(tree_points, image_shape, grid_size=6):
    rows, cols = resolve_grid_size(image_shape, grid_size)
    height, width = image_shape[:2]
    grid = np.zeros((rows, cols), dtype=np.int32)
    cell_width = max(width / cols, 1.0)
    cell_height = max(height / rows, 1.0)

    for point in tree_points:
        row, col = point_to_grid_cell(point, image_shape, (rows, cols))
        grid[row, col] += 1

    return {
        "grid": grid,
        "rows": rows,
        "cols": cols,
        "cell_width": cell_width,
        "cell_height": cell_height,
    }


def compute_zone_thresholds(grid):
    occupied_counts = grid[grid > 0]
    if occupied_counts.size == 0:
        return 0, 0
    if occupied_counts.size == 1:
        single_value = int(occupied_counts[0])
        return single_value, single_value + 1
    if np.min(occupied_counts) == np.max(occupied_counts):
        uniform_value = int(occupied_counts[0])
        return uniform_value, uniform_value + 1

    low_cutoff = int(math.ceil(np.quantile(occupied_counts, 0.35)))
    high_cutoff = int(math.ceil(np.quantile(occupied_counts, 0.70)))
    low_cutoff = max(low_cutoff, 1)
    high_cutoff = max(high_cutoff, low_cutoff + 1) if np.max(occupied_counts) > low_cutoff else low_cutoff
    return low_cutoff, high_cutoff


def classify_zone(tree_count, low_cutoff, high_cutoff):
    if tree_count <= 0:
        return "No Trees"
    if tree_count >= high_cutoff:
        return "High Density"
    if tree_count >= low_cutoff:
        return "Medium Density"
    return "Low Density"


def build_zone_details(grid_info, image_shape):
    grid = grid_info["grid"]
    low_cutoff, high_cutoff = compute_zone_thresholds(grid)
    zones = []
    zone_level_map = np.empty(grid.shape, dtype=object)
    height, width = image_shape[:2]

    for row in range(grid_info["rows"]):
        for col in range(grid_info["cols"]):
            x1 = int(round(col * grid_info["cell_width"]))
            y1 = int(round(row * grid_info["cell_height"]))
            x2 = int(round(min(width, (col + 1) * grid_info["cell_width"])))
            y2 = int(round(min(height, (row + 1) * grid_info["cell_height"])))
            tree_count = int(grid[row, col])
            zone_level = classify_zone(tree_count, low_cutoff, high_cutoff)
            zone_level_map[row, col] = zone_level
            zones.append(
                {
                    "zone_id": zone_name(row, col),
                    "row": row,
                    "col": col,
                    "bounds": (x1, y1, x2, y2),
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    "tree_count": tree_count,
                    "density_level": zone_level,
                    "priority": ZONE_PRIORITY[zone_level],
                }
            )

    return zones, zone_level_map


def build_recommendations(zones, occupied_zones, overall_label):
    if not occupied_zones:
        return [
            "No trees were detected in the current image, so no inspection route is required.",
            "Verify the image quality or lower the confidence threshold if trees are expected in this site.",
        ]

    top_zone = occupied_zones[0]
    recommendations = [
        f"Zone {top_zone['zone_id']} has the highest tree density with {top_zone['tree_count']} trees. Inspect this zone first.",
    ]

    medium_or_high = [
        zone["zone_id"]
        for zone in occupied_zones
        if zone["density_level"] in {"High Density", "Medium Density"}
    ]
    if len(medium_or_high) > 1:
        recommendations.append(
            f"After the primary hotspot, continue through {', '.join(medium_or_high[1:4])} to cover the densest remaining zones efficiently."
        )

    if overall_label == "High Density":
        recommendations.append(
            "The site is globally dense, so allocate extra field time and avoid treating low-density edges as representative of the whole area."
        )
    elif overall_label == "Low Density":
        recommendations.append(
            "Most cells are lightly populated, so a shorter verification pass can focus on the occupied pockets instead of full-site coverage."
        )
    else:
        recommendations.append(
            "The site has mixed-density coverage, so prioritize hotspots first and use the route plan to connect medium-density zones with minimal backtracking."
        )

    return recommendations


def analyze_tree_density(tree_points, image_shape, grid_size=6, area_hectares=None):
    grid_info = build_density_grid(tree_points, image_shape, grid_size=grid_size)
    zones, zone_level_map = build_zone_details(grid_info, image_shape)
    occupied_zones = [zone for zone in zones if zone["tree_count"] > 0]
    occupied_zones.sort(
        key=lambda zone: (zone["priority"], zone["tree_count"], -zone["row"], -zone["col"]),
        reverse=True,
    )

    zone_counter = Counter(zone["density_level"] for zone in zones)
    total_cells = len(zones)
    occupied_cells = len(occupied_zones)
    overall_summary = summarize_density(len(tree_points), image_shape)

    score = overall_summary["score"]
    score_label = f"{score:.2f} trees per megapixel"
    if area_hectares and area_hectares > 0:
        score = len(tree_points) / area_hectares
        score_label = f"{score:.2f} trees per hectare"
        overall_label = classify_density(score)
        overall_summary = {
            "score": score,
            "label": overall_label,
            "description": (
                "Sparse tree presence across the surveyed area with relatively open access corridors."
                if overall_label == "Low Density"
                else "Moderate tree concentration across the surveyed area, with several pockets that deserve early inspection."
                if overall_label == "Medium Density"
                else "High tree concentration across the surveyed area, indicating stronger ecological sensitivity and tighter access corridors."
            ),
            "recommendation": (
                "Use the route plan to cover occupied pockets efficiently while validating open zones quickly."
                if overall_label == "Low Density"
                else "Start with the highlighted dense pockets and then connect the remaining medium-density zones."
                if overall_label == "Medium Density"
                else "Inspect the highest-density zones first and keep route deviations minimal to reduce disturbance in dense canopy sections."
            ),
        }

    if occupied_zones:
        top_zone = occupied_zones[0]
        overall_summary["description"] = (
            f"{overall_summary['description']} The strongest hotspot is zone {top_zone['zone_id']} "
            f"with {top_zone['tree_count']} detected trees."
        )
        overall_summary["recommendation"] = (
            f"{overall_summary['recommendation']} Prioritize zone {top_zone['zone_id']} first."
        )

    inspection_order = [zone["zone_id"] for zone in occupied_zones]
    recommendations = build_recommendations(zones, occupied_zones, overall_summary["label"])

    return {
        "grid_counts": grid_info["grid"],
        "grid_shape": (grid_info["rows"], grid_info["cols"]),
        "cell_width": grid_info["cell_width"],
        "cell_height": grid_info["cell_height"],
        "zones": zones,
        "zone_level_map": zone_level_map,
        "zone_counts": {
            "No Trees": zone_counter.get("No Trees", 0),
            "Low Density": zone_counter.get("Low Density", 0),
            "Medium Density": zone_counter.get("Medium Density", 0),
            "High Density": zone_counter.get("High Density", 0),
        },
        "occupied_cells": occupied_cells,
        "occupied_share": (occupied_cells / total_cells) * 100 if total_cells else 0.0,
        "max_trees_in_zone": max((zone["tree_count"] for zone in occupied_zones), default=0),
        "highest_density_zone": occupied_zones[0] if occupied_zones else None,
        "hotspot_zones": occupied_zones[:5],
        "inspection_order": inspection_order,
        "summary": {
            **overall_summary,
            "score_label": score_label,
        },
        "recommendations": recommendations,
    }
