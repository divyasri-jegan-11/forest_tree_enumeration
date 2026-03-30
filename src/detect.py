import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = (
    Path("runs/detect/train/weights/best.pt"),
    Path("best.pt"),
    Path("yolo26n.pt"),
    Path("yolov8n.pt"),
    Path("yolov8m.pt"),
)


def resolve_model_path(model_path=None):
    if model_path:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model weights were not found at: {candidate}")

    env_model_path = os.getenv("TREE_MODEL_PATH")
    if env_model_path:
        return resolve_model_path(env_model_path)

    for candidate in MODEL_CANDIDATES:
        resolved = PROJECT_ROOT / candidate
        if resolved.exists():
            return resolved

    searched_paths = "\n".join(str(PROJECT_ROOT / candidate) for candidate in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "No YOLO model weights were found. Place a trained model in one of these paths:\n"
        f"{searched_paths}"
    )


@lru_cache(maxsize=4)
def load_model(model_path=None):
    resolved_model_path = resolve_model_path(model_path)
    return YOLO(str(resolved_model_path))


def calculate_ndvi(image_rgb):
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    For RGB images, approximates NIR using R+G channels and RED using R channel.
    NDVI = (NIR - RED) / (NIR + RED)
    Returns normalized NDVI array [0, 1] and classified vegetation map.
    """
    image = image_rgb.astype(np.float32) / 255.0
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    
    # For RGB images without true NIR, approximate NIR using red and green channels
    nir_approx = red + green
    
    # Calculate NDVI with numerical stability
    denominator = nir_approx + red
    ndvi = np.where(
        denominator > 0,
        (nir_approx - red) / denominator,
        0.0
    )
    
    # Normalize to [0, 1] range
    ndvi_min = np.min(ndvi)
    ndvi_max = np.max(ndvi)
    if ndvi_max > ndvi_min:
        ndvi_normalized = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)
    else:
        ndvi_normalized = ndvi
    
    return ndvi_normalized, ndvi


def classify_vegetation_health(ndvi):
    """
    Classify vegetation based on NDVI values.
    0.0-0.33: Dead/Sparse vegetation (light)
    0.33-0.66: Moderate vegetation (medium)
    0.66-1.0: Healthy/Dense vegetation (dark)
    """
    classes = np.zeros_like(ndvi, dtype=np.uint8)
    classes[(ndvi >= 0.33) & (ndvi < 0.66)] = 1
    classes[ndvi >= 0.66] = 2
    
    labels = ["Sparse (Low NDVI)", "Moderate (Medium NDVI)", "Dense (High NDVI)"]
    counts = [int(np.sum(classes == idx)) for idx in range(3)]
    total = max(int(classes.size), 1)
    shares = [(count / total) * 100 for count in counts]
    
    return {
        "classes": classes,
        "labels": labels,
        "counts": counts,
        "shares": shares,
        "mean_ndvi": float(np.mean(ndvi)),
        "max_ndvi": float(np.max(ndvi)),
        "min_ndvi": float(np.min(ndvi)),
    }


def estimate_tree_size(box_area, image_area):

    if image_area <= 0:
        return "Unknown"

    relative_area = box_area / image_area
    if relative_area < 0.0025:
        return "Small"
    if relative_area < 0.01:
        return "Medium"
    return "Large"


def draw_detection_box(
    image,
    box_coords,
    label_text,
    box_color,
    box_thickness,
    font_scale,
    label_alpha,
    show_labels,
):
    x1, y1, x2, y2 = box_coords
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness)

    if not show_labels:
        return

    text_padding = 6
    (text_width, text_height), _ = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        2,
    )
    label_x1 = x1
    label_y2 = max(text_height + 14, y1)
    label_y1 = max(0, label_y2 - text_height - text_padding * 2)
    label_x2 = min(image.shape[1] - 1, x1 + text_width + text_padding * 2)

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (label_x1, label_y1),
        (label_x2, label_y2),
        box_color,
        -1,
    )
    cv2.addWeighted(overlay, label_alpha, image, 1 - label_alpha, 0, image)
    cv2.putText(
        image,
        label_text,
        (label_x1 + text_padding, label_y2 - text_padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
    )


def draw_tree_center(image, center_point, point_color):
    cv2.circle(image, center_point, 4, (255, 255, 255), -1)
    cv2.circle(image, center_point, 3, point_color, -1)


def build_tree_record(box, detection_index, image_shape):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1 = max(0, min(x1, image_shape[1] - 1))
    x2 = max(0, min(x2, image_shape[1] - 1))
    y1 = max(0, min(y1, image_shape[0] - 1))
    y2 = max(0, min(y2, image_shape[0] - 1))

    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    box_area = width * height
    image_area = image_shape[0] * image_shape[1]
    confidence = float(box.conf[0])

    return {
        "id": detection_index,
        "bbox": [x1, y1, x2, y2],
        "center": (center_x, center_y),
        "confidence": confidence,
        "bbox_area": box_area,
        "relative_area": box_area / image_area if image_area else 0.0,
        "size_class": estimate_tree_size(box_area, image_area),
    }


def detect(
    image_path,
    conf=0.45,
    iou=0.4,
    model_path=None,
    show_labels=False,
    show_centers=False,
):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at: {image_path}")

    model = load_model(model_path)
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)

    annotated_image = image.copy()
    detections = []
    box_color = (110, 128, 255)
    center_color = (36, 168, 95)
    box_thickness = 2

    for result in results:
        for box in result.boxes:
            tree_record = build_tree_record(box, len(detections) + 1, image.shape)
            if tree_record["confidence"] < conf:
                continue
            if tree_record["bbox_area"] < 80:
                continue

            detections.append(tree_record)
            label_text = (
                f"Tree {tree_record['id']} | "
                f"{tree_record['confidence'] * 100:.1f}% | "
                f"{tree_record['size_class']}"
            )
            draw_detection_box(
                annotated_image,
                tree_record["bbox"],
                label_text,
                box_color,
                box_thickness,
                font_scale=0.42,
                label_alpha=0.74,
                show_labels=show_labels,
            )
            if show_centers:
                draw_tree_center(annotated_image, tree_record["center"], center_color)

    confidences = [record["confidence"] for record in detections]
    tree_points = [record["center"] for record in detections]
    resolved_model_path = resolve_model_path(model_path)

    overlay = annotated_image.copy()
    summary_width = min(max(230, image.shape[1] // 3), image.shape[1] - 12)
    cv2.rectangle(overlay, (12, 12), (summary_width, 52), box_color, -1)
    cv2.addWeighted(overlay, 0.72, annotated_image, 0.28, 0, annotated_image)
    cv2.putText(
        annotated_image,
        f"Detected trees: {len(tree_points)}",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    return {
        "annotated_image": annotated_image,
        "count": len(tree_points),
        "tree_points": tree_points,
        "detections": detections,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "model_name": resolved_model_path.name,
        "model_path": str(resolved_model_path),
        "image_shape": image.shape,
    }
