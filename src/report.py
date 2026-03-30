from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_report(summary):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=36,
        bottomMargin=36,
        leftMargin=42,
        rightMargin=42,
    )

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BodyCopy",
        parent=styles["BodyText"],
        leading=16,
        spaceAfter=8,
    )

    story = [
        Paragraph("TreeSight Analysis Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}",
            styles["Italic"],
        ),
        Spacer(1, 18),
    ]

    project_details = [
        ["Project field", "Value"],
        ["Project name", summary.get("project_name", "TreeSight")],
        ["Site name", summary.get("site_name", "Not specified")],
        ["Image source", summary.get("image_source", "Not specified")],
        ["Survey area", summary.get("survey_area_label", "Not specified")],
    ]

    project_table = Table(project_details, colWidths=[165, 305])
    project_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315c4a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D7CF")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FBF8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EFF6F2")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    metrics = [
        ["Metric", "Value"],
        ["Tree count", str(summary["tree_count"])],
        ["Density label", summary["density_label"]],
        ["Density score", summary.get("density_score_label", f"{summary['density_score']:.2f}")],
        ["Average confidence", f"{summary['average_confidence'] * 100:.1f}%"],
        ["Tree occupancy share", summary.get("occupied_share_label", "Not estimated")],
        ["Trees per hectare", summary.get("trees_per_hectare_label", "Not specified")],
        ["Route length", summary.get("route_steps_label", f"{summary['route_steps']} segments")],
        ["Route method", summary.get("route_method_label", "Not specified")],
        ["Model", summary["model_name"]],
        ["Estimated carbon impact", summary.get("carbon_impact_label", "Not estimated")],
        ["Compensatory plantation", summary.get("plantation_label", "Not estimated")],
    ]

    table = Table(metrics, colWidths=[165, 305])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1C4A35")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D7CF")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FBF8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EFF6F2")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    vegetation_snapshot = [
        ["Vegetation class", "Share of image"],
        ["Sparse", summary.get("veg_sparse_label", "Not estimated")],
        ["Moderate", summary.get("veg_moderate_label", "Not estimated")],
        ["Dense", summary.get("veg_dense_label", "Not estimated")],
    ]

    vegetation_table = Table(vegetation_snapshot, colWidths=[165, 305])
    vegetation_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DD8A3A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8C7B3")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFF9F2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FCF4E8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.extend(
        [
            Paragraph("Project overview", styles["Heading2"]),
            project_table,
            Spacer(1, 18),
            Paragraph("Executive summary", styles["Heading2"]),
            Paragraph(
                "This report summarizes the latest image-based tree enumeration run for the selected diversion assessment site. "
                "The analysis combines AI-based tree detection, density interpretation, route planning, and environmental impact estimation.",
                body_style,
            ),
            Spacer(1, 8),
            Paragraph("Data sources and inputs", styles["Heading2"]),
            Paragraph(
                "The current run is based on the uploaded site image and the user-provided project context, including site name, image source, and assessed area in hectares where available. "
                "These inputs are used to produce tree-count, density, vegetation, route, and impact outputs for the selected site.",
                body_style,
            ),
            Spacer(1, 8),
            Paragraph("Analysis summary", styles["Heading2"]),
            table,
            Spacer(1, 18),
            Paragraph("Vegetation composition snapshot", styles["Heading2"]),
            vegetation_table,
            Spacer(1, 18),
            Paragraph("Methodology and approach", styles["Heading2"]),
            Paragraph(
                "The website processes uploaded satellite, drone, or aerial imagery using a trained object detection model. "
                "Detected tree locations are converted into a spatial density grid, which is then used to estimate canopy pressure, identify hotspot zones, and guide a density-priority route optimization routine.",
                body_style,
            ),
            Spacer(1, 8),
            Paragraph("Key findings", styles["Heading2"]),
            Paragraph(summary["density_description"], body_style),
            Paragraph(summary["recommendation"], body_style),
            Spacer(1, 8),
            Paragraph("Route planning interpretation", styles["Heading2"]),
            Paragraph(
                "The route output should be treated as a planning aid that highlights a lower-friction alignment through the processed image. "
                "It is intended to support review of possible access corridors rather than replace site-specific engineering or statutory approvals.",
                body_style,
            ),
            Spacer(1, 8),
            Paragraph("Environmental impact guidance", styles["Heading2"]),
            Paragraph(
                summary.get(
                    "impact_summary",
                    "Environmental impact estimates were not provided for this run.",
                ),
                body_style,
            ),
            Spacer(1, 8),
            Paragraph("Limitations and validation notes", styles["Heading2"]),
            Paragraph(
                "The results are image-based planning estimates. Tree detections, vegetation classes, and route outputs should be field-validated before final diversion decisions, compensatory planting calculations, or environmental clearances are issued.",
                body_style,
            ),
            Spacer(1, 8),
            Paragraph("Recommended next actions", styles["Heading2"]),
            Paragraph(
                "Validate the AI-based findings against field observations, review the suggested route alongside local ecological constraints, "
                "and use the summary metrics to support diversion planning and environmental decision-making.",
                body_style,
            ),
        ]
    )

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()
