# Forest Intelligence Website

Forest Intelligence is a browser-based AI application for analyzing tree coverage in aerial or satellite imagery with advanced vegetation health assessment and route optimization.

## Website features

- **Tree Detection**: YOLO-based tree detection with bounding box output
- **Tree Counting**: Automated detection count with confidence metrics
- **NDVI Analysis**: Normalized Difference Vegetation Index for vegetation health assessment
- **Vegetation Classification**: Sparse, moderate, and dense vegetation distribution analysis
- **Density Scoring**: Normalized density metrics with occupancy analysis
- **Density Heatmap**: Visual representation of tree concentration hotspots
- **Smart Route Optimization**: Accessibility-aware pathfinding that considers:
  - Density zone priorities (high-density zones inspected first)
  - Vegetation health/NDVI (accounts for accessibility challenges in dense vegetation)
  - Optimal traversal cost minimization
- **Environmental Impact**: Carbon footprint and plantation requirement estimates
- **Downloadable PDF Report**: Comprehensive analysis report with all metrics

## New Features in This Version

### NDVI Vegetation Health Index
- Calculates vegetation health using red and green channel approximation
- Classifies areas as Sparse, Moderate, or Dense vegetation
- Provides mean, max, and min NDVI metrics for forest officials
- Helps identify accessibility constraints in dense vegetation areas

### Improved Route Optimization
- **A* Pathfinding (Default)**: Optimal route planning using heuristic-guided search
  - Uses Euclidean distance heuristic to find near-optimal paths
  - Accounts for density zone priorities and accessibility costs
  - Efficiently handles large numbers of trees (scalable)
  - Balances inspection priorities with traversal efficiency
  
- **Available Algorithms**:
  1. **A* Pathfinding** (Default) - Best for most use cases
     - Optimal/near-optimal solution with good performance
     - Considers both distance and vegetation accessibility
     - Works efficiently for 10+ trees
  
  2. **Dijkstra's Algorithm** - Guaranteed shortest path
     - Finds absolute shortest traversal distance
     - Slower than A* but optimal for small datasets
     - Best for <50 trees where speed isn't critical
  
  3. **Nearest Neighbor** - Fast heuristic
     - Quick approximation for large datasets (100+ trees)
     - Greedy approach but good results
     - Suitable when speed is priority over optimality
  
  4. **Exact TSP** - Brute force optimization
     - Guaranteed optimal for <10 trees
     - Exponential time complexity
     - Only used automatically for very small tree counts

- **Density Multipliers**: Accessibility costs adapt to vegetation type:
  - High-density: 1.5x traversal cost (thicker vegetation)
  - Medium-density: 1.0x baseline cost
  - Low-density: 0.7x reduced cost (easier access)
  - Open area: 0.5x minimal cost

## Project structure

- `app/app.py` - Streamlit website frontend with NDVI and advanced route visualization
- `src/detect.py` - YOLO model loading, image detection, and NDVI calculation
- `src/density.py` - density scoring, zone classification, and recommendations
- `src/route.py` - accessibility-aware route optimization with density weighting
- `src/report.py` - PDF report generation
- `src/train.py` - model training entry point
- `src/split_data.py` - dataset split helper

## Setup

1. Place your dataset inside `dataset/`
2. Train a model or provide local weights such as `runs/detect/train/weights/best.pt`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the website

```bash
streamlit run app/app.py
```

## Optional training command

```bash
yolo detect train data=dataset/data.yaml model=yolov8m.pt epochs=80
```

## Analysis Metrics for Forest Officials

The platform provides forest officials with:

1. **Detection Metrics**: Tree count, confidence, and size classification
2. **Density Analysis**: Grid-based density distribution with occupied/clear area ratios  
3. **Vegetation Health (NDVI)**: Vegetation classification for ecosystem assessment and accessibility planning
4. **Inspection Priority**: Recommended zone visitation order based on density hotspots
5. **Optimal Route Planning**: A* algorithm pathfinding that:
   - Minimizes traversal distance across inspection zones
   - Prioritizes high-density areas for early inspection
   - Accounts for vegetation accessibility (NDVI-based cost adjustment)
   - Provides realistic field time estimates based on terrain difficulty
   - Reduces unnecessary backtracking through dense vegetation
6. **Environmental Impact**: Carbon impact and plantation requirements for offset planning
7. **Accessibility Assessment**: NDVI metrics identify dense vegetation areas requiring extra field time

## How A* Algorithm Helps Decision Making

For forest officials planning field surveys, A* offers:

- **Efficient Route Planning**: Near-optimal routes that save field time and resources
- **Density-Aware Inspection**: Automatically prioritizes high-density hotspots for thorough assessment
- **Realistic Time Estimation**: Routes account for traversal difficulty in different vegetation zones
- **Data-Driven Decisions**: All recommendations based on satellite/drone imagery analysis
- **Scalability**: Handles small sites (10 trees) to large forests (1000+ trees) efficiently
- **Flexibility**: Alternative algorithms available (Dijkstra, Nearest Neighbor) for different use cases

