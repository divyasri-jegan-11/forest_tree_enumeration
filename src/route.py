import math
import heapq
import numpy as np


# =========================
# 🔥 UTIL FUNCTIONS
# =========================
def clamp_point(point, image_shape):
    h, w = image_shape[:2]
    return (
        min(max(int(point[0]), 0), w - 1),
        min(max(int(point[1]), 0), h - 1),
    )


def default_point(image_shape, location):
    h, w = image_shape[:2]
    return {
        "top_left": (0, 0),
        "bottom_right": (w - 1, h - 1),
    }.get(location, (0, 0))


def euclidean_distance(a, b):
    return math.dist(a, b)


# =========================
# 🌳 COST MAP (FIXED)
# =========================
def compute_cost_map(tree_points, image_shape, scale=4):
    h, w = image_shape[:2]
    hs, ws = h // scale, w // scale

    cost = np.ones((hs, ws), dtype=np.float32)
    density = np.zeros((hs, ws), dtype=np.float32)

    # 🔥 Build density map
    for tx, ty in tree_points:
        gx, gy = int(tx // scale), int(ty // scale)

        for dy in range(-12, 13):
            for dx in range(-12, 13):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < ws and 0 <= ny < hs:

                    dist = math.hypot(dx, dy)

                    if dist < 2:
                        cost[ny, nx] = 9999  # tree block

                    if dist < 12:
                        density[ny, nx] += (12 - dist)

    # 🔥 Normalize density
    if density.max() > 0:
        density = density / density.max()

    # 🔥 APPLY STRONG NON-LINEAR PENALTY
    for y in range(hs):
        for x in range(ws):
            d = density[y, x]

            # 🚀 KEY FIX: non-linear scaling
            cost[y, x] += (d ** 2) * 40

    return cost


# =========================
# 🚀 DIJKSTRA (IMPROVED)
# =========================
def dijkstra_route(start, end, tree_points, image_shape, scale=4):
    h, w = image_shape[:2]
    hs, ws = h // scale, w // scale

    cost_map = compute_cost_map(tree_points, image_shape, scale)

    start = (int(start[1] // scale), int(start[0] // scale))
    end = (int(end[1] // scale), int(end[0] // scale))

    pq = [(0, start)]
    came = {}
    dist = {start: 0}

    directions = [(-1,0),(1,0),(0,-1),(0,1),
                  (-1,-1),(-1,1),(1,-1),(1,1)]

    while pq:
        curr_cost, node = heapq.heappop(pq)

        if node == end:
            break

        for dy, dx in directions:
            ny, nx = node[0] + dy, node[1] + dx

            if 0 <= ny < hs and 0 <= nx < ws:
                if cost_map[ny, nx] >= 9999:
                    continue

                move_cost = math.sqrt(2) if dx != 0 and dy != 0 else 1

                # 🔥 TURN PENALTY
                prev = came.get(node, None)
                turn_penalty = 0

                if prev:
                    dir1 = (node[0]-prev[0], node[1]-prev[1])
                    dir2 = (ny-node[0], nx-node[1])

                    if dir1 != dir2:
                        turn_penalty = 2.0

                # 🔥 STRAIGHT-LINE PREVENTION
                straight_penalty = 0.3

                new_cost = (
                    curr_cost
                    + move_cost * cost_map[ny, nx]
                    + turn_penalty
                    + straight_penalty
                )

                if (ny, nx) not in dist or new_cost < dist[(ny, nx)]:
                    dist[(ny, nx)] = new_cost

                    priority = new_cost + 0.7 * math.hypot(end[0]-ny, end[1]-nx)

                    heapq.heappush(pq, (priority, (ny, nx)))
                    came[(ny, nx)] = node

    # reconstruct
    path = []
    current = end

    while current != start:
        path.append(current)
        if current not in came:
            return [(start[1]*scale, start[0]*scale),
                    (end[1]*scale, end[0]*scale)]
        current = came[current]

    path.append(start)
    path.reverse()

    return [(p[1]*scale, p[0]*scale) for p in path]


# =========================
# ✨ SMOOTHING
# =========================
def simplify_path(points):
    if len(points) < 3:
        return points

    simplified = [points[0]]

    for i in range(1, len(points)-1):
        prev = simplified[-1]
        curr = points[i]
        nxt = points[i+1]

        v1 = (curr[0]-prev[0], curr[1]-prev[1])
        v2 = (nxt[0]-curr[0], nxt[1]-curr[1])

        if v1 != v2:
            simplified.append(curr)

    simplified.append(points[-1])
    return simplified


def bezier_smooth(points):
    if len(points) < 3:
        return points

    smooth = []

    for i in range(len(points)-2):
        p0 = np.array(points[i])
        p1 = np.array(points[i+1])
        p2 = np.array(points[i+2])

        for t in np.linspace(0, 1, 15):
            point = (1-t)**2 * p0 + 2*(1-t)*t*p1 + t**2*p2
            smooth.append(tuple(point.astype(int)))

    return smooth


# =========================
# 🔥 FINAL FUNCTION
# =========================
def get_optimized_route(
    tree_points,
    image_shape,
    density_analysis=None,
    start_point=None,
    end_point=None,
    **kwargs
):

    start_point = clamp_point(start_point or default_point(image_shape, "top_left"), image_shape)
    end_point = clamp_point(end_point or default_point(image_shape, "bottom_right"), image_shape)

    route_points = dijkstra_route(
        start_point,
        end_point,
        tree_points,
        image_shape,
        scale=4
    )

    route_points = simplify_path(route_points)
    route_points = bezier_smooth(route_points)

    total_distance = sum(
        euclidean_distance(a, b)
        for a, b in zip(route_points, route_points[1:])
    )

    return {
        "route_points": route_points,
        "start_point": start_point,
        "end_point": end_point,
        "total_distance": total_distance,
        "segments": len(route_points)-1,
        "method": "Consistent Optimal Path",
    }