###
# QUADTREE
###

# Create a new quadtree node
def create(x1, y1, x2, y2, max_depth=4):
    return {
        "bounds": (x1, y1, x2, y2),
        "max_depth": max_depth,
        "exclusions": [],
        "quadrants": None
    }

# Subdivide the quadtree node into 4 quadrants
def qt_subdivide(qt):
    x1, y1, x2, y2 = qt["bounds"]
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    qt["quadrants"] = [
        create(x1, y1, mid_x, mid_y, qt["max_depth"] - 1),
        create(mid_x, y1, x2, mid_y, qt["max_depth"] - 1),
        create(x1, mid_y, mid_x, y2, qt["max_depth"] - 1),
        create(mid_x, mid_y, x2, y2, qt["max_depth"] - 1)
    ]

# Add an area to the quadtree
def qt_add(qt, x1, y1, x2, y2, iterations):
    if qt["quadrants"] is None and len(qt["exclusions"]) < 4 or qt["max_depth"] == 0:
        qt["exclusions"].append({"bounds": (x1, y1, x2, y2), "iterations": iterations})
    else:
        if qt["quadrants"] is None:
            qt_subdivide(qt)
        for subquad in qt["quadrants"]:
            if qt_overlaps(subquad, x1, y1, x2, y2):
                qt_add(subquad, x1, y1, x2, y2, iterations)

# Check if areas overlap
def qt_overlaps(qt, x1, y1, x2, y2):
    bx1, by1, bx2, by2 = qt["bounds"]
    return not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2)

# Check if a point is in an exclusion area
def qt_check(qt, x, y):
    if qt["quadrants"] is None:
        for exclusion in qt["exclusions"]:
            ex_x1, ex_y1, ex_x2, ex_y2 = exclusion["bounds"]
            if ex_x1 <= x <= ex_x2 and ex_y1 <= y <= ex_y2:
                return exclusion["iterations"]
        return None
    for subquad in qt["quadrants"]:
        if qt_contains(subquad, x, y):
            return qt_check(subquad, x, y)
    return None

# Check if a point is contained within a quadtree node
def qt_contains(qt, x, y):
    x1, y1, x2, y2 = qt["bounds"]
    return x1 <= x <= x2 and y1 <= y <= y2
