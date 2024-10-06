###
# QUADTREE
###

# Create a new quadtree node
def qt_create(x1, y1, x2, y2, max_depth=4):
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
        qt_create(x1, y1, mid_x, mid_y, qt["max_depth"] - 1),
        qt_create(mid_x, y1, x2, mid_y, qt["max_depth"] - 1),
        qt_create(x1, mid_y, mid_x, y2, qt["max_depth"] - 1),
        qt_create(mid_x, mid_y, x2, y2, qt["max_depth"] - 1)
    ]

def qt_add(qt, area, iterations):
    # Check if we're in a leaf node or if max depth is reached
    if qt["quadrants"] == None and len(qt["exclusions"]) < 4 or qt["max_depth"] == 0:
        qt["exclusions"].append({"bounds": (area["x1"], area["y1"], area["x2"], area["y2"]), "iterations": iterations})
    else:
        if qt["quadrants"] == None:
            qt_subdivide(qt)
        # Check each subquad and recursively add if overlaps
        for subquad in qt["quadrants"]:
            if qt_overlaps(subquad, area):
                qt_add(subquad, area, iterations)

def qt_check_area(qt, area):
    # If we're in a leaf node, check exclusions
    if qt["quadrants"] == None:
        for exclusion in qt["exclusions"]:
            ex_x1, ex_y1, ex_x2, ex_y2 = exclusion["bounds"]
            # Check if the exclusion fully contains the area
            if contains_area(ex_x1, ex_y1, ex_x2, ex_y2, area["x1"], area["y1"], area["x2"], area["y2"]):
                return exclusion["iterations"]
        return None

    # Recursively check subquads if they contain the area
    for subquad in qt["quadrants"]:
        if contains_area(subquad["bounds"][0], subquad["bounds"][1], subquad["bounds"][2], subquad["bounds"][3], area["x1"], area["y1"], area["x2"], area["y2"]):
            result = qt_check_area(subquad, area)
            if result != None:
                return result
    
    return None

def contains_area(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2):
    # Returns True if area (x1_2, y1_2, x2_2, y2_2) is fully contained within (x1_1, y1_1, x2_1, y2_1)
    return x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2

def qt_check_point(qt, x, y):
    # Point check logic (same as the original point check)
    if qt["quadrants"] == None:
        for exclusion in qt["exclusions"]:
            ex_x1, ex_y1, ex_x2, ex_y2 = exclusion["bounds"]
            if ex_x1 <= x and x <= ex_x2 and ex_y1 <= y and y <= ex_y2:
                return exclusion["iterations"]
        return None

    for subquad in qt["quadrants"]:
        if qt_contains(subquad, x, y):
            return qt_check_point(subquad, x, y)
    
    return None

def qt_overlaps(quad, area):
    return overlaps_area(area["x1"], area["y1"], area["x2"], area["y2"], quad["bounds"][0], quad["bounds"][1], quad["bounds"][2], quad["bounds"][3])

def overlaps_area(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2):
    # Returns True if the two areas overlap
    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

# Check if a point is contained within a quadtree node
def qt_contains(qt, x, y):
    x1, y1, x2, y2 = qt["bounds"]
    return x1 <= x and x <= x2 and y1 <= y and y <= y2

# Returns the number of quads stored in this quadtree
def qt_count(qt):
    # Start by counting the current node
    count = 1  # Count the current quad
    
    # Check if the quadtree node has been subdivided into quadrants
    if qt["quadrants"]:
        # Recursively count the quads in each quadrant
        for subquad in qt["quadrants"]:
            count += qt_count(subquad)
    
    # Return the total count
    return count

# Prunes any quads (and their subtrees) that are completely outside the given area.
def qt_prune_outside(qt, area):
    # If the quadtree is a leaf node (no sub-quads)
    if not qt["quadrants"]:  # Check if quadrants list is empty or doesn't exist
        # Filter out exclusions that are completely outside the new area
        qt["exclusions"] = [exclusion for exclusion in qt["exclusions"]
            if overlaps_area(exclusion["bounds"][0], exclusion["bounds"][1],
                                exclusion["bounds"][2], exclusion["bounds"][3],
                                area["x1"], area["y1"], area["x2"], area["y2"])]
        return

    # For internal nodes, recursively prune subquads
    remaining_quadrants = []
    for subquad in qt["quadrants"]:
        if overlaps_area(subquad["bounds"][0], subquad["bounds"][1],
                         subquad["bounds"][2], subquad["bounds"][3],
                         area["x1"], area["y1"], area["x2"], area["y2"]):
            # If the subquad overlaps with the area, prune inside that subquad
            qt_prune_outside(subquad, area)
            remaining_quadrants.append(subquad)

    # Replace quadrants with the pruned list of subquads
    qt["quadrants"] = remaining_quadrants if remaining_quadrants else None
