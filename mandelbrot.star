#
# TODO
# - Expose some optimization controls 
# - Verify get_cached_pixel isn't recalculating color
# - Introduce natural log * 1.55 scaling for color gradient
# - Does it make sense to come up with a point map, so I don't have to keep calculating mandelbrot coords at each zoom level?
# 
load("random.star", "random")
load("render.star", "render")
load("time.star", "time")
load("math.star", "math")

ZOOM_GROWTH = 1.1
FRAME_DURATION_MS = 200
MAX_FRAMES = int(15000 / FRAME_DURATION_MS)
MIN_ITER = 20
ZOOM_TO_ITER = 1
MAX_ZOOM = math.pow(ZOOM_GROWTH, MAX_FRAMES)
BLACK_COLOR = "#000000"
ESCAPE_THRESHOLD = 4.0
MAX_INT = int(math.pow(2, 53))
CTRX, CTRY, MINX, MINY, MAXX, MAXY = -0.75, 0, -2.5, -0.875, 1.0, 0.8753
POI_ZOOM_GROWTH = ZOOM_GROWTH
POI_ZOOM_DEPTH = MAX_FRAMES #26 #int(math.pow(MAX_ZOOM, 1/POI_ZOOM_GROWTH))  # int(MAX_FRAMES / FRAME_DURATION)
BLACK_PIXEL = render.Box(width=1, height=1, color=BLACK_COLOR)
MAX_ITER = int(math.round(MIN_ITER + ZOOM_TO_ITER * math.pow(ZOOM_GROWTH, MAX_FRAMES)) + 1)
NUM_GRADIENT_STEPS = 32
OPTIMIZE_MIN_ESC_DIFF = 1
OPTIMIZE_MIN_ITER = 1000
GRADIENT_SCALE_FACTOR = 2 # 1.55
BOUNDARY_THRESHOLD = 20
AA_JITTER_SAMPLES = 4
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 32
MAX_PIXEL_X = DISPLAY_WIDTH - 1
MAX_PIXEL_Y = DISPLAY_HEIGHT - 1

def main(config):
    random.seed(1)
    app = {"config": config}

    # Generate the animation with all frames
    frames = get_animation_frames(app)
    return render.Root(
        delay = FRAME_DURATION_MS,
        child = render.Box(render.Animation(frames)),
    )

def get_animation_frames(app):
    print("Determining point of interest")
    tx, ty = find_point_of_interest()   # Choose a point of interest    
    # tx,ty = -0.743DISPLAY_WIDTH388703, 0.13182590421
    #tx,ty = 0,0

    x, y = CTRX, CTRY                   # Mandelbrot starts centered
    frames = list()                     # List to store frames of the animation

    app['target'] = (tx, ty)
    app['gradient'] = get_random_gradient()
    app['zoom_level'] = 1.0 # Initialize the zoom level

    # Generate multiple frames for animation
    print("Generating frames")
    for frame in range(MAX_FRAMES):
        print("Generating frame #" + str(frame))
        frame = draw_mandelbrot(app, x, y)
        frames.append(frame)
        app['zoom_level'] *= ZOOM_GROWTH
        x, y = (x * 0.9 + tx * 0.1), (y * 0.9 + ty * 0.1)

    actual_max_iter = int(MIN_ITER + app['zoom_level'] * ZOOM_TO_ITER)
    print("Calculated max iterations:" + str(MAX_ITER) + " Actual:" + str(actual_max_iter))
    print("Final zoom level:", app['zoom_level'])

    return frames

def rnd():
    return float(random.number(0, MAX_INT)) / float (MAX_INT)

def float_range(start, end, num_steps, inclusive=False):
    step_size = (float(end) - float(start)) / num_steps
    result = []
    for i in range(num_steps):
        result.append(start + i * step_size)
    if inclusive:
        result.append(end)
    return result

def find_point_of_interest():
    x, y, zoom, last_escape = CTRX, CTRY, 1, 0
    for num in range(POI_ZOOM_DEPTH):
        x, y, last_escape = trace_boundary(x, y, zoom, num)
        print("Settled on POI " + str(x) + "," + str(y) + " with zoom " + str(zoom) + " esc: " + str(last_escape))
        zoom *= POI_ZOOM_GROWTH
    print("POI final zoom level:", zoom)
    return x, y

def trace_boundary(x, y, zoom_level, frame_num):
    step = 1 / zoom_level
    grid_size = 50  # Adjust grid size depending on zoom level or fixed value
    points_of_interest = []
    best_x, best_y, best_escape = x, y, 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Sample points in the zoomed region
            x_sample = x + i * step
            y_sample = y + j * step

            # Compute the escape distance at the current point
            iter1, dist1 = mandelbrot_calc(x_sample, y_sample, int(MIN_ITER + zoom_level * ZOOM_TO_ITER))

            # Check adjacent points for boundary detection
            x_adjacent = x_sample + step
            y_adjacent = y_sample + step
            iter2, dist2 = mandelbrot_calc(x_adjacent, y_adjacent, int(MIN_ITER + zoom_level * ZOOM_TO_ITER))

            # If there is a significant difference in escape distances, we found a boundary
            if abs(iter1 - iter2) > BOUNDARY_THRESHOLD:
                points_of_interest.append((x_sample, y_sample, dist1))

                # Choose the point closest to the escape threshold (without exceeding it)
                if dist1 > best_escape and dist1 < ESCAPE_THRESHOLD:
                    best_x, best_y, best_escape = x_sample, y_sample, dist1

    # If no boundary is found, return the current best point
    if len(points_of_interest) == 0:
        print("No boundary found at this level, returning best found POI")
        return best_x, best_y, best_escape

    return best_x, best_y, best_escape


# Map value v from one range to another
def map_range(v, min1, max1, min2, max2):
    # print("map_range v:", v, "first:", min1, max1, "second:", min2, max2)
    return min2 + (max2 - min2) * (v - min1) / (max1 - min1)

# Performs the mandelbrot calculation on a single point
# Returns both the escape distance and the number of iterations 
# (cannot exceed iter_limit)
def mandelbrot_calc(x, y, iter_limit):
    zr, zi, cr, ci = 0.0, 0.0, x, y

    dist = 0
    for iter in range(1, iter_limit + 1):
        # Precompute squares to avoid repeating the same multiplication
        zr2 = zr * zr
        zi2 = zi * zi

        # Check if the point has escaped (this should happen after both zr and zi are updated)
        dist = zr2 + zi2
        if dist > ESCAPE_THRESHOLD:
            return iter, dist

        # Perform z = z^2 + c
        zi = 2 * zr * zi + ci
        zr = zr2 - zi2 + cr

    return MAX_ITER, dist

def int_to_hex(n):
    if n > 255:
        fail("Can't convert value " + str(n) + " to hex digit")
    hex_digits = "0123456789ABCDEF"
    return hex_digits[n // 16] + hex_digits[n % 16]

# Convert RGB values to a hexadecimal color code
def rgb_to_hex(r, g, b):
    return "#" + int_to_hex(r) + int_to_hex(g) + int_to_hex(b)

def get_gradient_color(app, iter):
    if iter >= MAX_ITER:
        return BLACK_COLOR
    
    gradient = app['gradient']

    # For debugging to isolate gradient issues
    # v = gradient[iter % NUM_GRADIENT_STEPS] 
    # return rgb_to_hex(v[0], v[1], v[2])

    # Convert iterations to a color
    # t = math.log(iter, 2) / NUM_GRADIENT_STEPS % 1.0
    t = (math.pow(math.log(iter), GRADIENT_SCALE_FACTOR) / NUM_GRADIENT_STEPS) % 1.0

    # Number of keyframes
    num_keyframes = len(gradient) - 1
    
    # Ensure we are covering the whole gradient range
    frame_pos = t * num_keyframes
    lower_frame = int(frame_pos)  # Index of the lower keyframe
    upper_frame = min(lower_frame + 1, num_keyframes)  # Index of the upper keyframe
    
    # Fractional part for interpolation between the two keyframes
    local_t = frame_pos - float(lower_frame)
    
    # Get the colors of the two keyframes to blend between
    color_start = gradient[lower_frame]
    color_end = gradient[upper_frame]
    
    # Perform linear interpolation (LERP) between the two colors
    r = int(color_start[0] + local_t * (color_end[0] - color_start[0]))
    g = int(color_start[1] + local_t * (color_end[1] - color_start[1]))
    b = int(color_start[2] + local_t * (color_end[2] - color_start[2]))

    # Return the hex color code
    return rgb_to_hex(r, g, b)

def random_color_tuple():
    return (random.number(0, 255), random.number(0, 255), random.number(0, 255))

def get_random_gradient():
    print ("Generating gradient")
    gradient = []
    color = random_color_tuple()
    for i in range(0, NUM_GRADIENT_STEPS):
        color = alter_color(color)
        gradient.append(color)
    return gradient

# At least one channel flipped, another randomized
def alter_color(color):
    flip_idx = random.number(0,2)
    rnd_idx = (flip_idx + random.number(1,2)) % 3
    keep_idx = 3 - flip_idx - rnd_idx
    new_color = [0,0,0]
    new_color[flip_idx] = 255 - color[flip_idx]
    new_color[rnd_idx] = random.number(0, 255)
    new_color[keep_idx] = color[keep_idx]
    return new_color

def hsl_to_hex(h, s, l):
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

# Renders a line
# Returns the number of iterations found if they are all the same
# If match_iter is passed something other than False, then it will
# compare all iterations against this value
# Returns the number of iterations, or False if they are not all the same
def render_line_opt(app, match_iter, pix, set, max_iter):
    # print("render_line_opt match_iter:", match_iter, "pix:", pix, "set:", set, "max_iter:", max_iter)
    
    # Determine whether the line is vertical or horizontal
    is_vertical = pix['x1'] == pix['x2']
    
    # Set start and end based on whether the line is vertical or horizontal
    if is_vertical:
        start, end = pix['y1'], pix['y2']
    else:
        start, end = pix['x1'], pix['x2']

    # Initialize xp, yp and xm, ym for iteration
    xp, yp = pix['x1'], pix['y1']
    xm, ym = set['x1'], set['y1']

    for val in range(start, end + 1):        
        # Update xm and ym based on whether it's vertical or horizontal
        if is_vertical:
            ym = map_range(val, pix['y1'], pix['y2'], set['y1'], set['y2'])
            yp = val  # Update yp in vertical case
        else:
            xm = map_range(val, pix['x1'], pix['x2'], set['x1'], set['x2'])
            xp = val  # Update xp in horizontal case

        # Get the pixel iteration count
        cache = cache_pixel(app, xp, yp, xm, ym, max_iter)

        # Initialize match_iter on first iteration
        if match_iter == -1:
            match_iter = cache

        elif match_iter != cache:
            return False
        
    # All iterations along the line were identical
    return match_iter

def alt(obj, field, value):
    # Create a shallow copy manually
    c = {}
    for k, v in obj.items():
        c[k] = v
    c[field] = value  # Set the new value for the specified field
    return c


def render_mandelbrot_area(app, pix, set, iter_limit):
    dxp, dyp = int(pix['x2'] - pix['x1']), int(pix['y2'] - pix['y1'])
    dxm, dym = (set['x2'] - set['x1']) / float(dxp), (set['y2'] - set['y1']) / float(dyp)
    # print("render_mandelbrot_area:", pix, set, iter_limit, "dp:", dxp, dyp, "dm:", dxm, dym)

    # A border with the same iterations can be filled with the same color
    match = render_line_opt(app, False, alt(pix, 'y2', pix['y1']), alt(set, 'y2', set['y1']), iter_limit)
    if match != False:
        match = render_line_opt(app, match, alt(pix, 'y1', pix['y2']), alt(set, 'y1', set['y2']), iter_limit)
    if match != False:
        match = render_line_opt(app, match, alt(pix, 'x2', pix['x1']), alt(set, 'x2', set['x1']), iter_limit)
    if match != False:
        match = render_line_opt(app, match, alt(pix, 'x1', pix['x2']), alt(set, 'x1', pix['x2']), iter_limit)

    if match != False:
        # print("Flooding filling region:", pix, " with iter:", match1)
        for y in range(pix['y1'], pix['y2'] + 1):
            for x in range(pix['x1'], pix['x2'] + 1):
                set_pixel(app, x, y, match)

    # Subdivide further
    else:
        if dxp > 2 and dxp >= dyp:
            # Horizontal split
            splitxp = int(dxp / 2)
            sxp_left = splitxp + pix['x1']
            sxp_right = sxp_left + 1
            sxm_left = set['x1'] + splitxp * dxm
            sxm_right = set['x1'] + (splitxp + 1) * dxm
            render_mandelbrot_area(app, alt(pix, 'x2', sxp_left),  alt(set, 'x2', sxm_left),  iter_limit)
            render_mandelbrot_area(app, alt(pix, 'x1', sxp_right), alt(set, 'x1', sxm_right), iter_limit)

        elif dyp > 2 and dyp >= dxp:
            # Vertical split
            splityp = int(dyp / 2)
            syp_above = splityp + pix['y1']
            syp_below = syp_above + 1
            sym_above = set['y1'] + splityp * dym
            sym_below = set['y1'] + (splityp + 1) * dym
            render_mandelbrot_area(app, alt(pix, 'y2', syp_above), alt(set, 'y2', sym_above), iter_limit)
            render_mandelbrot_area(app, alt(pix, 'y1', syp_below), alt(set, 'y1', sym_below), iter_limit)
        else:
            cache_pixel(app, pix['x1'], pix['y1'], set['x1'], set['y1'], iter_limit)
            cache_pixel(app, pix['x1'], pix['y2'], set['x1'], set['y2'], iter_limit)
            cache_pixel(app, pix['x2'], pix['y1'], set['x2'], set['y1'], iter_limit)
            cache_pixel(app, pix['x2'], pix['y2'], set['x2'], set['y2'], iter_limit)

# Calculates the number of iterations for a point on the map and returns it
# If value is unavailable, it calculates it now
def cache_pixel(app, xp, yp, xm, ym, iter_limit):
    stored_val = get_pixel(app, xp, yp)
    if stored_val == -1:
        iter, esc = mandelbrot_calc(xm, ym, iter_limit)

        if stored_val != -1:
            print("RECALC for pixel " + str(xp) + "," + str(yp) + " iter:" + str(iter) + " esc:" + str(esc) + " MB:" + str(xm) + "," + str(ym))

        # print("Calc for pixel " + str(xp1) + "," + str(yp1) + " iter:" + str(iter) + " esc:" + str(esc) + " MB:" + str(xm1) + "," + str(ym1))

        set_pixel(app, xp, yp, iter)
        return iter
    return stored_val

# Set the number of iterations for a point on the map
def set_pixel(app, xp, yp, value):
    if xp < 0 or xp >= DISPLAY_WIDTH or yp < 0 or yp >= DISPLAY_HEIGHT:
        fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")
    app['map'][yp][xp] = value

# Returns the number of iterations for a point on the map
def get_pixel(app, xp, yp):
    if xp < 0 or xp >= DISPLAY_WIDTH or yp < 0 or yp >= DISPLAY_HEIGHT:
        fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")
    return app['map'][yp][xp]

# A map contains either the escape value for that point (as a negative number)
# or the pixel color (as a positive value) or -1 (uninitialized)
def create_empty_map(): 
    map = list()
    for y in range(DISPLAY_HEIGHT):
        row = list()
        for x in range(DISPLAY_WIDTH):
            row.append(int(-1))
        map.append(row)    
    return map

def draw_mandelbrot(app, x, y):
    iterations = int(MIN_ITER + app['zoom_level'] * ZOOM_TO_ITER)
    
    # Determine coordinates
    half_width = (MAXX - MINX) / app['zoom_level'] / 2.0
    half_height = (MAXY - MINY) / app['zoom_level'] / 2.0
    minx, miny  = x - half_width, y - half_height
    maxx, maxy  = x + half_width, y + half_height

    # Create the map
    pix = { "x1": 0, "y1": 0, "x2": MAX_PIXEL_X, "y2": MAX_PIXEL_Y }
    set = { "x1": minx, "y1": miny, "x2": maxx, "y2": maxy }
    app['map'] = create_empty_map()    
    render_mandelbrot_area(app, pix, set, iterations)
    return render_map(app)

# Converts a map to a Tidbyt Column made up of Rows made up of Boxes
def render_map(app):
    # Loop through each pixel in the display
    rows = list()
    for y in range(DISPLAY_HEIGHT):
        row = list()
        next_iter = -1
        run_length = 0

        for x in range(DISPLAY_WIDTH):
            iter = get_pixel(app, x, y)
            if iter == -1:
                print("Unresolved pixel at", x, y)
                iter = MAX_ITER

            # Add a 1x1 box with the appropriate color to the row        
            if next_iter == -1: # First color of row
                run_length = 1
                next_iter = iter
            elif iter == next_iter: # Color run detected
                run_length += 1
            else: # Color change
                if run_length == 1 and next_iter == MAX_ITER:
                    row.append(BLACK_PIXEL)
                else:
                    color = get_gradient_color(app, next_iter)
                    row.append(render.Box(width=run_length, height=1, color=color))
                run_length = 1
                next_iter = iter

        # Add last box
        if run_length == 1 and next_iter == MAX_ITER:
            row.append(BLACK_PIXEL)
        else:
            color = get_gradient_color(app, next_iter)
            row.append(render.Box(width=run_length, height=1, color=color))

        # Add the row to the grid
        rows.append(render.Row(children = row))

    return render.Column(
        children = rows,
    )

