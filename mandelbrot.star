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

ZOOM_GROWTH = 1.06
FRAME_DURATION_MS = 80
MAX_FRAMES = int(15000 / FRAME_DURATION_MS)
MIN_ITER = 20
ZOOM_TO_ITER = 1.0
BLACK_COLOR = "#000000"
ESCAPE_THRESHOLD = 4.0
MAX_INT = int(math.pow(2, 53))
CTRX, CTRY, MINX, MINY, MAXX, MAXY = -0.75, 0, -2.5, -0.875, 1.0, 0.8753
POI_ACROSS = 50
POI_DOWN = 25
POI_ZOOM_GROWTH = ZOOM_GROWTH
BLACK_PIXEL = render.Box(width=1, height=1, color=BLACK_COLOR)
MAX_ITER = math.round(MIN_ITER + ZOOM_TO_ITER * math.pow(ZOOM_GROWTH, MAX_FRAMES)) + 1
NUM_GRADIENT_STEPS = 32
OPTIMIZE_MIN_ESC_DIFF = 1
OPTIMIZE_MIN_ITER = 1000
GRADIENT_SCALE_FACTOR = 1.55

def main(config):
    random.seed(time.now().unix)
    #random.seed(0)

    # Generate the animation with all frames
    frames = get_animation_frames()
    return render.Root(
        delay = FRAME_DURATION_MS,
        child = render.Box(render.Animation(frames)),
    )

def get_animation_frames():
    print("Determining point of interest")
    tx, ty = find_point_of_interest()   # Choose a point of interest    
    # tx,ty = -0.74364388703, 0.13182590421
    #tx,ty = 0,0
    x, y = CTRX, CTRY                   # Mandelbrot starts centered
    zoom_level = 1.0                    # Initialize zoom level
    frames = list()                     # List to store frames of the animation

    gradient = get_random_gradient()

    # Generate multiple frames for animation
    print("Generating frames")
    for frame in range(MAX_FRAMES):
        print("Generating frame #" + str(frame))
        frame = draw_mandelbrot(x, y, gradient, zoom_level)
        frames.append(frame)
        zoom_level *= ZOOM_GROWTH
        x, y = (x * 0.9 + tx * 0.1), (y * 0.9 + ty * 0.1)

    actual_max_iter = int(MIN_ITER + zoom_level * ZOOM_TO_ITER)
    print("Calculated max iterations:" + str(MAX_ITER) + " Actual:" + str(actual_max_iter))

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
    for num in range(MAX_FRAMES):
        x, y, last_escape = find_interesting_point_near(x, y, zoom, num, last_escape)
        print("Settled on POI " + str(x) + "," + str(y) + " with zoom " + str(zoom) + " esc:" + str(last_escape))
        zoom *= POI_ZOOM_GROWTH
    return (x, y)

def find_interesting_point_near(x, y, zoom_level, frame_num, last_escape):
    step = 1 / zoom_level
    (best_x, best_y, best_escape) = x, y, last_escape
    early_threshold = ESCAPE_THRESHOLD - 1 + frame_num / MAX_FRAMES
    minx, maxx, miny, maxy = MINX*step, MAXX*step, MINY*step, MAXY*step
    w, h = maxx-minx, maxy-miny
    stepx, stepy = w / POI_ACROSS, h / POI_DOWN
    offx, offy = rnd() * stepx + x, rnd() * stepy + y

    for newy in float_range(miny + offy, maxy + offy, POI_ACROSS):
        for newx in float_range(minx + offx, maxx + offx, POI_DOWN):
            _, escape_distance = mandelbrot_calc(newx, newy, int(MIN_ITER + zoom_level * ZOOM_TO_ITER))

            # Look for points with a magnitude close to the escape threshold (4) without exceeding it
            if escape_distance < ESCAPE_THRESHOLD and escape_distance > best_escape:
                print(" - Found better POI", newx, newy, "has escape", escape_distance)
                best_x, best_y, best_escape = newx, newy, escape_distance
                if escape_distance > early_threshold:
                    print (" --- AND BREAKING EARLY")
                    break

    return best_x, best_y, best_escape

# Map value v from one range to another
def map_range(v, min1, max1, min2, max2):
    return min2 + (max2 - min2) * (v - min1) / (max1 - min1)

# Performs the mandelbrot calculation on a single point
# Returns both the escape distance and the number of iterations 
# (cannot exceed iter_limit)
def mandelbrot_calc(a, b, iter_limit):
    zr, zi, cr, ci = 0.0, 0.0, a, b

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

def get_gradient_color(iter, gradient):
    if iter >= MAX_ITER:
        return BLACK_COLOR

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

def render_line_opt(last_iter, xp1, yp1, xp2, yp2, xm1, ym1, xm2, ym2, map, max_iter):
    # Determine whether the line is vertical or horizontal
    is_vertical = xp2 == xp1
    
    # Set start and end based on whether the line is vertical or horizontal
    if is_vertical:
        start, end = yp1, yp2
    else:
        start, end = xp1, xp2

    # Initialize xp, yp and xm, ym for iteration
    xp, yp = xp1, yp1
    xm, ym = xm1, ym1

    for val in range(start, end + 1):        
        # Update xm and ym based on whether it's vertical or horizontal
        if is_vertical:
            ym = map_range(val, yp1, yp2, ym1, ym2)
            yp = val  # Update yp in vertical case
        else:
            xm = map_range(val, xp1, xp2, xm1, xm2)
            xp = val  # Update xp in horizontal case

        # Get the pixel iteration count
        pixel = get_cached_pixel(xp, yp, xm, ym, map, max_iter)

        # Initialize last_iter on first iteration
        if last_iter == -1:
            last_iter = pixel
        # Bail out early if iterations don't match
        elif last_iter != pixel:
            return False
        
    # All iterations along the line were identical
    return last_iter


def render_mandelbrot_area(xp1, yp1, xp2, yp2, xm1, ym1, xm2, ym2, map, iter, gradient):
    dxp, dyp = int(xp2 - xp1), int(yp2 - yp1)
    dxm, dym = (xm2 - xm1) / float(dxp), (ym2 - ym1) / float(dyp)

    # A border with the same iterations can be filled with the same color
    line_iter = render_line_opt(-1, xp1, yp1, xp2, yp1, xm1, ym1, xm2, ym1, map, iter)
    if line_iter != False:
        line_iter = render_line_opt(line_iter, xp1, yp2, xp2, yp2, xm1, ym2, xm2, ym2, map, iter)
    if line_iter != False:
        line_iter = render_line_opt(line_iter, xp1, yp1, xp1, yp2, xm1, ym1, xm1, ym2, map, iter)
    if line_iter != False:
        line_iter = render_line_opt(line_iter, xp2, yp1, xp2, yp2, xm2, ym1, xm1, ym2, map, iter)
    if line_iter != False:
        color = get_gradient_color(line_iter, gradient)
        # print("Flooding filling region:", xp1, yp1, xp2, yp2, " to color ", color)
        for y in range(yp1, yp2 + 1):
            for x in range(xp1, xp2 + 1):
                set_pixel(x, y, map, color)
        return map

    # Subdivide further
    else:
        if dxp > 2 and dxp >= dyp:
            # Horizontal split
            splitxp = int(dxp / 2)
            sxp_left = splitxp + xp1
            sxp_right = sxp_left + 1
            sxm_left = xm1 + splitxp * dxm
            sxm_right = xm1 + (splitxp + 1) * dxm
            map = render_mandelbrot_area(xp1, yp1, sxp_left, yp2, xm1, ym1, sxm_left, ym2, map, iter, gradient)
            map = render_mandelbrot_area(sxp_right, yp1, xp2, yp2, sxm_right, ym1, xm2, ym2, map, iter, gradient)
        elif dyp > 2 and dyp >= dxp:
            # Vertical split
            splityp = int(dyp / 2)
            syp_top = splityp + yp1
            syp_bottom = syp_top + 1
            sym_top = ym1 + splityp * dym
            sym_bottom = ym1 + (splityp + 1) * dym
            map = render_mandelbrot_area(xp1, yp1, xp2, syp_top, xm1, ym1, xm2, sym_top, map, iter, gradient)
            map = render_mandelbrot_area(xp1, syp_bottom, xp2, yp2, xm1, sym_bottom, xm2, ym2, map, iter, gradient)
        else:
            # Final coloring with individual points
            grad1 = get_gradient_color(get_cached_pixel(xp1, yp1, xm1, ym1, map, iter), gradient)
            grad2 = get_gradient_color(get_cached_pixel(xp1, yp2, xm1, ym2, map, iter), gradient)
            grad3 = get_gradient_color(get_cached_pixel(xp2, yp2, xm2, ym2, map, iter), gradient)
            grad4 = get_gradient_color(get_cached_pixel(xp2, yp1, xm2, ym1, map, iter), gradient)

            set_pixel(xp1, yp1, map, grad1)
            set_pixel(xp1, yp2, map, grad2)
            set_pixel(xp2, yp2, map, grad3)
            set_pixel(xp2, yp1, map, grad4)

    return map

# Calculate iterations to escape for a pixel if necessary otherwise return it from the cache
def get_cached_pixel(xp1, yp1, xm1, ym1, map, iterations):
    val = get_pixel(xp1, yp1, map)
    if val == MAX_INT or val >= 0:
        iter, esc = mandelbrot_calc(xm1, ym1, iterations)
        if val != MAX_INT:
            print("RECALC for pixel " + str(xp1) + "," + str(yp1) + " iter:" + str(iter) + " esc:" + str(esc) + " MB:" + str(xm1) + "," + str(ym1))
        # print("Calc for pixel " + str(xp1) + "," + str(yp1) + " iter:" + str(iter) + " esc:" + str(esc) + " MB:" + str(xm1) + "," + str(ym1))
        set_pixel(xp1, yp1, map, -iter)
        return iter
    return -val

# Set a pixel value in the map
def set_pixel(xp, yp, map, value):
    if xp < 0 or xp >= 64 or yp < 0 or yp >= 32:
        fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")
    map[yp][xp] = value

# Get a pixel value from the map
def get_pixel(xp, yp, map):
    if xp < 0 or xp >= 64 or yp < 0 or yp >= 32:
        fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")
    return map[yp][xp]

# A map contains either the escape value for that point (as a negative number)
# or the pixel color (as a positive value) or MAX_INT (uninitialized)
def create_empty_map(): 
    map = list()
    for y in range(32):
        row = list()
        for x in range(64):
            row.append(int(MAX_INT))
        map.append(row)    
    return map

def draw_mandelbrot(center_x, center_y, gradient, zoom_level):
    iterations = int(MIN_ITER + zoom_level * ZOOM_TO_ITER)
    
    # Determine coordinates
    half_width = (MAXX - MINX) / zoom_level / 2.0
    half_height = (MAXY - MINY) / zoom_level / 2.0
    minx, miny  = center_x - half_width, center_y - half_height
    maxx, maxy  = center_x + half_width, center_y + half_height

    # Create the map
    map = create_empty_map()    
    map = render_mandelbrot_area(0, 0, 63, 31, minx, miny, maxx, maxy, map, iterations, gradient)
    return render_map(map, gradient)

# Converts a map to a Tidbyt Column made up of Rows made up of Boxes
def render_map(map, gradient):
    # Loop through each pixel in the display (64x32)
    rows = list()
    for y in range(32):
        row = list()
        next_color = ""
        run_length = 0

        for x in range(64):
            color = get_pixel(x, y, map)
            if type(color) == 'int' and color < 0:
                color = get_gradient_color(-color, gradient)

            # Add a 1x1 box with the appropriate color to the row        
            if next_color == "": # First color of row
                run_length = 1
                next_color = color
            elif color == next_color: # Color run detected
                run_length += 1
            else: # Color change
                if run_length == 1 and next_color == BLACK_COLOR:
                    row.append(BLACK_PIXEL)
                else:
                    row.append(render.Box(width=run_length, height=1, color=next_color))
                run_length = 1
                next_color = color

        # Add last box
        if run_length == 1 and next_color == BLACK_COLOR:
            row.append(BLACK_PIXEL)
        else:
            row.append(render.Box(width=run_length, height=1, color=next_color))

        # Add the row to the grid
        rows.append(render.Row(children = row))

    return render.Column(
        children = rows,
    )

