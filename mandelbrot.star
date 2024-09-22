#
# TODO
# - Expose some optimization controls 
# - Adjust neighboring iterations based on sample escape times:
#   - if a point escapes quickly, reduce the iterations)
# - Bounding box optimizaitons:
#   - sample 4 points
#   - if iteration count is very close, assume entire rectangle escapes at that iteration
#   - if not, subdivide with more points
#   - force horizontal subdivision of first four points, to avoid mandelbrot symmetry causing false fill
# 
load("random.star", "random")
load("render.star", "render")
load("time.star", "time")
load("math.star", "math")

ZOOM_GROWTH = 1.05
FRAME_DURATION_MS = 80
MAX_FRAMES = int(15000 / FRAME_DURATION_MS)
MIN_ITER = 20
ZOOM_TO_ITER = 1.0
BLACK_COLOR = "#000000"
ESCAPE_THRESHOLD = 4.0
MAX_INT = int(math.pow(2, 53))
CTRX, CTRY, MINX, MINY, MAXX, MAXY = -0.75, 0, -2.5, -0.875, 1.0, 0.8753
POI_CHECKS_PER_ZOOM_LEVEL = 100
POI_ACROSS = 20
POI_DOWN = 10
BLACK_PIXEL = render.Box(width=1, height=1, color=BLACK_COLOR)
MAX_ITER = math.round(MIN_ITER + ZOOM_TO_ITER * math.pow(ZOOM_GROWTH, MAX_FRAMES)) + 1
NUM_GRADIENT_STEPS = 32
OPTIMIZE_MIN_ESC_DIFF = 0
OPTIMIZE_MIN_ITER = 1000

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
        zoom *= ZOOM_GROWTH
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

    # Normalize iteration count between 0 and 1
    # t = iter / MAX_ITER % 1.0

    t = math.log(iter, 2) / NUM_GRADIENT_STEPS % 1.0


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

def render_mandelbrot_area(xp1, yp1, xp2, yp2, xm1, ym1, xm2, ym2, map, max_iter, gradient):
    # print("Rendering area " + str(xp1) + "," + str(yp1) + " to " + str(xp2) + "," + str(yp2) \
    #     + " which is " + str(xm1) + "," + str(ym1) + " to " + str(xm2) + "," + str(ym2)) 
    # if xp1 < 0 or xp2 >=64 or yp1 < 0 or yp2 >= 32 or xm1 < MINX or xm2 > MAXX or ym1 < MINY or ym2 > MAXY:
        # fail("One or more coordinates is out of range PT1:", xp1, yp2, "PT2:", xp2, yp2, "MB1:", xm1, ym1, "MB2:", xm2, ym2)

    dxp, dyp  = int(xp2 - xp1), int(yp2 - yp1) # Difference between pixel numbers
    dxm, dym = xm2 - xm1 / float(dxp), xm2 - xm1 / float(dxp) # Range of mb set area

    # check four corners
    iter1 = get_cached_pixel(xp1, yp1, xm1, ym1, map, max_iter)
    iter2 = get_cached_pixel(xp1, yp2, xm1, ym2, map, max_iter)
    iter3 = get_cached_pixel(xp2, yp2, xm2, ym2, map, max_iter)
    iter4 = get_cached_pixel(xp2, yp1, xm2, ym1, map, max_iter)

    # if iteration count is very close, assume entire rectangle escapes at that iteration
    if similar_iterations([iter1, iter2, iter3, iter4]):
        # print("Similar iterations (" + str(iter1) + "," + str(iter2) + "," + str(iter3) + "," + str(iter4) + "), flood filling") 
        color = get_gradient_color(iter1, gradient)
        for y in range(yp1, yp2 + 1):
            for x in range(xp1, xp2 + 1):
                set_pixel(x, y, map, color)

    # if not, subdivide with more points
    else:
        # Subdivide on x
        if dxp > 2 and dxp >= dyp:
            # print("* Splitting horizontally")
            pixel_width = (xm2 - xm1) / dxp
            splitxp = int(dxp / 2)
            sxp_left = splitxp + xp1
            sxp_right = sxp_left + 1
            sxm_left = xm1 + splitxp * pixel_width
            sxm_right = xm1 + (splitxp + 1) * pixel_width
            map = render_mandelbrot_area(xp1, yp1, sxp_left, yp2, xm1, ym1, sxm_left, ym2, map, max_iter, gradient)
            map = render_mandelbrot_area(sxp_right, yp1, xp2, yp2, sxm_right, ym1, xm2, ym2, map, max_iter, gradient)

        # Subdivide on y
        elif dyp > 2 and dyp >= dxp:
            # print("* Splitting vertically")
            pixel_height = (ym2 - ym1) / dyp
            splityp = int(dyp / 2)
            syp_top = splityp + yp1
            syp_bottom = syp_top + 1
            sym_top = ym1 + splityp * pixel_height
            sym_bottom = ym1 + (splityp + 1) * pixel_height
            map = render_mandelbrot_area(xp1, yp1, xp2, syp_top, xm1, ym1, xm2, sym_top, map, max_iter, gradient)
            map = render_mandelbrot_area(xp1, syp_bottom, xp2, yp2, xm1, sym_bottom, xm2, ym2, map, max_iter, gradient)

        # No more subdivision
        else:
            grad1 = get_gradient_color(iter1, gradient)
            grad2 = get_gradient_color(iter2, gradient)
            grad3 = get_gradient_color(iter3, gradient)
            grad4 = get_gradient_color(iter4, gradient)
            # print("* Coloring borders (" + grad1 + "/" + grad2 + "/" + grad3 + "/" + grad4 + ") and calling it a day")
            set_pixel(xp1, yp1, map, grad1)
            set_pixel(xp1, yp2, map, grad2)
            set_pixel(xp2, yp2, map, grad3)
            set_pixel(xp2, yp1, map, grad4)
    
    # All done
    return map

def similar_iterations(escapes):
    size = len(escapes)
    if size > 1:
        for i in range(1, size):
            if escapes[i] > OPTIMIZE_MIN_ITER or abs(escapes[0] - escapes[i]) > OPTIMIZE_MIN_ESC_DIFF:
                return False
    return True      

# Calculate escape value if necessary otherwise return it from the cache
def get_cached_pixel(xp1, yp1, xm1, ym1, map, iterations):
    val = get_pixel(xp1, yp1, map)
    if val == MAX_INT or val >= 0:
        (iter, esc) = mandelbrot_calc(xm1, ym1, iterations)
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

