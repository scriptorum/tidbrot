#
# TODO
# Find POV still not great, it seems to always pick close to the center line
# Expose some optimization controls 
# 
load("random.star", "random")
load("render.star", "render")
load("time.star", "time")
load("math.star", "math")

ZOOM_GROWTH = 1.05
FRAME_DURATION_MS = 100
MAX_FRAMES = int(15000 / FRAME_DURATION_MS)
MIN_ITER = 20
ZOOM_TO_ITER = 0.2
BLACK_COLOR = "#000000"
ESCAPE_THRESHOLD = 4.0
MAX_INT = int(math.pow(2, 53))
CTRX, CTRY, MINX, MAXX, MINY, MAXY = -0.75, 0, -2.5, 1.0, -0.875, 0.8753
POI_CHECKS_PER_ZOOM_LEVEL = 100
POI_ACROSS = 20
POI_DOWN = 10
BLACK_PIXEL = render.Box(width=1, height=1, color=BLACK_COLOR)
MAX_ITER = math.round(MIN_ITER + ZOOM_TO_ITER * math.pow(ZOOM_GROWTH, MAX_FRAMES)) + 1
NUM_GRADIENT_STEPS = 12

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
    #tx,ty = -0.74364388703, 0.13182590421
    x, y = CTRX, CTRY                   # Mandelbrot starts centered
    zoom_level = 1.0                    # Initialize zoom level
    frames = list()                     # List to store frames of the animation

    gradient = get_random_gradient()

    # Generate multiple frames for animation
    print("Generating frames")
    for frame in range(MAX_FRAMES):
        print("Generating frame #" + str(frame))
        frame = draw_mandelbrot_at(float(x), float(y), zoom_level, gradient)
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
            escape_distance = get_escape_proximity(newx, newy, int(MIN_ITER + zoom_level * ZOOM_TO_ITER))

            # Look for points with a magnitude close to the escape threshold (4) without exceeding it
            if escape_distance < ESCAPE_THRESHOLD and escape_distance > best_escape:
                print(" - Found better POI", newx, newy, "has escape", escape_distance)
                best_x, best_y, best_escape = newx, newy, escape_distance
                if escape_distance > early_threshold:
                    print (" --- AND BREAKING EARLY")
                    break

    return best_x, best_y, best_escape


# Escape proximity calculation
def get_escape_proximity(a, b, iter_limit):
    _, escape_distance = mandelbrot_calc(a, b, iter_limit)
    return escape_distance

# Function to draw Mandelbrot at a specific center and zoom level
def draw_mandelbrot_at(center_x, center_y, zoom_level, gradient):
    rows = list()
    iter_limit = int(MIN_ITER + zoom_level * ZOOM_TO_ITER)
    
    # Loop through each pixel in the display (64x32)
    for y in range(32):
        row = list()
        next_color = ""
        run_length = 0

        # Calculate the bounds for zooming
        zoom_xmin = center_x - (1.5 / zoom_level)
        zoom_xmax = center_x + (1.5 / zoom_level)
        zoom_ymin = center_y - (1.0 / zoom_level)
        zoom_ymax = center_y + (1.0 / zoom_level)

        for x in range(64):
            # Map the pixel to a zoomed-in complex number
            a = map_range(x, 0, 64, zoom_xmin, zoom_xmax)
            b = map_range(y, 0, 32, zoom_ymin, zoom_ymax)

            # Compute the color at this point
            color = get_mandelbrot_color(a, b, iter_limit, gradient)

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

# Map value v from one range to another
def map_range(v, min1, max1, min2, max2):
    return min2 + (max2 - min2) * (v - min1) / (max1 - min1)

def mandelbrot_calc(a, b, iter_limit):
    zr, zi, cr, ci = 0.0, 0.0, a, b

    dist = 0
    for iter in range(1, iter_limit + 1):
        # Precompute squares to avoid repeating the same multiplication
        zr2 = zr * zr
        zi2 = zi * zi

        # Perform z = z^2 + c
        zi = 2 * zr * zi + ci
        zr = zr2 - zi2 + cr

        # Check if the point has escaped
        dist = zr2 + zi2
        if dist > ESCAPE_THRESHOLD:
            return iter, dist

    return iter_limit, dist

def get_mandelbrot_color(a, b, iter_limit, gradient):
    iter, _ = mandelbrot_calc(a, b, iter_limit)

    if iter == iter_limit:
        return BLACK_COLOR
    
    color = get_gradient_color(iter, gradient)

    return color


def int_to_hex(n):
    if n > 255:
        fail("Can't convert value " + str(n) + " to hex digit")
    hex_digits = "0123456789ABCDEF"
    return hex_digits[n // 16] + hex_digits[n % 16]

# Convert RGB values to a hexadecimal color code
def rgb_to_hex(r, g, b):
    return "#" + int_to_hex(r) + int_to_hex(g) + int_to_hex(b)

def get_gradient_color(iter, gradient):
    # Normalize iteration count between 0 and 1
    t = iter / MAX_ITER % 1.0

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
