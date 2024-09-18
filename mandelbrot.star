load("random.star", "random")
load("render.star", "render")
load("time.star", "time")
load("math.star", "math")

ZOOM_GROWTH = 1.06
FRAME_DURATION_MS = 200
MAX_FRAMES = int(15000 / FRAME_DURATION_MS)
MIN_ITER = 20
ZOOM_TO_ITER = 0.5
BLACK_COLOR = "#000000"
ESCAPE_THRESHOLD = 4.0
MAX_INT = int(math.pow(2, 53))
CTRX, CTRY, MINX, MAXX, MINY, MAXY = -0.75, 0, -2.5, 1.0, -0.875, 0.8753
POI_DOWN = 10
POI_ACROSS = int(POI_DOWN * 2)

def main(config):
    #random.seed(time.now().unix)
    #random.seed(0)
    frames = get_animation_frames()


    # Return the animation with all frames
    return render.Root(
        delay = FRAME_DURATION_MS,
        child = render.Box(render.Animation(frames)),
    )

def get_animation_frames():
    tx, ty = find_point_of_interest()   # Choose a point of interest    
    x, y = CTRX, CTRY                   # Mandelbrot starts centered
    zoom_level = 1.0                    # Initialize zoom level
    frames = list()                     # List to store frames of the animation

    # Generate multiple frames for animation
    for _ in range(MAX_FRAMES):
        frame = draw_mandelbrot_at(float(x), float(y), zoom_level)
        frames.append(frame)
        zoom_level *= ZOOM_GROWTH
        x, y = (x * 0.9 + tx * 0.1), (y * 0.9 + ty * 0.1)

    return frames

def find_point_of_interest():
    x, y, zoom = 0, 0, 1
    for _ in range(MAX_FRAMES):
        x, y = find_interesting_point_near(x, y, zoom)
        print("Found point at " + str(x) + "," + str(y) + " with zoom " + str(zoom))
        zoom *= ZOOM_GROWTH
    return (x, y)

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

# TODO Precompute search grid points and put into list; shuffle list before each use; exit early if great candidate found
def find_interesting_point_near(x, y, zoom_level):
    step = 1 / zoom_level
    (best_x, best_y, best_escape) = (float(x), float(y), get_escape_proximity(x,y, int(MIN_ITER + zoom_level * ZOOM_TO_ITER)))

    minx, maxx, miny, maxy = MINX*step, MAXX*step, MINY*step, MAXY*step
    w, h = maxx-minx, maxy-miny
    stepx, stepy = w / POI_ACROSS, h / POI_DOWN
    offx, offy = rnd() * stepx + x, rnd() * stepy + y

    for newy in float_range(miny + offy, maxy + offy, POI_ACROSS):
        for newx in float_range(minx + offx, maxx + offx, POI_DOWN):
            # Check if the point has higher non-escaping distance
            escape_distance = get_escape_proximity(newx, newy, int(MIN_ITER + zoom_level * ZOOM_TO_ITER))
            if escape_distance < ESCAPE_THRESHOLD and escape_distance > best_escape:
                print (" - Found better POI " + str(newx) + "," + str(newy) + " has escape " + str(escape_distance))
                (best_x, best_y, best_escape) = (newx, newy, escape_distance)

    return (best_x, best_y)

# Function to draw Mandelbrot at a specific center and zoom level
def draw_mandelbrot_at(center_x, center_y, zoom_level):
    rows = list()
    iter = int(MIN_ITER + zoom_level * ZOOM_TO_ITER)

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
            color = get_mandelbrot_color(a, b, iter)

            # Add a 1x1 box with the appropriate color to the row        
            if next_color == "": # First color of row
                run_length = 1
                next_color = color
            elif color == next_color: # Color run detected
                run_length += 1
            else: # Color change
                row.append(render.Box(width=run_length, height=1, color=next_color))
                run_length = 1
                next_color = color

        # Add last box
        row.append(render.Box(width=run_length, height=1, color=next_color))

        # Add the row to the grid
        rows.append(render.Row(children = row))

    return render.Column(
        children = rows,
    )

# Map value v from one range to another
def map_range(v, min1, max1, min2, max2):
    return min2 + (max2 - min2) * (v - min1) / (max1 - min1)

# Escape proximity calculation
def get_escape_proximity(a, b, max_iter):
    _, escape_distance = mandelbrot_calc(a, b, max_iter)
    print ("** POI at " + str(a) + "," + str(b) + " has escape " + str(escape_distance))
    return escape_distance

# Mandelbrot function that returns a color based on escape time
def get_mandelbrot_color(a, b, max_iter):
    iter, _ = mandelbrot_calc(a, b, max_iter)
    return get_color(iter, max_iter)

def mandelbrot_calc(a, b, max_iter):
    # Initialize z = 0 + 0i
    zr, zi, cr, ci = 0.0, 0.0, a, b

    dist = 0
    for i in range(1, max_iter + 1):
        # Perform z = z^2 + c
        zr_next = zr * zr - zi * zi + cr
        zi_next = 2 * zr * zi + ci
        zr, zi = zr_next, zi_next

        # Check if the point has escaped
        dist = zr * zr + zi * zi
        if dist > ESCAPE_THRESHOLD:
            return i, dist

    return max_iter, dist  # If it doesn't escape, return max_iter (black)

def get_color(iteration, max_iter):
    if iteration == max_iter:
        return BLACK_COLOR  # Black for points inside the set

    # Normalize the iteration count to the range [0, 1]
    t = iteration / max_iter 

    # Use a hue-based color scheme for more variety
    hue = int(360 * t)  # Map t to a hue in degrees (0 to 360)
    saturation = 0.5 + 0.3 * (t % 1)  # Gradually change saturation for smoothness
    lightness = 0.5 + 0.2 * (t % 1)  # Gradually vary lightness to avoid harsh transitions

    return hsl_to_hex(hue, saturation, lightness)

# Helper function to convert an integer to two-digit hexadecimal value
def int_to_hex(n):
    hex_digits = "0123456789ABCDEF"
    return hex_digits[n // 16] + hex_digits[n % 16]

# Convert RGB values to a hexadecimal color code
def rgb_to_hex(r, g, b):
    return "#" + int_to_hex(r) + int_to_hex(g) + int_to_hex(b)

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

    return rgb_to_hex(r, g, b)
