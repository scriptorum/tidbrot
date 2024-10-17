#
# TODO
# - Derive certain variables from a budgeting system
#    + Measure the time used to find a POI
#    + If not a lot of time, display a single frame with high AA
#    + Otherwise do an animation
#

load("math.star", "math")
load("random.star", "random")
load("render.star", "render")
load("schema.star", "schema")
load("time.star", "time")

MIN_ITER = 100  # minimum iterations, raise if initial zoom is > 1
ZOOM_TO_ITER = 1.0  # 1.0 standard, less for faster calc, less accuracy
ESCAPE_THRESHOLD = 4.0  # 4.0 standard, less for faster calc, less accuracy
DISPLAY_WIDTH = 64  # Tidbyt is 64 pixels wide
DISPLAY_HEIGHT = 32  # Tidbyt is 32 pixels high
NUM_GRADIENT_STEPS = 64  # Higher = more color variation
GRADIENT_SCALE_FACTOR = 1.55  # 1.55 = standard, less for more colors zoomed in, more for few colors zoomed in
MAX_POI_SAMPLES = 100000  # Number of random points to check for POI-worthiness
CTRX, CTRY = -0.75, 0  # mandelbrot center
MINX, MINY, MAXX, MAXY = -2.5, -0.875, 1.0, 0.8753  # Bounds to use for mandelbrot set
MAX_COLOR_CHANNEL = 8  # Max quantized channel values (helps reduce Image Too Large errors)
CHANNEL_MULT = 255.9999 / MAX_COLOR_CHANNEL  # Conversion from quantized value to full range color channel (0-255)
BLACK_COLOR = "#000000"  # Shorthand for black color
MAX_INT = int(math.pow(2, 53))  # Guesstimate for Starlark max_int
BLACK_PIXEL = render.Box(width = 1, height = 1, color = BLACK_COLOR)  # Pregenerated 1x1 pixel black box

def main(config):
    seed = time.now().unix
    random.seed(seed)
    print ("Using random seed:", seed)
    app = {"config": config}

    # RANGE      -->  1 = 1x1 pixel no blending, 2 = 2x2 pixel blend
    # MULTIPLIER -->  1 = no or mini AA, 2 = 2AA (2x2=4X samples)
    # OFFSET     -->  0 = 1:1, 1 = oversample by 1 pixel (use with RANGE 2, MULT 1 for mini AA)
    oversampling = config.str("oversampling", "8x")
    if oversampling == "none":
        app["oversample_range"] = 1
        app["oversample_multiplier"] = 1
        app["oversample_offset"] = 0
    elif oversampling == "mini":
        app["oversample_range"] = 2
        app["oversample_multiplier"] = 1
        app["oversample_offset"] = 1
    elif oversampling == "2x":
        app["oversample_range"] = 2
        app["oversample_multiplier"] = 2
        app["oversample_offset"] = 0
    elif oversampling == "4x":
        app["oversample_range"] = 4
        app["oversample_multiplier"] = 4
        app["oversample_offset"] = 0
    elif oversampling == "8x":
        app["oversample_range"] = 8
        app["oversample_multiplier"] = 8
        app["oversample_offset"] = 0
    else:
        return err("Unknown oversampling value: %s" % oversampling)
    print("Oversampling RANGE:", app["oversample_range"], "MULT:", app["oversample_multiplier"], "OFFSET:", app["oversample_offset"])

    # Calculate internal map dimensions
    app["map_width"] = DISPLAY_WIDTH * app["oversample_multiplier"] + app["oversample_offset"]  # Pixels samples per row
    app["map_height"] = DISPLAY_HEIGHT * app["oversample_multiplier"] + app["oversample_offset"]  # Pixel samples per column
    app["max_pixel_x"] = app["map_width"] - 1  # Maximum sample for x
    app["max_pixel_y"] = app["map_height"] - 1  # Maximum sample for y

    # Determine what POI to zoom onto
    app["target"] = 0, 0
    app["zoom_level"] = rnd(app) * 1000
    poi_type = config.str("poi", "search")
    if poi_type == "search":
        poi_id = timer_start(app, "poi")
        app["target"] = find_point_of_interest(app)  # Choose a point of interest
        app["desc"] = "random"
        timer_stop(app, "poi", poi_id)
    elif poi_type == "specific":
        app["target"] = float(config.str("poi_coord_real", 0)), float(config.str("poi_coord_imaginary", 0))
        app["desc"] = "your coordinates"
    else:
        return err("Unrecognized POI type: {}".format(poi_type))
    print("POI target:", app["target"][0], app["target"][1], "Desc:", app["desc"])
    print("Zoom Level:", app["zoom_level"])

    app['palette'] = config.str('palette', 'random')
    if app['palette'] != 'random' and app['palette'] != 'red' and app['palette'] != 'green' and app['palette'] != 'blue':
        return err("Unrecognized palette type: {}".format(app['palette']))
    print("Color Palette:", app['palette'])

    app['max_iter'] = int(MIN_ITER + app["zoom_level"] * ZOOM_TO_ITER)

    # Generate the animation with all frames
    frames = get_frames(app)

    timer_display(app)

    return render.Root(
        delay = 1500,
        child = render.Box(render.Animation(frames)),
    )

def get_frames(app):
    id = timer_start(app, "get_random_gradient")
    app["gradient"] = get_random_gradient(app)
    timer_stop(app, "get_random_gradient", id)

    print("Generating frame")
    frames = list()  # List to store frames of the animation
    id = timer_start(app, "render_mandelbrot")
    frame = render_mandelbrot(app, app["target"][0], app["target"][1])
    timer_stop(app, "render_mandelbrot", id)
    frames.append(frame)

    print(
        "Link:",
        "https://mandel.gart.nz/?Re={}&Im={}&iters={}&zoom={}&colourmap=5&maprotation=0&axes=0&smooth=0".format(
            app["target"][0],
            app["target"][1],
            app['max_iter'],
            app["zoom_level"] * 600, # approx ratio of my zoom to theirs
        ),
    )

    return frames

def float_range(start, end, num_steps, inclusive = False):
    step_size = (float(end) - float(start)) / num_steps
    result = []
    for i in range(num_steps):
        result.append(start + i * step_size)
    if inclusive:
        result.append(end)
    return result

def find_point_of_interest(app):
    print("Determining point of interest")
    max_iter = int(MIN_ITER + (app['zoom_level'] + 1) * ZOOM_TO_ITER)
    x, y, best = find_poi_near(app, CTRX, CTRY, 0.0, (MAXX - MINX), MAX_POI_SAMPLES, max_iter)
    print("Settled on POI:", x, y, "escape:", best)
    return x, y

def find_poi_near(app, x, y, esc, depth, num_samples, iter_limit):
    bestx, besty, best_escape = x, y, esc

    for _ in range(num_samples):
        x, y = bestx + (rnd(app) - 0.5) * depth, besty + (rnd(app) - 0.5) * depth
        _, last_escape = mandelbrot_calc(app, x, y, iter_limit)
        if last_escape < ESCAPE_THRESHOLD and last_escape > best_escape:
            bestx, besty, best_escape = x, y, last_escape

    #    print("Best POI so far:", bestx, besty, "escape:", best_escape)
    return bestx, besty, best_escape

# Map value v from one range to another
def map_range(v, min1, max1, min2, max2):
    return min2 + (max2 - min2) * (v - min1) / (max1 - min1)

# Performs the mandelbrot calculation on a single point
# Returns both the escape distance and the number of iterations
# (cannot exceed iter_limit)
def mandelbrot_calc(app, x, y, iter_limit):
    id = timer_start(app, "calc")

    # Initialize z (zr, zi) and c (cr, ci)
    x2, y2, w = 0.0, 0.0, 0.0

    # Use a for loop to simulate the behavior of a while loop
    for iteration in range(1, iter_limit + 1):
        # Check if the point has escaped
        if x2 + y2 > ESCAPE_THRESHOLD:
            timer_stop(app, "calc", id)
            return (iteration, x2 + y2)

        # Calculate new zr and zi (x and y in pseudocode)
        zr = x2 - y2 + x
        zi = w - x2 - y2 + y

        # Update squares and w for the next iteration
        x2 = zr * zr
        y2 = zi * zi
        w = (zr + zi) * (zr + zi)

    # End timing and return the max iteration and the final distance if escape condition is not met
    timer_stop(app, "calc", id)
    return (iter_limit, x2 + y2)

def int_to_hex(n):
    if n > 255:
        fail("Can't convert value " + str(n) + " to hex digit")
    hex_digits = "0123456789ABCDEF"
    return hex_digits[n // 16] + hex_digits[n % 16]

# Convert RGB values to a hexadecimal color code
def rgb_to_hex(r, g, b):
    return "#" + int_to_hex(r) + int_to_hex(g) + int_to_hex(b)

def get_gradient_color(app, iter):
    id = timer_start(app, "get_gradient_color")

    r, g, b = get_gradient_rgb(app, iter)
    color = rgb_to_hex(r, g, b)

    timer_stop(app, "get_gradient_color", id)
    return color

def dump(app):
    for y in range(app["map_height"]):
        row = ""
        for x in range(app["map_width"]):
            if get_pixel(app, x, y) < 0:
                row += " "
            else:
                row += "X"
        print(row)

def get_gradient_rgb(app, iter):
    if iter == app["max_iter"]:
        return (0, 0, 0)

    elif iter > app["max_iter"] or iter < 0:
        dump(app)
        fail("Bad iterations in get_gradient_rgb:", iter)

    # Convert iterations to a color
    t = (math.pow(math.log(iter), GRADIENT_SCALE_FACTOR) / NUM_GRADIENT_STEPS) % 1.0

    # Number of keyframes
    num_keyframes = len(app["gradient"]) - 1
    #print("Num keyframes:", num_keyframes)

    # Ensure we are covering the whole gradient range
    frame_pos = t * num_keyframes

    #print("Frame pos:", frame_pos)
    lower_frame = int(frame_pos)  # Index of the lower keyframe
    upper_frame = min(lower_frame + 1, num_keyframes)  # Index of the upper keyframe

    # Fractional part for interpolation between the two keyframes
    local_t = frame_pos - float(lower_frame)

    # Get the colors of the two keyframes to blend between
    color_start = app["gradient"][lower_frame]
    color_end = app["gradient"][upper_frame]

    # Perform linear interpolation (LERP) between the two colors
    r = int(color_start[0] + local_t * (color_end[0] - color_start[0]))
    g = int(color_start[1] + local_t * (color_end[1] - color_start[1]))
    b = int(color_start[2] + local_t * (color_end[2] - color_start[2]))

    return (r, g, b)

# Blends RGB colors together
# Also converts from quantized values to full color spectrum
def blend_rgbs(*rgbs):
    tr, tg, tb = 0, 0, 0
    count = 0
    for i in range(0, len(rgbs) - 1):
        r, g, b = rgbs[i]
        tr += r
        tg += g
        tb += b
        count += 1

    if count == 0:
        return rgb_to_hex(rgbs[0][0], rgbs[0][1], rgbs[0][2])
    elif count == 2:  # Lame optimization
        return rgb_to_hex(int(tr << 2 * CHANNEL_MULT), int(tg << 2 * CHANNEL_MULT), int(tb << 2 * CHANNEL_MULT))
    elif count == 4:  # Lame optimization
        return rgb_to_hex(int(tr << 4 * CHANNEL_MULT), int(tg << 4 * CHANNEL_MULT), int(tb << 4 * CHANNEL_MULT))

    return rgb_to_hex(int(tr / count * CHANNEL_MULT), int(tg / count * CHANNEL_MULT), int(tb / count * CHANNEL_MULT))

def random_color_tuple():
    return (random.number(0, MAX_COLOR_CHANNEL), random.number(0, MAX_COLOR_CHANNEL), random.number(0, MAX_COLOR_CHANNEL))


def get_random_gradient(app):
    id = timer_start(app, "get_random_gradient")
    pal_type = app["palette"]
    print("Generating {} gradient".format(pal_type))

    color = [0, 0, 0]
    primary_channel, channel2, channel3 = 0, 0, 0

    if pal_type == 'random':
        color = list(random_color_tuple())
    else:
        if pal_type == 'red':
            primary_channel = 0
        elif pal_type == 'green':
            primary_channel = 1
        elif pal_type == 'blue':
            primary_channel = 2
        color[primary_channel] = MAX_COLOR_CHANNEL
        channel2 = (primary_channel + 1) % 3
        channel3 = (primary_channel + 2) % 3
            
    gradient = []
    for step in range(0, NUM_GRADIENT_STEPS):
        if pal_type == "random":
            gradient.append(tuple(color))
            color = alter_color_rgb(color)
        else:
            # Deterministic calculation for the primary channel using sine wave
            phase = step / NUM_GRADIENT_STEPS * 2 * math.pi  # Full cycle for primary channel
            intensity = (math.sin(phase) + 1) / 2 * MAX_COLOR_CHANNEL

            # Set the primary channel with smooth transition
            color[primary_channel] = int(intensity)

            # Use independent frequencies for the secondary and tertiary channels
            freq2 = NUM_GRADIENT_STEPS * 0.75  # Slightly faster frequency for channel2
            freq3 = NUM_GRADIENT_STEPS * 0.5  # Even faster frequency for channel3

            # Apply independent sine variations for the secondary and tertiary channels
            color[channel2] = int(((math.sin(step / freq2 * 2 * math.pi) + 1) / 2) * MAX_COLOR_CHANNEL * 0.6)  # Adjust scale to 60%
            color[channel3] = int(((math.sin(step / freq3 * 2 * math.pi) + 1) / 2) * MAX_COLOR_CHANNEL * 0.4)  # Adjust scale to 40%

            gradient.append(tuple(color))

    timer_stop(app, "get_random_gradient", id)
    return gradient

# At least one channel flipped, another randomized
def alter_color_rgb(color):
    flip_idx = random.number(0, 2)
    rnd_idx = (flip_idx + random.number(1, 2)) % 3
    keep_idx = 3 - flip_idx - rnd_idx
    new_color = [0, 0, 0]
    new_color[flip_idx] = MAX_COLOR_CHANNEL - color[flip_idx]
    new_color[rnd_idx] = random.number(0, MAX_COLOR_CHANNEL)
    new_color[keep_idx] = color[keep_idx]
    return new_color

# Renders a line
# Returns the number of iterations found if they are all the same
# If match_iter is passed something other than False, then it will
# compare all iterations against this value
# Returns the number of iterations, or False if they are not all the same
def generate_line_opt(app, match_iter, pix, set, max_iter):
    id = timer_start(app, "generate_line_opt")
    xm_step, ym_step = 0, 0

    # Determine whether the line is vertical or horizontal
    is_vertical = pix["x1"] == pix["x2"]

    if is_vertical:
        start, end = pix["y1"], pix["y2"]

        # Precompute xm and the step size for ym to avoid repeated map_range calls
        xm = set["x1"]
        ym_step = (set["y2"] - set["y1"]) / (pix["y2"] - pix["y1"])
        ym = set["y1"]
    else:
        start, end = pix["x1"], pix["x2"]

        # Precompute ym and the step size for xm to avoid repeated map_range calls
        ym = set["y1"]
        xm_step = (set["x2"] - set["x1"]) / (pix["x2"] - pix["x1"])
        xm = set["x1"]

    xp, yp = pix["x1"], pix["y1"]

    for val in range(start, end + 1):
        # Update xm and ym incrementally without calling map_range
        if is_vertical:
            yp = val  # Update vertical position
            ym += ym_step  # Increment ym by precomputed step
        else:
            xp = val  # Update horizontal position
            xm += xm_step  # Increment xm by precomputed step

        # Get the pixel iteration count
        cache_id = timer_start(app, "generate_line_opt/generate_pixel")
        cache = generate_pixel(app, xp, yp, xm, ym, max_iter)
        timer_stop(app, "generate_line_opt/generate_pixel", cache_id)

        # Initialize match_iter on first iteration
        if match_iter == -1:
            match_iter = cache

            # Bail early: Not all iterations along the line were identical
        elif match_iter != cache:
            timer_stop(app, "generate_line_opt", id)
            return False

    # All iterations along the line were identical
    timer_stop(app, "generate_line_opt", id)
    return match_iter

# Copies an object and alters one field to a new value
def alt(obj, field, value):
    c = {}
    for k, v in obj.items():
        c[k] = v
    c[field] = value
    return c

def flood_fill(app, area, iter):
    id = timer_start(app, "flood_fill")

    for y in range(area["y1"], area["y2"] + 1):
        for x in range(area["x1"], area["x2"] + 1):
            app["map"][y * app["map_width"] + x] = iter  #set_pixel(app, x, y, iter)
    timer_stop(app, "flood_fill", id)

def generate_mandelbrot_area(app, pix, set, iter_limit):
    id = timer_start(app, "generate_mandelbrot_area")

    # Initialize the stack with the first region to process
    stack = [(pix, set)]

    # We will dynamically increase the stack in the loop
    for _ in range(MAX_INT):  # Why no while loop, damn you starlark
        if len(stack) == 0:
            break

        # Pop the last item from the stack
        current_pix, current_set = stack.pop()

        dmdp_id = timer_start(app, "generate_mandelbrot_area_dmdp")
        dxp, dyp = int(current_pix["x2"] - current_pix["x1"]), int(current_pix["y2"] - current_pix["y1"])
        dxm, dym = float(current_set["x2"] - current_set["x1"]) / float(dxp), float(current_set["y2"] - current_set["y1"]) / float(dyp)
        timer_stop(app, "generate_mandelbrot_area_dmdp", dmdp_id)

        # A small box can be filled in with the same color if the corners are identical
        done = False
        if dxp <= 6 and dyp <= 6:
            iter1 = generate_pixel(app, current_pix["x1"], current_pix["y1"], current_set["x1"], current_set["y1"], iter_limit)
            iter2 = generate_pixel(app, current_pix["x2"], current_pix["y2"], current_set["x2"], current_set["y2"], iter_limit)
            iter3 = generate_pixel(app, current_pix["x1"], current_pix["y2"], current_set["x1"], current_set["y2"], iter_limit)
            iter4 = generate_pixel(app, current_pix["x2"], current_pix["y1"], current_set["x2"], current_set["y1"], iter_limit)
            if iter1 == iter2 and iter2 == iter3 and iter3 == iter4:
                flood_fill(app, current_pix, iter1)
                done = True

        # A border with the same iterations can be filled with the same color
        if not done:
            iter = generate_line_opt(app, -1, alt(current_pix, "y2", current_pix["y1"]), alt(current_set, "y2", current_set["y1"]), iter_limit)
            if iter != False:
                iter = generate_line_opt(app, iter, alt(current_pix, "y1", current_pix["y2"]), alt(current_set, "y1", current_set["y2"]), iter_limit)
            if iter != False:
                iter = generate_line_opt(app, iter, alt(current_pix, "x2", current_pix["x1"]), alt(current_set, "x2", current_set["x1"]), iter_limit)
            if iter != False:
                iter = generate_line_opt(app, iter, alt(current_pix, "x1", current_pix["x2"]), alt(current_set, "x1", current_set["x2"]), iter_limit)
            if iter != False:
                flood_fill(app, current_pix, iter)
                done = True

        # Perform vertical split
        if not done and dyp >= 3 and dyp >= dxp:
            split_id = timer_start(app, "generate_mandelbrot_area_split")

            splityp = int(dyp / 2)
            syp_above = splityp + current_pix["y1"]
            syp_below = syp_above + 1
            sym_above = current_set["y1"] + splityp * dym
            sym_below = current_set["y1"] + (splityp + 1) * dym

            # Add sub-regions to the stack
            stack.append((alt(current_pix, "y1", syp_below), alt(current_set, "y1", sym_below)))
            stack.append((alt(current_pix, "y2", syp_above), alt(current_set, "y2", sym_above)))
            timer_stop(app, "generate_mandelbrot_area_split", split_id)

            # Perform horizontal split
        elif not done and dxp >= 3 and dyp >= 3:
            split_id = timer_start(app, "generate_mandelbrot_area_split")
            splitxp = int(dxp / 2)
            sxp_left = splitxp + current_pix["x1"]
            sxp_right = sxp_left + 1
            sxm_left = current_set["x1"] + splitxp * dxm
            sxm_right = current_set["x1"] + (splitxp + 1) * dxm
            timer_stop(app, "generate_mandelbrot_area_split", split_id)

            # Add sub-regions to the stack
            stack.append((alt(current_pix, "x1", sxp_right), alt(current_set, "x1", sxm_right)))
            stack.append((alt(current_pix, "x2", sxp_left), alt(current_set, "x2", sxm_left)))

            # This is a small area with differing iterations, calculate/mark them individually
        elif not done:
            final_generate_id = timer_start(app, "generate_mandelbrot_area final_generate_pixel")
            for offy in range(0, dyp + 1):
                for offx in range(0, dxp + 1):
                    generate_pixel(app, current_pix["x1"] + offx, current_pix["y1"] + offy, current_set["x1"] + (dxm * offx), current_set["y1"] + (dym * offy), iter_limit)
            timer_stop(app, "generate_mandelbrot_area final_generate_pixel", final_generate_id)

    timer_stop(app, "generate_mandelbrot_area", id)

# Calculates the number of iterations for a point on the map and returns it
# Tries to gather the pixel data from the cache if available
def generate_pixel(app, xp, yp, xm, ym, iter_limit):
    id = timer_start(app, "generate_pixel")

    stored_val = app["map"][yp * app["map_width"] + xp]  # get_pixel(app, xp, yp)
    if stored_val != -1:
        timer_stop(app, "generate_pixel", id)
        return stored_val

    # Normal mandelbrot calculation
    iter, _ = mandelbrot_calc(app, xm, ym, iter_limit)

    # print("m:", xm, ym, "p:", xp, yp, "iter:", iter)
    if iter == iter_limit:
        iter = app["max_iter"]

    # Save iterations for pixel in map
    app["map"][yp * app["map_width"] + xp] = iter  # set_pixel(app, xp, yp, iter)

    timer_stop(app, "generate_pixel", id)
    return iter

def set_pixel(app, xp, yp, value):
    id = timer_start(app, "set_pixel")
    # Check if xp and yp are within valid bounds
    # if xp < 0 or xp >= app['map_width'] or yp < 0 or yp >= app['map_height']:
    #     fail("Bad set_pixel(" + str(xp) + "," + str(yp) + ") call")

    app["map"][yp * app["map_width"] + xp] = value
    timer_stop(app, "set_pixel", id)

def get_pixel(app, xp, yp):
    id = timer_start(app, "get_pixel")
    # Check if xp and yp are within valid bounds
    # if xp < 0 or xp >= app['map_width'] or yp < 0 or yp >= app['map_height']:
    #     fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")

    value = app["map"][yp * app["map_width"] + xp]

    timer_stop(app, "get_pixel", id)
    return value

def create_empty_map(app):
    id = timer_start(app, "create_empty_map")
    map_size = app["map_width"] * app["map_height"]
    map = []
    for _ in range(map_size):
        map.append(-1)  # Manually append -1 for each entry
    app["map"] = map
    timer_stop(app, "create_empty_map", id)

def render_mandelbrot(app, x, y):
    iterations = int(MIN_ITER + app["zoom_level"] * ZOOM_TO_ITER)

    # Determine coordinates
    half_width = (MAXX - MINX) / app["zoom_level"] / 2.0
    half_height = (MAXY - MINY) / app["zoom_level"] / 2.0
    minx, miny = x - half_width, y - half_height
    maxx, maxy = x + half_width, y + half_height
    app["center"] = (x, y)

    # Create the map
    create_empty_map(app)
    # print("Current center point:", x, y, "Iter:", iterations)

    # Generate the map
    pix = {"x1": 0, "y1": 0, "x2": app["max_pixel_x"], "y2": app["max_pixel_y"]}
    set = {"x1": minx, "y1": miny, "x2": maxx, "y2": maxy}
    app["region"] = set
    generate_mandelbrot_area(app, pix, set, iterations)

    # Render the map to the display
    return render_display(app)

# Converts a map to a Tidbyt Column made up of Rows made up of Boxes
def render_display(app):
    rd_id = timer_start(app, "render_display")

    # Loop through each pixel in the display
    rows = list()
    osx, osy = 0, 0
    color = 0
    for _ in range(DISPLAY_HEIGHT):  # y
        row = list()
        next_color = ""
        run_length = 0
        osx = 0

        for _ in range(DISPLAY_WIDTH):  # x
            # Get color from single pixel
            if DISPLAY_WIDTH == app["map_width"]:
                iter = app["map"][osy * app["map_width"] + osx]  #get_pixel(app, osx, osy)
                rgb = get_gradient_rgb(app, iter)
                color = rgb_to_hex(int(rgb[0] * CHANNEL_MULT), int(rgb[1] * CHANNEL_MULT), int(rgb[2] * CHANNEL_MULT))

                # Get color by oversampling
            else:
                samples = []
                for offy in range(app["oversample_range"]):
                    for offx in range(app["oversample_range"]):
                        iter = app["map"][(osy + offy) * app["map_width"] + osx + offx]  # iter = get_pixel(app, osx + offx , osy + offy)
                        samples.append(iter)

                rgbs = []
                for sample in samples:
                    rgbs.append(get_gradient_rgb(app, sample))

                id = timer_start(app, "blend_rgbs")
                color = blend_rgbs(*rgbs)
                timer_stop(app, "blend_rgbs", id)

            # Add a 1x1 box with the appropriate color to the row
            if next_color == "":  # First color of row
                run_length = 1
                next_color = color
            elif color == next_color:  # Color run detected
                run_length += 1
            else:  # Color change
                add_box(row, run_length, next_color)
                run_length = 1
                next_color = color

            osx += app["oversample_multiplier"]
        osy += app["oversample_multiplier"]

        # Add the row to the grid
        id = timer_start(app, "add_box")
        add_box(row, run_length, color)  # Add last box for row
        timer_stop(app, "add_box", id)
        rows.append(render.Row(children = row))

    timer_stop(app, "render_display", rd_id)

    return render.Column(
        children = rows,
    )

def add_box(row, run_length, color):
    if run_length == 1 and color == BLACK_PIXEL:
        row.append(BLACK_PIXEL)
    else:
        row.append(render.Box(width = run_length, height = 1, color = color))

def rnd(app):
    id = timer_start(app, "rnd")
    val = float(random.number(0, MAX_INT)) / float(MAX_INT)
    timer_stop(app, "rnd", id)
    return val

def get_schema():
    return schema.Schema(
        version = "1",
        fields = [
            schema.Dropdown(
                id = "oversampling",
                name = "Oversampling",
                desc = "Oversampling Method",
                icon = "border-none",
                default = "none",
                options = [
                    schema.Option(value = "none", display = "None"),
                    schema.Option(value = "mini", display = "Mini AA (slow)"),
                    schema.Option(value = "2x", display = "2X AA (slower)"),
                    schema.Option(value = "4x", display = "4X AA (much slower)"),
                    schema.Option(value = "8x", display = "8X AA (imagine even slower)"),
                ],
            ),
            schema.Dropdown(
                id = "palette",
                name = "Palette",
                desc = "Color palette/gradient",
                default = "random",
                icon = "palette",
                options = [
                    schema.Option(value = "random", display = "Random"),
                    schema.Option(value = "red", display = "Red-ish"),
                    schema.Option(value = "blue", display = "Blue-ish"),
                    schema.Option(value = "green", display = "Green-ish"),
                ],
            ),
            schema.Dropdown(
                id = "quantize",
                name = "Color Quantization",
                desc = "Shades per color channel",
                icon = "umbrella_beach",
                default = "8",
                options = [
                    schema.Option(value = "4", display = "4 shades"),
                    schema.Option(value = "8", display = "8 shades"),
                    schema.Option(value = "16", display = "16 shades"),
                    schema.Option(value = "32", display = "32 shades"),
                    schema.Option(value = "64", display = "64 shades"),
                    schema.Option(value = "128", display = "128 shades"),
                    schema.Option(value = "255", display = "256 shades"),
                ],
            ),
            schema.Dropdown(
                id = "poi",
                name = "POI",
                desc = "Point of interest (where to focus)",
                icon = "arrows-to-eye",
                default = "search",
                options = [
                    schema.Option(value = "search", display = "Search for new POI"),
                    schema.Option(value = "specific", display = "Entered coordinates"),
                    schema.Option(value = "popular", display = "Random popular POI"),
                ],
            ),
            schema.Generated(
                id = "poi_options",
                source = "poi",
                handler = poi_options,
            ),
        ],
    )

def poi_options(type):
    if type == "specific":
        return [
            schema.Text(
                id = "poi_coord_real",
                name = "Real",
                desc = "Real POI coordinate",
                icon = "hashtag",
                default = "-1.5",
            ),
            schema.Text(
                id = "poi_coord_imaginary",
                name = "Imaginary",
                desc = "Imaginary POI coordinate",
                icon = "hashtag",
                default = "0.0",
            ),
            schema.Text(
                id = "zoom_level",
                name = "Zoom Level",
                desc = "Initial zoom level",
                icon = "hashtag",
                default = "1.0",
            ),
        ]
    elif type == "search":
        return [
            schema.Text(
                id = "poi_points",
                name = "POI Points",
                desc = "Number of random points to search",
                icon = "hashtag",
                default = "10000",
            ),
            schema.Text(
                id = "zoom_level_min",
                name = "Zoom Min",
                desc = "Initial magnification (at minimum)",
                icon = "hashtag",
                default = "10.0",
            ),
            schema.Text(
                id = "zoom_level_mult",
                name = "Zoom Random",
                desc = "Additional magnification (random amount)",
                icon = "hashtag",
                default = "1000.0",
            ),
        ]        

    # Popular have no custom options
    return []

def err(msg):
    return render.Root(
        render.WrappedText(
            content = msg,
            width = 64,
            color = "#f00",
        ),
    )

###
# TIMERS
###

def timer_start(app, category):
    if "profiling" not in app:
        app["profiling"] = {}

    # Ensure category exists
    if category not in app["profiling"]:
        app["profiling"][category] = {
            "elapsed": 0,  # Store total elapsed time
            "timers": [],  # Store individual timer start times
        }

    # Get the current time in nanoseconds and store it as a unique timer
    start_time_ns = time.now().unix_nano
    timer_id = len(app["profiling"][category]["timers"])
    app["profiling"][category]["timers"].append(start_time_ns)

    # Return the timer ID for reference in end_time
    return timer_id

def timer_stop(app, category, timer_id):
    if "profiling" not in app or category not in app["profiling"]:
        fail("Must call start before end for category '{}'".format(category))

    # Ensure the timer_id is valid and exists
    timers = app["profiling"][category]["timers"]
    if timer_id >= len(timers) or timers[timer_id] == None:
        fail("Invalid timer_id '{}' for category '{}'".format(timer_id, category))

    # Get the current time as end time and calculate elapsed nanoseconds
    end_time_ns = time.now().unix_nano
    elapsed = end_time_ns - timers[timer_id]

    # Add the elapsed time to the total and mark the timer as stopped
    app["profiling"][category]["elapsed"] += elapsed
    app["profiling"][category]["timers"][timer_id] = None  # Mark timer as ended

def timer_display(app):
    # Display all categories' cumulative profiling results
    if "profiling" not in app:
        print("No profiling data available.")

    result = "PROFILE\n"

    for category in app["profiling"]:
        elapsed_time_ns = app["profiling"][category]["elapsed"]  # Total elapsed time in nanoseconds

        if elapsed_time_ns >= 1000000000:  # If more than or equal to 1 second
            elapsed_time_s = elapsed_time_ns / 1000000000.0  # Convert to seconds
            elapsed_time_s_rounded = math.round(elapsed_time_s * 1000) / 1000  # Round to 3 decimal places
            result += " + '{}': {} sec\n".format(category, elapsed_time_s_rounded)
        elif elapsed_time_ns >= 1000000:  # If more than or equal to 1 millisecond
            elapsed_time_ms = elapsed_time_ns / 1000000.0  # Convert to milliseconds
            elapsed_time_ms_rounded = math.round(elapsed_time_ms * 1000) / 1000  # Round to 3 decimal places
            result += " + '{}': {} ms\n".format(category, elapsed_time_ms_rounded)
        else:
            result += " + '{}': {} ns\n".format(category, elapsed_time_ns)  # Display nanoseconds directly

    print(result)
