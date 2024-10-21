# Next Step:
# - POI Nudging Routine:
#   + Grid up area of POI (say 8x4)
#   + Add extra points outside of area (an additional 4x2 perhaps)
#   + Perform fitness scores of centered at 0,0 +/- 2,1 and keep best
# - Adaptive AA:
#   + Start with 64x32 display
#   + If "small area with differing iterations" oversample that region
#   + Stop at max oversample
#   + Combine with timeout prediction at calc all oversamples at the same level before
#     sampling any deeper to dynamically stop when timeout is nigh
# - Time out prediction:
#   + Time mandelbrot calc for broad POI check
#   + Extrapolate to estimate time to calculate 64x32 no supersample
#   + Extrapolate to estimate depth of supersample possible under time limit
#   + Use to adjust zoom / super sample / iteration limit / POI search points
#
load("math.star", "math")
load("random.star", "random")
load("render.star", "render")
load("schema.star", "schema")
load("time.star", "time")

MIN_ITER = 100  # minimum iterations, raise if initial zoom is > 1
ZOOM_TO_ITER = 1.0  # 1.0 standard, less for faster calc, more for better accuracy
ESCAPE_THRESHOLD = 4.0  # 4.0 standard, less for faster calc, less accuracy
DISPLAY_WIDTH = 64  # Tidbyt is 64 pixels wide
DISPLAY_HEIGHT = 32  # Tidbyt is 32 pixels high
NUM_GRADIENT_STEPS = 64  # Higher = more color variation
GRADIENT_SCALE_FACTOR = 1.55  # 1.55 = standard, less for more colors zoomed in, more for few colors zoomed in
CTRX, CTRY = -0.75, 0  # mandelbrot center
MINX, MINY, MAXX, MAXY = -2.5, -0.875, 1.0, 0.8753  # Bounds to use for mandelbrot set
BLACK_COLOR = "#000000"  # Shorthand for black color
MAX_INT = int(math.pow(2, 53))  # Guesstimate for Starlark max_int
BLACK_PIXEL = render.Box(width = 1, height = 1, color = BLACK_COLOR)  # Pregenerated 1x1 pixel black box
POI_GRID_X = 8
POI_GRID_Y = 4
POI_ZOOM = DISPLAY_WIDTH / POI_GRID_X
POI_MAX_ZOOM = 100000
POI_MAX_TIME = 4  # Don't exceed POI_MAX_TIME seconds searching for POI
BRIGHTNESS_MIN = 16 # For brightness normalization, don't bring channel below this
GAMMA_CORRECTION = 1.1 # Mild gamma correction seems to work best

def main(config):
    seed = 1729531646 #time.now().unix
    random.seed(seed)
    print("Using random seed:", seed)
    app = {"config": config}

    # RANGE      -->  1 = 1x1 pixel no blending, 2 = 2x2 pixel blend
    # MULTIPLIER -->  1 = no or mini AA, 2 = 2AA (2x2=4X samples)
    # OFFSET     -->  0 = 1:1, 1 = oversample by 1 pixel (use with RANGE 2, MULT 1 for mini AA)
    oversampling = config.str("oversampling", "4x")
    if oversampling == "none":
        app["oversample"] = 1
    elif oversampling == "2x":
        app["oversample"] = 2
    elif oversampling == "4x":
        app["oversample"] = 4
    elif oversampling == "8x":
        app["oversample"] = 8
    else:
        return err("Unknown oversampling value: %s" % oversampling)
    print("Oversampling:", app["oversample"])

    # Calculate internal map dimensions
    app["map_width"] = DISPLAY_WIDTH * app["oversample"]  # Pixels samples per row
    app["map_height"] = DISPLAY_HEIGHT * app["oversample"]  # Pixel samples per column
    app["max_pixel_x"] = app["map_width"] - 1  # Maximum sample for x
    app["max_pixel_y"] = app["map_height"] - 1  # Maximum sample for y

    # Determine what POI to zoom onto
    app["target"] = 0, 0
    choose_poi(app)  # Choose a point of interest
    print("Zoom Level:", app["zoom_level"])

    app["palette"] = config.str("palette", "random")
    if app["palette"] != "random" and app["palette"] != "red" and app["palette"] != "green" and app["palette"] != "blue":
        return err("Unrecognized palette type: {}".format(app["palette"]))
    print("Color Palette:", app["palette"])

    # Generate the animation with all frames
    frames = get_frames(app)

    return render.Root(
        delay = 1000,
        child = render.Box(render.Animation(frames)),
    )

# Ya know, it uh, normalizes ... the brightness
def normalize_brightness(rgb_map):
    # Determine low and high channel values
    lowest_channel = 255
    highest_channel = 0
    for i in range(len(rgb_map['data'])):
        channels = hex_to_rgb(rgb_map['data'][i])
        for c in range(len(channels)):
            if channels[c] < lowest_channel:
                lowest_channel = channels[c]
            elif channels[c] > highest_channel:
                highest_channel = channels[c]

    # Determine adjustment
    subtraction = 0
    if lowest_channel > BRIGHTNESS_MIN:
        subtraction = lowest_channel - BRIGHTNESS_MIN
    multiplier = (255.0 + subtraction) / highest_channel
    
    print ("Normalizing brightness ... Lowest:", lowest_channel, "Highest:", highest_channel, "Subtract:", subtraction, "Mult:", multiplier)

    # Normalize data
    for i in range(len(rgb_map['data'])):
        channels = list(hex_to_rgb(rgb_map['data'][i]))
        for c in range(len(channels)):
            channels[c] = int(channels[c] * multiplier - subtraction)
        rgb_map['data'][i] = rgb_to_hex(*channels)

# Apply gamma correction
def gamma_correction(rgb_map):
    print ("Applying gamma correction of", GAMMA_CORRECTION)

    for i in range(len(rgb_map['data'])):
        channels = list(hex_to_rgb(rgb_map['data'][i]))
        for c in range(len(channels)):
            channels[c] = int(math.pow(channels[c] / 255.0, 1 / GAMMA_CORRECTION) * 255)
        rgb_map['data'][i] = rgb_to_hex(*channels)

def get_frames(app):
    print("Generating frame")

    app["gradient"] = get_random_gradient(app['palette'])

    # This was a loop for an animation, but the animation takes too long
    # to render on Tidbyt servers with any degree of quality
    frames = list()  # List to store frames of the animation
    rgb_map = render_mandelbrot(app, app["target"][0], app["target"][1])
    normalize_brightness(rgb_map)
    if GAMMA_CORRECTION > 1.0:
        gamma_correction(rgb_map)
    tidbytMap = render_tidbyt(rgb_map)
    frames.append(tidbytMap)

    print_link(app["target"][0], app["target"][1], app["max_iter"], app["zoom_level"])
    return frames

# Makes for easier debugging to compare results to an existing renderer
def print_link(x, y, iter, zoom):
    print(
        "https://mandel.gart.nz/?Re={}&Im={}&iters={}&zoom={}&colourmap=5&maprotation=0&axes=0&smooth=0".format(
            x,
            y,
            iter,
            str(int(zoom * 600)),
        ),
    )

def float_range(start, end, num_steps, inclusive = False):
    step_size = (float(end) - float(start)) / num_steps
    result = []
    for i in range(num_steps):
        result.append(start + i * step_size)
    if inclusive:
        result.append(end)
    return result

def choose_poi(app):
    # Find a respectable point of interest
    print("Determining point of interest")
    x, y, best, zoom = find_poi()

    # Make it our target
    print("Settled on POI:", x, y, "score:", best)
    app["target"] = x, y
    app["zoom_level"] = zoom
    app["max_iter"] = int(MIN_ITER + zoom * ZOOM_TO_ITER)

# Finds a respectable point of interest ... allgedly
def find_poi():
    start_time = time.now().unix
    end_time = start_time + POI_MAX_TIME
    bestx, besty, best_score, best_zoom = 0, 0, 0, 1
    search_areas = []
    search_areas.append((MINX, MINY, MAXX, MAXY, 1))

    for _ in range(MAX_INT):  # Woe, there be no while loops
        if time.now().unix > end_time:
            break
        if len(search_areas) == 0:
            break

        minx, miny, maxx, maxy, zoom = search_areas.pop()
        iter_limit = int(MIN_ITER + zoom * ZOOM_TO_ITER)

        # Grid up area
        cell_width = (maxx - minx) / POI_GRID_X
        cell_height = (maxy - miny) / POI_GRID_Y
        cell_width_half = cell_width / 2.0
        cell_height_half = cell_height / 2.0
        score = 0
        best_iter_in_grid = 0

        # Divide area into grid cells and evaluate each
        for y in range(POI_GRID_Y):
            if time.now().unix > end_time:
                break
            for x in range(POI_GRID_X):
                if time.now().unix > end_time:
                    break
                sampx = (rnd() - 0.5 + x) * cell_width + minx
                sampy = (rnd() - 0.5 + y) * cell_height + miny
                iter, _ = mandelbrot_calc(sampx, sampy, iter_limit)
                if iter >= iter_limit:  # Did not escape
                    continue
                if zoom > POI_MAX_ZOOM:  # At max zoom level
                    continue

                new_min_x = sampx - cell_width_half
                new_min_y = sampy - cell_height_half
                new_max_x = sampx + cell_width_half
                new_max_y = sampy + cell_height_half
                new_zoom = zoom * POI_ZOOM
                score += iter

                # Only zoom in if we didn't find a better candidate in the grid
                if iter > best_iter_in_grid:
                    best_iter_in_grid = iter
                    search_areas.append((new_min_x, new_min_y, new_max_x, new_max_y, new_zoom))

        # Look at the combined score of all the grid cells
        # If this the best one so far, note it!
        if score > best_score:
            bestx, besty, best_score, best_zoom = (minx + maxx) / 2, (maxy + miny) / 2, score, zoom
            print("Found better POI:", bestx, besty, "score:", best_score, "zoom:", best_zoom)
            print_link(bestx, besty, iter_limit, best_zoom)

    # In theory this should stop shortly after hitting POI_MAX_TIME
    # But sometimes the Tidbyt servers throttle their apps
    # So I've gotta sprinkle this check everywhere
    if time.now().unix > end_time:
        print("Exceeded time limit for POI search:", (time.now().unix - start_time), "seconds")

    return bestx, besty, best_score, best_zoom

# Performs the mandelbrot calculation on a single point
# Returns both the escape distance and the number of iterations
# (cannot exceed iter_limit)
def mandelbrot_calc(x, y, iter_limit):
    # Initialize z (zr, zi) and c (cr, ci)
    x2, y2, w = 0.0, 0.0, 0.0

    # Use a for loop to simulate the behavior of a while loop
    for iteration in range(1, iter_limit + 1):
        # Check if the point has escaped
        if x2 + y2 > ESCAPE_THRESHOLD:
            return (iteration, x2 + y2)

        # Calculate new zr and zi (x and y in pseudocode)
        zr = x2 - y2 + x
        zi = w - x2 - y2 + y

        # Update squares and w for the next iteration
        x2 = zr * zr
        y2 = zi * zi
        w = (zr + zi) * (zr + zi)

    # End timing and return the max iteration and the final distance if escape condition is not met
    return (iter_limit, x2 + y2)

def int_to_hex(n):
    if n > 255:
        fail("Can't convert value " + str(n) + " to hex digit")
    hex_digits = "0123456789ABCDEF"
    return hex_digits[n // 16] + hex_digits[n % 16]

# Convert RGB values to a hexadecimal color code
def rgb_to_hex(r, g, b):
    return "#" + int_to_hex(r) + int_to_hex(g) + int_to_hex(b)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)    

def get_gradient_rgb(gradient, max_iter, iter):
    if iter == max_iter:
        return (0, 0, 0)

    elif iter > max_iter or iter < 0:
        fail("Bad iterations in get_gradient_rgb:", iter)

    # Convert iterations to a color
    t = (math.pow(math.log(iter), GRADIENT_SCALE_FACTOR) / NUM_GRADIENT_STEPS) % 1.0

    # Number of keyframes
    num_keyframes = len(gradient) - 1
    #print("Num keyframes:", num_keyframes)

    # Ensure we are covering the whole gradient range
    frame_pos = t * num_keyframes

    #print("Frame pos:", frame_pos)
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
        return rgb_to_hex(int(tr << 2), int(tg << 2), int(tb << 2))
    elif count == 4:  # Lame optimization
        return rgb_to_hex(int(tr << 4), int(tg << 4), int(tb << 4))

    return rgb_to_hex(int(tr / count), int(tg / count), int(tb / count))

def get_random_gradient(pal_type):
    color = [0, 0, 0]
    primary_channel, channel2, channel3 = 0, 0, 0

    if pal_type == "random":
        color = list(random_color_tuple())
    else:
        if pal_type == "red":
            primary_channel = 0
        elif pal_type == "green":
            primary_channel = 1
        elif pal_type == "blue":
            primary_channel = 2
        color[primary_channel] = 255
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
            intensity = (math.sin(phase) + 1) / 2 * 255

            # Set the primary channel with smooth transition
            color[primary_channel] = int(intensity)

            # Use independent frequencies for the secondary and tertiary channels
            freq2 = NUM_GRADIENT_STEPS * 0.75  # Slightly faster frequency for channel2
            freq3 = NUM_GRADIENT_STEPS * 0.5  # Even faster frequency for channel3

            # Apply independent sine variations for the secondary and tertiary channels
            color[channel2] = int(((math.sin(step / freq2 * 2 * math.pi) + 1) / 2) * 255 * 0.6)  # Adjust scale to 60%
            color[channel3] = int(((math.sin(step / freq3 * 2 * math.pi) + 1) / 2) * 255 * 0.4)  # Adjust scale to 40%

            gradient.append(tuple(color))

    return gradient

# At least one channel flipped, another randomized
def alter_color_rgb(color):
    flip_idx = random.number(0, 2)
    rnd_idx = (flip_idx + random.number(1, 2)) % 3
    keep_idx = 3 - flip_idx - rnd_idx
    new_color = [0, 0, 0]
    new_color[flip_idx] = 255 - color[flip_idx]
    new_color[rnd_idx] = random.number(0, 255)
    new_color[keep_idx] = color[keep_idx]
    return new_color

# Renders a line
# Returns the number of iterations found if they are all the same
# If match_iter is passed something other than False, then it will
# compare all iterations against this value
# Returns the number of iterations, or False if they are not all the same
def generate_line_opt(iter_map, max_iter, match_iter, pix, set, iter_limit):
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
        cache = generate_pixel(iter_map, max_iter, xp, yp, xm, ym, iter_limit)

        # Initialize match_iter on first iteration
        if match_iter == -1:
            match_iter = cache

            # Bail early: Not all iterations along the line were identical
        elif match_iter != cache:
            return False

    # All iterations along the line were identical
    return match_iter

# Copies an object and alters one field to a new value
def alt(obj, field, value):
    c = {}
    for k, v in obj.items():
        c[k] = v
    c[field] = value
    return c

def flood_fill(iter_map, area, iter):
    for y in range(area["y1"], area["y2"] + 1):
        for x in range(area["x1"], area["x2"] + 1):
            iter_map['data'][y * iter_map['width'] + x] = iter

def generate_mandelbrot_area(iter_map, max_iter, orig_pix, orig_set, iter_limit):
    # Initialize the stack with the first region to process
    stack = [(orig_pix, orig_set)]

    # We will dynamically increase the stack in the loop
    for _ in range(MAX_INT):  # Why no while loop, damn you starlark
        if len(stack) == 0:
            break

        # Pop the last item from the stack
        pix, set = stack.pop()

        # Determine some deltas
        dxp, dyp = int(pix["x2"] - pix["x1"]), int(pix["y2"] - pix["y1"])
        dxm, dym = float(set["x2"] - set["x1"]) / float(dxp), float(set["y2"] - set["y1"]) / float(dyp)

        # A small box can be filled in with the same color if the corners are identical
        done = False
        if dxp <= 6 and dyp <= 6:
            iter1 = generate_pixel(iter_map, max_iter, pix["x1"], pix["y1"], set["x1"], set["y1"], iter_limit)
            iter2 = generate_pixel(iter_map, max_iter, pix["x2"], pix["y2"], set["x2"], set["y2"], iter_limit)
            iter3 = generate_pixel(iter_map, max_iter, pix["x1"], pix["y2"], set["x1"], set["y2"], iter_limit)
            iter4 = generate_pixel(iter_map, max_iter, pix["x2"], pix["y1"], set["x2"], set["y1"], iter_limit)
            if iter1 == iter2 and iter2 == iter3 and iter3 == iter4:
                flood_fill(iter_map, pix, iter1)
                done = True

        # A border with the same iterations can be filled with the same color
        if not done:
            iter = generate_line_opt(iter_map, max_iter, -1, alt(pix, "y2", pix["y1"]), alt(set, "y2", set["y1"]), iter_limit)
            if iter != False:
                iter = generate_line_opt(iter_map, max_iter, iter, alt(pix, "y1", pix["y2"]), alt(set, "y1", set["y2"]), iter_limit)
            if iter != False:
                iter = generate_line_opt(iter_map, max_iter, iter, alt(pix, "x2", pix["x1"]), alt(set, "x2", set["x1"]), iter_limit)
            if iter != False:
                iter = generate_line_opt(iter_map, max_iter, iter, alt(pix, "x1", pix["x2"]), alt(set, "x1", set["x2"]), iter_limit)
            if iter != False:
                flood_fill(iter_map, pix, iter)
                done = True

        # Perform vertical split
        if not done and dyp >= 3 and dyp >= dxp:
            splityp = int(dyp / 2)
            syp_above = splityp + pix["y1"]
            syp_below = syp_above + 1
            sym_above = set["y1"] + splityp * dym
            sym_below = set["y1"] + (splityp + 1) * dym

            # Add sub-regions to the stack
            stack.append((alt(pix, "y1", syp_below), alt(set, "y1", sym_below)))
            stack.append((alt(pix, "y2", syp_above), alt(set, "y2", sym_above)))

            # Perform horizontal split
        elif not done and dxp >= 3 and dyp >= 3:
            splitxp = int(dxp / 2)
            sxp_left = splitxp + pix["x1"]
            sxp_right = sxp_left + 1
            sxm_left = set["x1"] + splitxp * dxm
            sxm_right = set["x1"] + (splitxp + 1) * dxm

            # Add sub-regions to the stack
            stack.append((alt(pix, "x1", sxp_right), alt(set, "x1", sxm_right)))
            stack.append((alt(pix, "x2", sxp_left), alt(set, "x2", sxm_left)))

            # This is a small area with differing iterations, calculate/mark them individually
        elif not done:
            for offy in range(0, dyp + 1):
                for offx in range(0, dxp + 1):
                    generate_pixel(iter_map, max_iter, pix["x1"] + offx, pix["y1"] + offy, set["x1"] + (dxm * offx), set["y1"] + (dym * offy), iter_limit)

# Calculates the number of iterations for a point on the map and returns it
# Tries to gather the pixel data from the cache if available
def generate_pixel(iter_map, max_iter, xp, yp, xm, ym, iter_limit):
    stored_val = iter_map['data'][yp * iter_map['width'] + xp] 
    if stored_val != -1:
        return stored_val

    # Normal mandelbrot calculation
    iter, _ = mandelbrot_calc(xm, ym, iter_limit)

    # print("m:", xm, ym, "p:", xp, yp, "iter:", iter)
    if iter == iter_limit:
        iter = max_iter

    # Save iterations for pixel in map
    iter_map['data'][yp * iter_map['width'] + xp] = iter 

    return iter

def create_map(width, height, fill = -1):
    data_size = width * height
    data = []
    for _ in range(data_size):
        data.append(fill)
    map_obj = { 'data': data, 'width': width, 'height': height }
    return map_obj

def render_mandelbrot(app, x, y):
    # Determine coordinates
    half_width = (MAXX - MINX) / app['zoom_level'] / 2.0
    half_height = (MAXY - MINY) / app['zoom_level'] / 2.0
    minx, miny = x - half_width, y - half_height
    maxx, maxy = x + half_width, y + half_height
    app['center'] = (x, y)

    # Create the map
    map = create_map(app['map_width'], app['map_height'])
    app['map'] = map

    # Generate the map
    pix = {'x1': 0, 'y1': 0, 'x2': app['max_pixel_x'], 'y2': app['max_pixel_y']}
    set = {'x1': minx, 'y1': miny, 'x2': maxx, 'y2': maxy}
    generate_mandelbrot_area(app['map'], app['max_iter'], pix, set, app['max_iter'])

    # Render an iteration map to an RGB map of the display size
    rgb_map = downsample(app['map'], DISPLAY_WIDTH, app['max_iter'], app['gradient'])
    return rgb_map

# Convert iteration map data to RGB map data and downsample
def downsample(iter_map, final_width, max_iter, gradient):
    osx, osy = 0, 0
    color = 0
    src_height = len(iter_map['data']) / iter_map['width']
    oversample = int(iter_map['width'] / src_height)
    final_height = int(final_width / oversample)
    rgb_map = create_map(final_width, final_height)

    for y in range(final_height):  # y
        osx = 0

        for x in range(final_width):  # x
            # Get color from single pixel, no oversampling
            if oversample == 1:
                iter = iter_map['data'][osy * iter_map['width'] + osx] 
                rgb = get_gradient_rgb(gradient, max_iter, iter)
                color = rgb_to_hex(int(rgb[0]), int(rgb[1]), int(rgb[2]))

            # Get color by oversampling
            else:
                samples = []
                for offy in range(oversample):
                    for offx in range(oversample):
                        iter = iter_map['data'][(osy + offy) * iter_map['width'] + osx + offx] 
                        samples.append(iter)

                rgbs = []
                for sample in samples:
                    rgbs.append(get_gradient_rgb(gradient, max_iter, sample))

                color = blend_rgbs(*rgbs)
                rgb_map['data'][y * final_width + x] = color

            osx += oversample
        osy += oversample
    
    return rgb_map # Returns rgb_map

# Converts an rgb map to a Tidbyt Column made up of Rows made up of Boxes
def render_tidbyt(rgb_map):
    # Loop through each pixel in the display
    rows = list()
    color = 0
    index = 0

    if len(rgb_map['data']) != DISPLAY_WIDTH * DISPLAY_HEIGHT:
        return err("Final map must be display size" + len(rgb_map['data']), False)

    for _ in range(DISPLAY_HEIGHT):  # y
        row = list()
        next_color = ""
        run_length = 0

        for _ in range(DISPLAY_WIDTH):  # x
            color = rgb_map['data'][index]
            index +=1

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

        # Add the row to the grid
        add_box(row, run_length, color)  # Add last box for row
        rows.append(render.Row(children = row))

    return render.Column(
        children = rows,
    )

def add_box(row, run_length, color):
    if run_length == 1 and color == BLACK_PIXEL:
        row.append(BLACK_PIXEL)
    else:
        row.append(render.Box(width = run_length, height = 1, color = color))

def random_color_tuple():
    return (random.number(0,255), random.number(0,255), random.number(0,255))

def rnd():
    return float(random.number(0, MAX_INT)) / float(MAX_INT)

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

def err(msg, include_root = True):
    text = render.WrappedText(
        content = msg,
        width = 64,
        color = "#f00",
    )

    if include_root:
        return render.Root(text)
    
    return text
