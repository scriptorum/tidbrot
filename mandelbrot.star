#
# TODO
# - Expose some optimization controls 
# - Derive certain variables from a budgeting system
#    + Measure the time used to find a POI
#    + If not a lot of time, display a single frame with high AA
#    + Otherwise do an animation
# 

load("random.star", "random")
load("render.star", "render")
load("time.star", "time")
load("math.star", "math")

ZOOM_GROWTH = 1.04              # 1 = no zoom in, 1.1 = 10% zoom per frame
FRAME_DURATION_MS = 150         # milliseconds per frame; for FPS, use value = 1000/fps
MIN_ITER = 30                   # minimum iterations, raise if initial zoom is > 1
OVERSAMPLE_RANGE = 1            # 1 = 1x1 pixel no blending, 2 = 2x2 pixel blend
OVERSAMPLE_MULTIPLIER = 1       # 1 = no AA, 2 = 2AA (2x2=4X samples)
OVERSAMPLE_OFFSET = 0           # 0 = 1:1, 1 = oversample by 1 pixel (use with RANGE 2, MULT 1 for mini AA)
ESCAPE_THRESHOLD = 4.0          # 4.0 standard, less for faster calc, less accuracy
ZOOM_TO_ITER = 1.0              # 1.0 standard, less for faster calc, less accuracy
DISPLAY_WIDTH = 64              # Tidbyt is 64 pixels wide
DISPLAY_HEIGHT = 32             # Tidbyt is 32 pixels high
NUM_GRADIENT_STEPS = 64         # Higher = more color variation
GRADIENT_SCALE_FACTOR = 1.55    # 1.55 = standard, less for more colors zoomed in, more for few colors zoomed in
MAX_POI_SAMPLES = 100000        # Number of random points to check for POI-worthiness
INITIAL_POSITION = 1.0          # 0 = start at 0,0, 1=start at POI, otherwise = blended start
TRANSLATE_PCT = 0.0             # 0=no movement, 0.9=adjust 10% towards target POI (only if INITIAL_POSITION < 1.0)
CTRX, CTRY = -0.75, 0           # mandelbrot center
MINX, MINY, MAXX, MAXY = -2.5, -0.875, 1.0, 0.8753  # Bounds to use for mandelbrot set
MAX_COLORS = 8                                      # Max quantized channel values (helps reduce Image Too Large errors)
CHANNEL_MULT = 255.9999 / MAX_COLORS                # Conversion from quantized value to full range color channel (0-255)

MAX_FRAMES = int(15000 / FRAME_DURATION_MS)         # Calc total frames in animation
MAX_ZOOM = math.pow(ZOOM_GROWTH, MAX_FRAMES)        # Calc max zoom
MAX_ITER_CALC = int(math.round(MIN_ITER + ZOOM_TO_ITER * math.pow(ZOOM_GROWTH, MAX_FRAMES)) + 1)    # Calc max iter
MAP_WIDTH  =  DISPLAY_WIDTH * OVERSAMPLE_MULTIPLIER + OVERSAMPLE_OFFSET  # Pixels samples per row
MAP_HEIGHT = DISPLAY_HEIGHT * OVERSAMPLE_MULTIPLIER + OVERSAMPLE_OFFSET  # Pixel samples per column
MAX_PIXEL_X = MAP_WIDTH - 1                  # Maximum sample for x
MAX_PIXEL_Y = MAP_HEIGHT - 1                 # Maximum sample for y
BLACK_COLOR = "#000000"                             # Shorthand for black color
MAX_INT = int(math.pow(2, 53))                      # Guesstimate for Starlark max_int
BLACK_PIXEL = render.Box(width=1, height=1, color=BLACK_COLOR)                  # Pregenerated 1x1 pixel black box

def main(config):
    random.seed(1)
    app = {"config": config}
    # print("Display:", DISPLAY_WIDTH, DISPLAY_HEIGHT, "MapSize:", MAP_WIDTH, MAP_HEIGHT, "MaxPixel:", MAX_PIXEL_X, MAX_PIXEL_Y)

    # MAX_ZOOM_CALC = float(MAX_ITER_CALC - MIN_ITER) / ZOOM_TO_ITER + INITIAL_ZOOM_LEVEL                 # Alt calc max zoom

    # Generate the animation with all frames
    frames = get_animation_frames(app)
    return render.Root(
        delay = FRAME_DURATION_MS,
        child = render.Box(render.Animation(frames)),
    )

def get_animation_frames(app):
    print("Determining point of interest")
    tx, ty = find_point_of_interest()   # Choose a point of interest    
    
    # Mandelbrot starts at center or over POI? Or somewhere in between
    x = (tx * INITIAL_POSITION - CTRX * (1 - INITIAL_POSITION)) 
    y = (ty * INITIAL_POSITION - CTRY * (1 - INITIAL_POSITION)) 

    frames = list()                     # List to store frames of the animation

    app['target'] = (tx, ty)
    app['gradient'] = get_random_gradient()
    app['zoom_level'] = rnd()*5+1  # 1.0 = Shows most of the mandelbrot set, -0.8 = all, 1+= zoomed in

    # Generate multiple frames for animation
    print("Generating frames")
    for frame in range(MAX_FRAMES):
        print("Generating frame #" + str(frame), " zoom:", app['zoom_level'])
        frame = draw_mandelbrot(app, x, y)
        frames.append(frame)
        app['zoom_level'] *= ZOOM_GROWTH
        # if TRANSLATE_PCT < 1.0 and TRANSLATE_PCT > 0 and INITIAL_POSITION != 1.0:
        #     x, y = (x * TRANSLATE_PCT + tx * (1-TRANSLATE_PCT)), (y * TRANSLATE_PCT + ty * (1-TRANSLATE_PCT))

    actual_max_iter = int(MIN_ITER + app['zoom_level'] * ZOOM_TO_ITER)
    print("Calculated max iterations:" + str(MAX_ITER_CALC) + " Actual:" + str(actual_max_iter))

    return frames

def float_range(start, end, num_steps, inclusive=False):
    step_size = (float(end) - float(start)) / num_steps
    result = []
    for i in range(num_steps):
        result.append(start + i * step_size)
    if inclusive:
        result.append(end)
    return result

def find_point_of_interest():
    x, y, best =  find_poi_near(CTRX, CTRY, 0.0, (MAXX-MINX), MAX_POI_SAMPLES, MAX_ITER_CALC)
    print("Settled on POI:", x, y, "escape:", best)
    return x, y

def find_poi_near(x, y, esc, depth, num_samples, iter_limit):
    bestx, besty, best_escape = x, y, esc

    for num in range(num_samples):
        x, y = bestx + (rnd() - 0.5) * depth, besty + (rnd() - 0.5) * depth
        iter, last_escape = mandelbrot_calc(x, y, iter_limit)
        if last_escape < ESCAPE_THRESHOLD and last_escape > best_escape:
            bestx, besty, best_escape = x, y, last_escape

#    print("Best POI so far:", bestx, besty, "escape:", best_escape)
    return bestx, besty, best_escape

# Map value v from one range to another
def map_range(v, min1, max1, min2, max2):
    # print("map_range v:", v, "first:", min1, max1, "second:", min2, max2)
    return min2 + (max2 - min2) * (v - min1) / (max1 - min1)

# Performs the mandelbrot calculation on a single point
# Returns both the escape distance and the number of iterations 
# (cannot exceed iter_limit)
def mandelbrot_calc(x, y, iter_limit):
    (iter, dist, _, _) = mandelbrot_calc_from(0, 0, x, y, iter_limit)
    return iter, dist

# Core mandelbrot calculation
# Supports starting from a zr/zi point other than 0
def mandelbrot_calc_from(zr, zi, x, y, iter_limit):
    cr, ci = x, y

    dist = 0
    for iter in range(1, iter_limit + 1):
        # Precompute squares to avoid repeating the same multiplication
        zr2 = zr * zr
        zi2 = zi * zi

        # Check if the point has escaped (this should happen after both zr and zi are updated)
        dist = zr2 + zi2
        if dist > ESCAPE_THRESHOLD:
            return (iter, dist, zr, zi)

        # Perform z = z^2 + c
        zi = 2 * zr * zi + ci
        zr = zr2 - zi2 + cr

    return (MAX_ITER_CALC, dist, zr, zi)

def int_to_hex(n):
    if n > 255:
        fail("Can't convert value " + str(n) + " to hex digit")
    hex_digits = "0123456789ABCDEF"
    return hex_digits[n // 16] + hex_digits[n % 16]

# Convert RGB values to a hexadecimal color code
def rgb_to_hex(r, g, b):
    return "#" + int_to_hex(r) + int_to_hex(g) + int_to_hex(b)

def get_gradient_color(app, iter):
    r,g,b = get_gradient_rgb(app,iter)
    return rgb_to_hex(r, g, b)

def get_gradient_rgb(app, iter):
    if iter >= MAX_ITER_CALC or iter < 0:
        return (0,0,0)
    
    # Convert iterations to a color
    t = (math.pow(math.log(iter), GRADIENT_SCALE_FACTOR) / NUM_GRADIENT_STEPS) % 1.0

    # Number of keyframes
    num_keyframes = len(app['gradient']) - 1
    #print("Num keyframes:", num_keyframes)
    
    # Ensure we are covering the whole gradient range
    frame_pos = t * num_keyframes
    #print("Frame pos:", frame_pos)
    lower_frame = int(frame_pos)  # Index of the lower keyframe
    upper_frame = min(lower_frame + 1, num_keyframes)  # Index of the upper keyframe
    
    # Fractional part for interpolation between the two keyframes
    local_t = frame_pos - float(lower_frame)
    
    # Get the colors of the two keyframes to blend between
    color_start = app['gradient'][lower_frame]
    color_end = app['gradient'][upper_frame]

    # if local_t < 0.5:
    #     return app['gradient'][lower_frame]
    # return app['gradient'][upper_frame]

    # Perform linear interpolation (LERP) between the two colors
    r = int(color_start[0] + local_t * (color_end[0] - color_start[0]))
    g = int(color_start[1] + local_t * (color_end[1] - color_start[1]))
    b = int(color_start[2] + local_t * (color_end[2] - color_start[2]))

    return (r, g, b)

# Blends RGB colors together
# Also converts from quantized values to full color spectrum
def blend_rgbs(*rgbs):    
    tr,tg,tb = 0, 0, 0
    count = 0
    for i in range(0, len(rgbs) - 1):
        r,g,b = rgbs[i]
        tr += r
        tg += g
        tb += b
        count += 1

    if count == 0:
        return rgb_to_hex(rgbs[0][0], rgbs[0][1], rgbs[0][2])

    return rgb_to_hex(int(tr / count * CHANNEL_MULT), int(tg / count * CHANNEL_MULT), int(tb / count * CHANNEL_MULT))

def random_color_tuple():
    return (random.number(0, MAX_COLORS), random.number(0, MAX_COLORS), random.number(0, MAX_COLORS))

def get_random_gradient():
    print ("Generating gradient")
    gradient = []
    color = random_color_tuple()
    for i in range(0, NUM_GRADIENT_STEPS):
        color = alter_color_rgb(color)
        gradient.append(color)
    return gradient

# At least one channel flipped, another randomized
def alter_color_rgb(color):
    flip_idx = random.number(0,2)
    rnd_idx = (flip_idx + random.number(1,2)) % 3
    keep_idx = 3 - flip_idx - rnd_idx
    new_color = [0,0,0]
    new_color[flip_idx] = MAX_COLORS - color[flip_idx]
    new_color[rnd_idx] = random.number(0, MAX_COLORS)
    new_color[keep_idx] = color[keep_idx]
    return new_color

# Renders a line
# Returns the number of iterations found if they are all the same
# If match_iter is passed something other than False, then it will
# compare all iterations against this value
# Returns the number of iterations, or False if they are not all the same
def generate_line_opt(app, match_iter, pix, set, max_iter):
    # print("generate_line_opt match_iter:", match_iter, "pix:", pix, "set:", set, "max_iter:", max_iter)
    
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
    zr, zi = 0.0, 0.0

    for val in range(start, end + 1):        
        # Update xm and ym based on whether it's vertical or horizontal
        if is_vertical:
            ym = map_range(val, pix['y1'], pix['y2'], set['y1'], set['y2'])
            yp = val  # Update yp in vertical case
        else:
            xm = map_range(val, pix['x1'], pix['x2'], set['x1'], set['x2'])
            xp = val  # Update xp in horizontal case

        # Get the pixel iteration count
        cache, zr, zi = generate_pixel(app, xp, yp, xm, ym, max_iter)

        # Initialize match_iter on first iteration
        if match_iter == -1:
            match_iter = cache

        elif match_iter != cache:
            return False, zr, zi
        
    # All iterations along the line were identical
    return match_iter, zr, zi

def alt(obj, field, value):
    # Create a shallow copy manually
    c = {}
    for k, v in obj.items():
        c[k] = v
    c[field] = value  # Set the new value for the specified field
    return c


def generate_mandelbrot_area(app, pix, set, iter_limit):
    dxp, dyp = int(pix['x2'] - pix['x1']), int(pix['y2'] - pix['y1'])
    dxm, dym = float(set['x2'] - set['x1']) / float(dxp), float(set['y2'] - set['y1']) / float(dyp)
    # print("generate_mandelbrot_area:", pix, set, iter_limit, "dp:", dxp, dyp, "dm:", dxm, dym)

    # No optimization render area:    
    # for y in range(pix['y1'], pix['y2'] + 1):
    #     for x in range(pix['x1'], pix['x2'] + 1):
    #         cache_pixel(app, x, y, dxm * float(x) + set['x1'], dym * float(y) + set['y1'], iter_limit)
    # return

    # A border with the same iterations can be filled with the same color
    match, zr, zi = generate_line_opt(app, False, alt(pix, 'y2', pix['y1']), alt(set, 'y2', set['y1']), iter_limit)
    if match != False:
        match, zr, zi = generate_line_opt(app, match, alt(pix, 'y1', pix['y2']), alt(set, 'y1', set['y2']), iter_limit)
    if match != False:
        match, zr, zi = generate_line_opt(app, match, alt(pix, 'x2', pix['x1']), alt(set, 'x2', set['x1']), iter_limit)
    if match != False:
        match, zr, zi = generate_line_opt(app, match, alt(pix, 'x1', pix['x2']), alt(set, 'x1', pix['x2']), iter_limit)

    if match != False:
        # print("Flooding filling region:", pix, " with iter:", match1)
        for y in range(pix['y1'], pix['y2'] + 1):
            for x in range(pix['x1'], pix['x2'] + 1):
                set_pixel(app, x, y, match, zr, zi)

    # Subdivide further
    else:
        if dxp > 2 and dxp >= dyp:
            # Horizontal split
            splitxp = int(dxp / 2)
            sxp_left = splitxp + pix['x1']
            sxp_right = sxp_left + 1
            sxm_left = set['x1'] + splitxp * dxm
            sxm_right = set['x1'] + (splitxp + 1) * dxm
            generate_mandelbrot_area(app, alt(pix, 'x2', sxp_left),  alt(set, 'x2', sxm_left),  iter_limit)
            generate_mandelbrot_area(app, alt(pix, 'x1', sxp_right), alt(set, 'x1', sxm_right), iter_limit)

        elif dyp > 2 and dyp >= dxp:
            # Vertical split
            splityp = int(dyp / 2)
            syp_above = splityp + pix['y1']
            syp_below = syp_above + 1
            sym_above = set['y1'] + splityp * dym
            sym_below = set['y1'] + (splityp + 1) * dym
            generate_mandelbrot_area(app, alt(pix, 'y2', syp_above), alt(set, 'y2', sym_above), iter_limit)
            generate_mandelbrot_area(app, alt(pix, 'y1', syp_below), alt(set, 'y1', sym_below), iter_limit)
        else:
            generate_pixel(app, pix['x1'], pix['y1'], set['x1'], set['y1'], iter_limit)
            generate_pixel(app, pix['x1'], pix['y2'], set['x1'], set['y2'], iter_limit)
            generate_pixel(app, pix['x2'], pix['y1'], set['x2'], set['y1'], iter_limit)
            generate_pixel(app, pix['x2'], pix['y2'], set['x2'], set['y2'], iter_limit)

# Calculates the number of iterations for a point on the map and returns it
# Tries to gather the pixel data from the cache if available
def generate_pixel(app, xp, yp, xm, ym, iter_limit):
    stored_val, zr, zi = get_pixel(app, xp, yp)
    if stored_val != -1:
        return stored_val, zr, zi

    # If this is the 2nd frame or later, reuse the old data to shorten iterations
    if 'last_map' in app:
        iter, _, zr, zi = mandelbrot_estimate(app, xp, yp, xm, ym, iter_limit)

    # Otherwise calculate the mandelbrot normally
    else:
        iter, _, zr, zi = mandelbrot_calc_from(0.0, 0.0, xm, ym, iter_limit)
        # print("m:", xm, ym, "p:", xp, yp, "iter:", iter)
        if iter == iter_limit:
            iter = MAX_ITER_CALC

    # Save pixel in map
    set_pixel(app, xp, yp, iter, zr, zi)

    return iter, zr, zi

# Looks at the last map's four related pixels to determine the start value for a mandelbrot calc
# Unfortunately this doesn't work well. :(
def mandelbrot_estimate(app, xp, yp, xm, ym, iter_limit):
    x_prev = (xp - app['center'][0]) / ZOOM_GROWTH + app['last_center'][0]
    y_prev = (yp - app['center'][1]) / ZOOM_GROWTH + app['last_center'][1]

    ul_x = int(x_prev + 0.5)  # Add 0.5 to better handle rounding
    ul_y = int(y_prev + 0.5)
    lr_x = ul_x + 1
    lr_y = ul_y + 1

    # return mandelbrot_calc_from(0.0, 0.0, xm, ym, iter_limit)

    # Edges may not have all four pixels
    if ul_x < 0 or lr_x > MAX_PIXEL_X or ul_y < 0 or lr_y > MAX_PIXEL_Y:
        return mandelbrot_calc_from(0.0, 0.0, xm, ym, iter_limit)
    
    # return MAX_ITER_CALC, 4, 4, 4    
    # print("Zoom:", app['zoom_level'], "m:", xm, ym, "p:", xp, yp, "base:", x_base, y_base, "frac:", x_frac, y_frac, "prev:", x_prev, y_prev)

    # Retrieve four neighboring pixels from the previous frame
    iter_tl, zr_tl, zi_tl = app['last_map'][ul_y][ul_x]       # Top-left
    iter_tr, zr_tr, zi_tr = app['last_map'][ul_y][lr_x]   # Top-right
    iter_bl, zr_bl, zi_bl = app['last_map'][lr_y][ul_x]   # Bottom-left
    iter_br, zr_br, zi_br = app['last_map'][lr_y][lr_x]  # Bottom-right

    # Bilinear interpolation for zr and zi
    x_frac = x_prev - float(ul_x)
    y_frac = y_prev - float(ul_y)

    # Interpolate zr and zi as before
    zr = (
        (1 - x_frac) * (1 - y_frac) * zr_tl +
        x_frac * (1 - y_frac) * zr_tr +
        (1 - x_frac) * y_frac * zr_bl +
        x_frac * y_frac * zr_br
    )
    zi = (
        (1 - x_frac) * (1 - y_frac) * zi_tl +
        x_frac * (1 - y_frac) * zi_tr +
        (1 - x_frac) * y_frac * zi_bl +
        x_frac * y_frac * zi_br
    )

    zr /= 256.0 # /= ZOOM_GROWTH
    zi /= 256.0
    # iter_limit = int(iter_limit / ZOOM_GROWTH)
    avg_iter = int((iter_tl + iter_tr + iter_bl + iter_br) / 4.0)
    min_iter = max(iter_limit - avg_iter, MIN_ITER)


    # print("Top-left (zr_tl, zi_tl):", zr_tl, zi_tl, "Iter:", iter_tl)
    # print("Top-right (zr_tr, zi_tr):", zr_tr, zi_tr, "Iter:", iter_tr)
    # print("Bottom-left (zr_bl, zi_bl):", zr_bl, zi_bl, "Iter:", iter_bl)
    # print("Bottom-right (zr_br, zi_br):", zr_br, zi_br, "Iter:", iter_br)
    # print("Interpolated (zr, zi):", zr, zi)

    (final_iter, final_dist, final_zr, final_zi) = mandelbrot_calc_from(zr, zi, xm, ym, int(iter_limit / 2))

    # print("Final iter:", final_iter, "dist:", final_dist, "zr:", final_zr, "zi:", final_zi)

    return (final_iter, final_dist, final_zr, final_zi)

    # return avg_limit, 0.0, zr, zi

# Set the number of iterations for a point on the map
def set_pixel(app, xp, yp, value, zr, zi):
    if xp < 0 or xp >= MAP_WIDTH or yp < 0 or yp >= MAP_HEIGHT:
        fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")
    app['map'][yp][xp] = (value, zr, zi)

# Returns the number of iterations for a point on the map
def get_pixel(app, xp, yp):
    if xp < 0 or xp >= MAP_WIDTH or yp < 0 or yp >= MAP_HEIGHT:
        fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")
    return app['map'][yp][xp]

# A map contains either the escape value for that point or -1 (uninitialized)
def create_empty_map(): 
    map = list()
    for y in range(MAP_HEIGHT):
        row = list()
        for x in range(MAP_WIDTH):
            row.append((int(-1), 0.0, 0.0))
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
    if 'map' in app:
        app['last_map'] = app["map"]
        app['last_center'] = app['center']
        app['last_zoom'] = app['zoom_level']
    app['map'] = create_empty_map()
    app['center'] = (x, y)
    # print("Current center point:", x, y, "Iter:", iterations)

    # Generate the map
    pix = { "x1": 0, "y1": 0, "x2": MAX_PIXEL_X, "y2": MAX_PIXEL_Y }
    set = { "x1": minx, "y1": miny, "x2": maxx, "y2": maxy }
    generate_mandelbrot_area(app, pix, set, iterations)

    # Render the map to the display
    return render_display(app)

# Converts a map to a Tidbyt Column made up of Rows made up of Boxes
def render_display(app):
    # Loop through each pixel in the display
    total_runs = 0
    rows = list()
    for y in range(DISPLAY_HEIGHT):
        osy = y*OVERSAMPLE_MULTIPLIER
        row = list()
        next_color = ""
        run_length = 0

        for x in range(DISPLAY_WIDTH):
            osx = x*OVERSAMPLE_MULTIPLIER
            
            if DISPLAY_WIDTH == MAP_WIDTH:
                iter, zr, zi = get_pixel(app, osx , osy)
                rgb = get_gradient_rgb(app, iter)
                color = rgb_to_hex(int(rgb[0] * CHANNEL_MULT), int(rgb[1] * CHANNEL_MULT), int(rgb[2] * CHANNEL_MULT))

            # Super sample this sheeit
            else:
                samples = []
                for offy in range(OVERSAMPLE_RANGE):
                    for offx in range(OVERSAMPLE_RANGE):  
                        iter, zr, zi = get_pixel(app, osx + offx , osy + offy)
                        samples.append(iter)

                rgbs = []
                for sample in samples:
                    rgbs.append(get_gradient_rgb(app, sample))

                color = blend_rgbs(*rgbs)
            
            # Add a 1x1 box with the appropriate color to the row        
            if next_color == "": # First color of row
                run_length = 1
                next_color = color
            elif color == next_color: # Color run detected
                run_length += 1
            else: # Color change
                addBox(row, run_length, next_color)
                total_runs += 1
                run_length = 1
                next_color = color

        # Add the row to the grid
        addBox(row, run_length, color) # Add last box for row
        total_runs += 1
        rows.append(render.Row(children = row))

    # pixel_count = DISPLAY_WIDTH * DISPLAY_HEIGHT
    # compression = int((total_runs / pixel_count) * 10000) / 100.0
    # print("Pixel count:", pixel_count, "Total runs:", total_runs, "Compression:", str(compression) + "%")

    return render.Column(
        children = rows,
    )

def addBox(row, run_length, color):
    if run_length == 1 and color == BLACK_PIXEL:
        row.append(BLACK_PIXEL)
    else:
        row.append(render.Box(width=run_length, height=1, color=color))

def rnd():
    return float(random.number(0, MAX_INT)) / float (MAX_INT)

