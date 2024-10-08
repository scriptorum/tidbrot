#
# TODO
# - Derive certain variables from a budgeting system
#    + Measure the time used to find a POI
#    + If not a lot of time, display a single frame with high AA
#    + Otherwise do an animation
# 

load("random.star", "random")
load("render.star", "render")
load("time.star", "time")
load("math.star", "math")
load("schema.star", "schema")
load("humanize.star", "humanize")

DEF_ZOOM_GROWTH = "1.04"        # 1 = no zoom in, 1.1 = 10% zoom per frame

MIN_ITER = 30                   # minimum iterations, raise if initial zoom is > 1
ESCAPE_THRESHOLD = 4.0          # 4.0 standard, less for faster calc, less accuracy
ZOOM_TO_ITER = 0.75             # 1.0 standard, less for faster calc, less accuracy
DISPLAY_WIDTH = 64              # Tidbyt is 64 pixels wide
DISPLAY_HEIGHT = 32             # Tidbyt is 32 pixels high
NUM_GRADIENT_STEPS = 32         # Higher = more color variation
GRADIENT_SCALE_FACTOR = 1.55    # 1.55 = standard, less for more colors zoomed in, more for few colors zoomed in
MAX_POI_SAMPLES = 100000        # Number of random points to check for POI-worthiness
CTRX, CTRY = -0.75, 0           # mandelbrot center
MINX, MINY, MAXX, MAXY = -2.5, -0.875, 1.0, 0.8753  # Bounds to use for mandelbrot set
MAX_COLOR_CHANNEL = 8                                      # Max quantized channel values (helps reduce Image Too Large errors)
CHANNEL_MULT = 255.9999 / MAX_COLOR_CHANNEL                # Conversion from quantized value to full range color channel (0-255)
BLACK_COLOR = "#000000"                             # Shorthand for black color
MAX_INT = int(math.pow(2, 53))                      # Guesstimate for Starlark max_int
BLACK_PIXEL = render.Box(width=1, height=1, color=BLACK_COLOR)                  # Pregenerated 1x1 pixel black box

def main(config):
    random.seed(4)
    app = {"config": config}
    
    # milliseconds per frame; for FPS, use value = 1000/fps
    frame_duration_ms = config.str("frames", "500")
    if not frame_duration_ms.isdigit():
        return err("Must be an integer: {}".format(frame_duration_ms))
    app['frame_duration_ms'] = int(frame_duration_ms)

    app['max_frames'] = int(15000 / app['frame_duration_ms'])
    print("Frame MS:", app['frame_duration_ms'])

    # Zoom growth 1.01 = 1% zoom in per frame
    app['zoom_growth'] = float(config.str('zoom', DEF_ZOOM_GROWTH))
    app['max_zoom'] = math.pow(app['zoom_growth'], app['max_frames'])
    print("Zoom:", app['zoom_growth'])

    # RANGE      -->  1 = 1x1 pixel no blending, 2 = 2x2 pixel blend
    # MULTIPLIER -->  1 = no or mini AA, 2 = 2AA (2x2=4X samples)
    # OFFSET     -->  0 = 1:1, 1 = oversample by 1 pixel (use with RANGE 2, MULT 1 for mini AA)
    oversampling = config.str("oversampling", "none")
    if oversampling == 'none':
        app['oversample_range'] = 1
        app['oversample_multiplier'] = 1
        app['oversample_offset'] = 0
    elif oversampling == 'mini':
        app['oversample_range'] = 2
        app['oversample_multiplier'] = 1
        app['oversample_offset'] = 1
    elif oversampling == '2x':
        app['oversample_range'] = 2
        app['oversample_multiplier'] = 2
        app['oversample_offset'] = 0
    elif oversampling == '4x':
        app['oversample_range'] = 4
        app['oversample_multiplier'] = 4
        app['oversample_offset'] = 0
    else: 
        return err("Unknown oversampling value: %s" % oversampling)        
    print("Oversampling RANGE:", app['oversample_range'], "MULT:", app['oversample_multiplier'], "OFFSET:", app['oversample_offset'])

    app['map_width'] = DISPLAY_WIDTH * app["oversample_multiplier"] + app["oversample_offset"]  # Pixels samples per row
    app['map_height'] = DISPLAY_HEIGHT * app["oversample_multiplier"] + app["oversample_offset"]  # Pixel samples per column
    app['max_pixel_x'] = app['map_width'] - 1                  # Maximum sample for x
    app['max_pixel_y'] = app['map_height'] - 1                 # Maximum sample for y

    # print("Display:", DISPLAY_WIDTH, DISPLAY_HEIGHT, "MapSize:", app['map_width'], app['map_height'], "MaxPixel:", app['max_pixel_x'], app['max_pixel_y'])

    # Generate the animation with all frames
    frames = get_animation_frames(app)

    display_times(app)

    return render.Root(
        delay = app['frame_duration_ms'],
        child = render.Box(render.Animation(frames)),
    )

def get_animation_frames(app):
    app['zoom_level'] = rnd(app)*5+1       # 1.0 = Shows most of the mandelbrot set, -0.8 = all, 1+= zoomed in
    app['max_iter'] = int(math.round(MIN_ITER + app['zoom_level'] * ZOOM_TO_ITER * math.pow(app['zoom_growth'], app['max_frames'])) + 1)    # Calc max iter -- it's wrong, doesn't include initial zoom

    id = start_time(app, "get_random_gradient")
    app['gradient'] = get_random_gradient(app)
    end_time(app, "get_random_gradient", id)

    id = start_time(app, "poi")
    x, y = find_point_of_interest(app)  # Choose a point of interest    
    app['target'] = (x, y)
    end_time(app, "poi", id)

    # Generate multiple frames for animation
    print("Generating frames")
    frames = list()                     # List to store frames of the animation
    for frame in range(app['max_frames']):
        print("Generating frame #" + str(frame), " zoom:", app['zoom_level'])
        id = start_time(app, "render_mandelbrot")
        frame = render_mandelbrot(app, x, y)
        end_time(app, "render_mandelbrot", id)
        frames.append(frame)
        app['zoom_level'] *= app['zoom_growth']

    actual_max_iter = int(MIN_ITER + app['zoom_level']/app['zoom_growth'] * ZOOM_TO_ITER)
    print("Calculated max iterations:" + str(app['max_iter']) + " Actual:" + str(actual_max_iter))

    return frames

def float_range(start, end, num_steps, inclusive=False):
    step_size = (float(end) - float(start)) / num_steps
    result = []
    for i in range(num_steps):
        result.append(start + i * step_size)
    if inclusive:
        result.append(end)
    return result

def find_point_of_interest(app):
    print("Determining point of interest")    
    x, y, best =  find_poi_near(app, CTRX, CTRY, 0.0, (MAXX-MINX), MAX_POI_SAMPLES, app['max_iter'])
    print("Settled on POI:", x, y, "escape:", best)
    return x, y

def find_poi_near(app, x, y, esc, depth, num_samples, iter_limit):
    bestx, besty, best_escape = x, y, esc

    for num in range(num_samples):
        x, y = bestx + (rnd(app) - 0.5) * depth, besty + (rnd(app) - 0.5) * depth
        iter, last_escape = mandelbrot_calc(app, x, y, iter_limit)
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
def mandelbrot_calc(app, x, y, iter_limit):
    id = start_time(app, "calc")
    
    # Initialize z (zr, zi) and c (cr, ci)
    x2, y2, w = 0.0, 0.0, 0.0

    # Use a for loop to simulate the behavior of a while loop
    for iteration in range(1, iter_limit + 1):
        # Check if the point has escaped
        if x2 + y2 > ESCAPE_THRESHOLD:
            end_time(app, "calc", id)
            return (iteration, x2 + y2)

        # Calculate new zr and zi (x and y in pseudocode)
        zr = x2 - y2 + x
        zi = w - x2 - y2 + y

        # Update squares and w for the next iteration
        x2 = zr * zr
        y2 = zi * zi
        w = (zr + zi) * (zr + zi)

    # End timing and return the max iteration and the final distance if escape condition is not met
    end_time(app, "calc", id)
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
    id = start_time(app, "get_gradient_color")

    r,g,b = get_gradient_rgb(app,iter)
    color = rgb_to_hex(r, g, b)

    end_time(app, "get_gradient_color", id)
    return color


def dump(app):
    for y in range(app['map_height']):
        row = ""
        for x in range(app['map_width']):
            if app['map'][y][x] < 0:
                row += " "
            else:
                row += "X"
        print(row)

def get_gradient_rgb(app, iter):
    if iter == app['max_iter']:
        return (0,0,0)
    
    elif iter > app['max_iter'] or iter < 0:
        dump(app)
        fail("Bad iterations in get_gradient_rgb:", iter)
                
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
    elif count == 2: # Lame optimization
        return rgb_to_hex(int(tr << 2 * CHANNEL_MULT), int(tg << 2 * CHANNEL_MULT), int(tb << 2 * CHANNEL_MULT))
    elif count == 4: # Lame optimization
        return rgb_to_hex(int(tr << 4 * CHANNEL_MULT), int(tg << 4 * CHANNEL_MULT), int(tb << 4 * CHANNEL_MULT))

    return rgb_to_hex(int(tr / count * CHANNEL_MULT), int(tg / count * CHANNEL_MULT), int(tb / count * CHANNEL_MULT))

def random_color_tuple():
    return (random.number(0, MAX_COLOR_CHANNEL), random.number(0, MAX_COLOR_CHANNEL), random.number(0, MAX_COLOR_CHANNEL))

def get_random_gradient(app):
    print ("Generating gradient")
    gradient = []
    color = random_color_tuple()
    for i in range(0, NUM_GRADIENT_STEPS):
        id = start_time(app, "alter_color_rgb")
        color = alter_color_rgb(color)
        end_time(app, "alter_color_rgb", id)
        gradient.append(color)
    return gradient

# At least one channel flipped, another randomized
def alter_color_rgb(color):
    flip_idx = random.number(0,2)
    rnd_idx = (flip_idx + random.number(1,2)) % 3
    keep_idx = 3 - flip_idx - rnd_idx
    new_color = [0,0,0]
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
    id = start_time(app, "generate_line_opt")

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

    for val in range(start, end + 1):        
        # Update xm and ym based on whether it's vertical or horizontal
        if is_vertical:
            ym = map_range(val, pix['y1'], pix['y2'], set['y1'], set['y2'])
            yp = val  # Update yp in vertical case
        else:
            xm = map_range(val, pix['x1'], pix['x2'], set['x1'], set['x2'])
            xp = val  # Update xp in horizontal case

        # Get the pixel iteration count
        cache = generate_pixel(app, xp, yp, xm, ym, max_iter)

        # Initialize match_iter on first iteration
        if match_iter == -1:
            match_iter = cache

        # Bail early: Not all iterations along the line were identical
        elif match_iter != cache:
            end_time(app, "generate_line_opt", id)
            return False
        
    # All iterations along the line were identical
    end_time(app, "generate_line_opt", id)
    return match_iter

# Copies an object and alters one field to a new value
def alt(obj, field, value):
    c = {}
    for k, v in obj.items():
        c[k] = v
    c[field] = value 
    return c

def flood_fill(app, area, iter):
    id = start_time(app, "flood_fill")

    for y in range(area['y1'], area['y2'] + 1):
        for x in range(area['x1'], area['x2'] + 1):
            set_pixel(app, x, y, iter)
    end_time(app, "flood_fill", id)





def generate_mandelbrot_area(app, pix, set, iter_limit):
    id = start_time(app, "generate_mandelbrot_area")

    # Initialize the stack with the first region to process
    stack = [(pix, set)]

    # We will dynamically increase the stack in the loop
    for i in range(MAX_INT): # Why no for loop, damn you
        if len(stack) == 0:
            break

        # Pop the last item from the stack
        current_pix, current_set = stack.pop()

        dmdp_id = start_time(app, "generate_mandelbrot_area_dmdp")
        dxp, dyp = int(current_pix['x2'] - current_pix['x1']), int(current_pix['y2'] - current_pix['y1'])
        dxm, dym = float(current_set['x2'] - current_set['x1']) / float(dxp), float(current_set['y2'] - current_set['y1']) / float(dyp)
        end_time(app, "generate_mandelbrot_area_dmdp", dmdp_id)

        # A border with the same iterations can be filled with the same color
        iter = generate_line_opt(app, -1, alt(current_pix, 'y2', current_pix['y1']), alt(current_set, 'y2', current_set['y1']), iter_limit)
        if iter != False:
            iter = generate_line_opt(app, iter, alt(current_pix, 'y1', current_pix['y2']), alt(current_set, 'y1', current_set['y2']), iter_limit)
        if iter != False:
            iter = generate_line_opt(app, iter, alt(current_pix, 'x2', current_pix['x1']), alt(current_set, 'x2', current_set['x1']), iter_limit)
        if iter != False:
            iter = generate_line_opt(app, iter, alt(current_pix, 'x1', current_pix['x2']), alt(current_set, 'x1', current_set['x2']), iter_limit)
        if iter != False:
            flood_fill(app, current_pix, iter)

        else:
            # Perform vertical split
            if dyp >= 3 and dyp >= dxp:
                split_id = start_time(app, "generate_mandelbrot_area_split")

                splityp = int(dyp / 2)
                syp_above = splityp + current_pix['y1']
                syp_below = syp_above + 1
                sym_above = current_set['y1'] + splityp * dym
                sym_below = current_set['y1'] + (splityp + 1) * dym

                end_time(app, "generate_mandelbrot_area_split", split_id)

                # Add sub-regions to the stack
                stack.append((alt(current_pix, 'y1', syp_below), alt(current_set, 'y1', sym_below)))
                stack.append((alt(current_pix, 'y2', syp_above), alt(current_set, 'y2', sym_above)))

            # Perform horizontal split
            elif dxp >= 3 and dyp >= 3:
                split_id = start_time(app, "generate_mandelbrot_area_split")
                splitxp = int(dxp / 2)
                sxp_left = splitxp + current_pix['x1']
                sxp_right = sxp_left + 1
                sxm_left = current_set['x1'] + splitxp * dxm
                sxm_right = current_set['x1'] + (splitxp + 1) * dxm
                end_time(app, "generate_mandelbrot_area_split", split_id)

                # Add sub-regions to the stack
                stack.append((alt(current_pix, 'x1', sxp_right), alt(current_set, 'x1', sxm_right)))
                stack.append((alt(current_pix, 'x2', sxp_left), alt(current_set, 'x2', sxm_left)))

            # This is a small area with differing iterations, calculate/mark them individually
            else:
                final_generate_id = start_time(app, "generate_mandelbrot_area final_generate_pixel")
                for offy in range(0, dyp + 1):
                    for offx in range(0, dxp + 1):
                        generate_pixel(app, current_pix['x1'] + offx, current_pix['y1'] + offy, current_set['x1'] + (dxm * offx), current_set['y1'] + (dym * offy), iter_limit)
                end_time(app, "generate_mandelbrot_area final_generate_pixel", final_generate_id)

    end_time(app, "generate_mandelbrot_area", id)






# Calculates the number of iterations for a point on the map and returns it
# Tries to gather the pixel data from the cache if available
def generate_pixel(app, xp, yp, xm, ym, iter_limit):
    id = start_time(app, "generate_pixel")

    stored_val = get_pixel(app, xp, yp)
    if stored_val != -1:
        end_time(app, "generate_pixel", id)
        return stored_val
    
    # Normal mandelbrot calculation
    iter, _ = mandelbrot_calc(app, xm, ym, iter_limit)
        
    # print("m:", xm, ym, "p:", xp, yp, "iter:", iter)
    if iter == iter_limit:
        iter = app['max_iter']

    # Save iterations for pixel in map
    set_pixel(app, xp, yp, iter)

    end_time(app, "generate_pixel", id)
    return iter

# Set the number of iterations for a point on the map
def set_pixel(app, xp, yp, value):
    id = start_time(app, "set_pixel")
    if xp < 0 or xp >= app['map_width'] or yp < 0 or yp >= app['map_height']:
        fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")
    app['map'][yp][xp] = value
    end_time(app, "set_pixel", id)

# Returns the number of iterations for a point on the map
def get_pixel(app, xp, yp):
    id = start_time(app, "get_pixel")
    if xp < 0 or xp >= app['map_width'] or yp < 0 or yp >= app['map_height']:
        fail("Bad get_pixel(" + str(xp) + "," + str(yp) + ") call")
    val = app['map'][yp][xp]
    end_time(app, "get_pixel", id)
    return val

# A map contains either the escape value for that point or -1 (uninitialized)
def create_empty_map(app): 
    id = start_time(app, "create_empty_map")
    map = list()
    for y in range(app['map_height']):
        row = list()
        for x in range(app['map_width']):
            row.append(int(-1))
        map.append(row)
    end_time(app, "create_empty_map", id)
    return map

def render_mandelbrot(app, x, y):
    iterations = int(MIN_ITER + app['zoom_level'] * ZOOM_TO_ITER)
    
    # Determine coordinates
    half_width = (MAXX - MINX) / app['zoom_level'] / 2.0
    half_height = (MAXY - MINY) / app['zoom_level'] / 2.0
    minx, miny  = x - half_width, y - half_height
    maxx, maxy  = x + half_width, y + half_height

    # Create the map
    app['map'] = create_empty_map(app)
    app['center'] = (x, y)
    # print("Current center point:", x, y, "Iter:", iterations)

    # Generate the map
    pix = { "x1": 0, "y1": 0, "x2": app['max_pixel_x'], "y2": app['max_pixel_y'] }
    set = { "x1": minx, "y1": miny, "x2": maxx, "y2": maxy }
    app['region'] = set
    generate_mandelbrot_area(app, pix, set, iterations)

    # Render the map to the display
    return render_display(app)

# Converts a map to a Tidbyt Column made up of Rows made up of Boxes
def render_display(app):
    rd_id = start_time(app, "render_display")
     # Loop through each pixel in the display
    total_runs = 0
    rows = list()
    osx, osy = 0, 0
    for y in range(DISPLAY_HEIGHT):
        row = list()
        next_color = ""
        run_length = 0
        osx = 0

        for x in range(DISPLAY_WIDTH):            
            # Get color from single pixel
            if DISPLAY_WIDTH == app['map_width']:
                iter = get_pixel(app, osx, osy)
                rgb = get_gradient_rgb(app, iter)
                color = rgb_to_hex(int(rgb[0] * CHANNEL_MULT), int(rgb[1] * CHANNEL_MULT), int(rgb[2] * CHANNEL_MULT))

            # Get color by oversampling
            else:
                samples = []
                for offy in range(app["oversample_range"]):
                    for offx in range(app["oversample_range"]):  
                        iter = get_pixel(app, osx + offx , osy + offy)
                        samples.append(iter)

                rgbs = []
                for sample in samples:
                    rgbs.append(get_gradient_rgb(app, sample))

                id = start_time(app, "blend_rgbs")
                color = blend_rgbs(*rgbs)
                end_time(app, "blend_rgbs", id)
           
            # Add a 1x1 box with the appropriate color to the row        
            if next_color == "": # First color of row
                run_length = 1
                next_color = color
            elif color == next_color: # Color run detected
                run_length += 1
            else: # Color change
                add_box(row, run_length, next_color)
                total_runs += 1
                run_length = 1
                next_color = color

            osx += app["oversample_multiplier"]
        osy += app["oversample_multiplier"]

        # Add the row to the grid
        id = start_time(app, "add_box")
        add_box(row, run_length, color) # Add last box for row
        end_time(app, "add_box", id)
        total_runs += 1
        rows.append(render.Row(children = row))

    # pixel_count = DISPLAY_WIDTH * DISPLAY_HEIGHT
    # compression = int((total_runs / pixel_count) * 10000) / 100.0
    # print("Pixel count:", pixel_count, "Total runs:", total_runs, "Compression:", str(compression) + "%")
    end_time(app, "render_display", rd_id)

    return render.Column(
        children = rows,
    )

def add_box(row, run_length, color):
    if run_length == 1 and color == BLACK_PIXEL:
        row.append(BLACK_PIXEL)
    else:
        row.append(render.Box(width=run_length, height=1, color=color))

def rnd(app):
    id = start_time(app, "rnd")
    val = float(random.number(0, MAX_INT)) / float (MAX_INT)
    end_time(app, "rnd", id)
    return val

def get_schema():
    return schema.Schema(
        version = "1",
        fields = [
            schema.Dropdown(
                id="oversampling",
                name="Oversampling",
                desc="Oversampling Method",
                icon="border-none",
                default="mini",
                options=[
                    schema.Option(value="none", display="None"),
                    schema.Option(value="mini", display="Mini AA"),
                    schema.Option(value="2x", display="2X AA"),
                    schema.Option(value="4x", display="4X AA"),
                ]        
            ),
            schema.Dropdown(
                id="zoom",
                name="Zoom",
                desc="Amount to zoom in per frame",
                icon="magnifying-glass-plus",
                default=DEF_ZOOM_GROWTH,
                options=[
                    schema.Option(value="1.01", display="1%"),
                    schema.Option(value="1.02", display="2%"),
                    schema.Option(value="1.03", display="3%"),
                    schema.Option(value="1.04", display="4%"),
                    schema.Option(value="1.05", display="5%"),
                    schema.Option(value="1.06", display="6%"),
                    schema.Option(value="1.07", display="7%"),
                    schema.Option(value="1.08", display="8%"),
                ]        
            ),
            schema.Text(
                id="frames",
                name="Frame Rate",
                desc="Milliseconds per frame",
                icon="clock",
                default="150",
            )
        ]
    )

def err(msg):
    return render.Root(
        render.Marquee(
            width=64,
            child=render.Text(msg),
            offset_start=5,
            offset_end=32,
        )
    )


def start_time(app, category):
    if "profiling" not in app:
        app["profiling"] = {}
    
    # Ensure category exists
    if category not in app["profiling"]:
        app["profiling"][category] = {
            "elapsed": 0,  # Store total elapsed time
            "timers": []   # Store individual timer start times
        }
    
    # Get the current time in nanoseconds and store it as a unique timer
    start_time_ns = time.now().unix_nano
    timer_id = len(app["profiling"][category]["timers"])
    app["profiling"][category]["timers"].append(start_time_ns)

    # Return the timer ID for reference in end_time
    return timer_id

def end_time(app, category, timer_id):
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

def display_times(app):
    # Display all categories' cumulative profiling results
    if "profiling" not in app:
        return "No profiling data available."
    
    result = "PROFILE\n"

    for category in app["profiling"]:
        elapsed_time_ns = app["profiling"][category]["elapsed"]  # Total elapsed time in nanoseconds

        print("Category: '{}', Raw elapsed ns: {}".format(category, elapsed_time_ns))

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
