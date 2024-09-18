load("render.star", "render")
load("random.star", "random")
load("time.star", "time")

ZOOM_GROWTH = 1.05
MAX_INTEREST_POINTS = 40
FRAME_DURATION_MS = 100
MAX_FRAMES = int(15000 / FRAME_DURATION_MS)
MIN_ITER = 20
ZOOM_TO_ITER = 0.5

def main(config):
    random.seed(int(time.now().unix / 15))
    frames = get_animation_frames()

    # Return the animation with all frames
    return render.Root(        
        delay = FRAME_DURATION_MS,
        child = render.Box(render.Animation(frames)),
    )

def get_animation_frames():
    # Choose a point of interest
    (zoom_center_x, zoom_center_y) = find_point_of_interest()

    # Initialize zoom level
    zoom_level = 1.0

    # List to store frames of the animation
    frames = list()
    
    # Generate multiple frames for animation
    for frame_num in range(MAX_FRAMES):
        frame = draw_mandelbrot_at(zoom_center_x, zoom_center_y, zoom_level)
        zoom_level *= ZOOM_GROWTH
        frames.append(frame)
    
    return frames

def find_point_of_interest():
    (x,y,zoom) = (0, 0, 1)    
    for _ in range(MAX_FRAMES):
        (x,y) = find_interesting_point_near(x,y,zoom)
        zoom *= ZOOM_GROWTH
    return (x,y)

def find_interesting_point_near(x, y, zoom_level):
    step = 1 / zoom_level
    (bestx, besty, bestesc) = (x, y, escape_proximity(x,y))

    for _ in range(MAX_INTEREST_POINTS):  # Check random points
        # Pick random point in range of current frame display
        dx = (random.number(0, 128) - 64) / 32 * step  # Larger range for x
        dy = (random.number(0, 64) - 32) / 32 * step   # Half range for y
        a = x + dx
        b = y + dy

        # Check if the point has higher non-escaping distance
        escape_distance = escape_proximity(a, b)
        if escape_distance < 4 and escape_distance > bestesc:
            (bestx, besty, bestesc) = (a, b, escape_distance)

    return (bestx, besty)

def escape_proximity(a, b):
    zr = 0
    zi = 0
    cr = a
    ci = b
    max_iter = 50
    escape_distance = 0  # Track how far it goes in the iterations

    for i in range(max_iter):
        zr_next = zr * zr - zi * zi + cr
        zi_next = 2 * zr * zi + ci
        zr = zr_next
        zi = zi_next
        escape_distance = zr * zr + zi * zi

        if escape_distance > 4:  # The point has escaped
            return escape_distance

    # Instead of returning 0, return the distance even if it didn't escape
    return escape_distance

# Interpolation function between two points
def interpolate(start, end, t):
    return start + t * (end - start)


# Function to draw Mandelbrot at a specific center and zoom level
def draw_mandelbrot_at(center_x, center_y, zoom_level):
    rows = list()
    iterations = int(MIN_ITER + zoom_level * ZOOM_TO_ITER)

    # Loop through each pixel in the display (32 rows by 64 columns)
    for y in range(32):
        row = list()
        for x in range(64):
            # Calculate the bounds for zooming
            zoom_xmin = center_x - (1.5 / zoom_level)
            zoom_xmax = center_x + (1.5 / zoom_level)
            zoom_ymin = center_y - (1.0 / zoom_level)
            zoom_ymax = center_y + (1.0 / zoom_level)
            
            # Map the pixel to a zoomed-in complex number
            a = map_range(x, 0, 64, zoom_xmin, zoom_xmax)
            b = map_range(y, 0, 32, zoom_ymin, zoom_ymax)
            
            # Compute the color at this point
            color = mandelbrot(a, b, iterations)

            # Add a 1x1 box with the appropriate color to the row
            row.append(
                render.Box(
                    width=1,
                    height=1,
                    color=color
                )
            )
        
        # Add the row to the grid
        rows.append(
            render.Row(children=row)
        )

    return render.Column(
        children=rows
    )

# Map value v from one range to another
def map_range(v, min1, max1, min2, max2):
    return min2 + (max2 - min2) * (v - min1) / (max1 - min1)

# Mandelbrot function that returns a color based on escape time
def mandelbrot(a, b, iterations):
    # Initialize z = 0 + 0i
    zr = 0
    zi = 0
    cr = a
    ci = b

    for i in range(iterations):
        # Perform z = z^2 + c
        zr_next = zr * zr - zi * zi + cr
        zi_next = 2 * zr * zi + ci
        zr = zr_next
        zi = zi_next

        # Check if the squared magnitude exceeds 2
        if zr * zr + zi * zi > 4:
            return get_color(i, iterations)  # Escaped, return color

    return '#000000'  # Did not escape (black for Mandelbrot set)

def get_color(iteration, max_iter):
    if iteration == max_iter:
        return '#000000'  # Black for points inside the set
    
    # Normalize the iteration count to the range [0, 1]
    t = iteration / max_iter

    # Use a hue-based color scheme for more variety
    hue = int(360 * t)  # Map t to a hue in degrees (0 to 360)
    saturation = 1.0    # Full saturation
    lightness = 0.5 #0.5 + 0.5 * t  # Lightness increases with t for brightness

    return hsl_to_hex(hue, saturation, lightness)

# Interpolate from current color to the target gradient
def interpolate_color_to_gradient(current_color, target_gradient, t, ratio):
    # Get the two colors from the gradient based on iteration
    num_colors = len(target_gradient) - 1
    scaled_t = t * num_colors
    color_index = int(scaled_t)
    color_t = scaled_t - color_index

    # Get the current and next target colors from the gradient
    target_color1 = hex_to_rgb(target_gradient[color_index])
    target_color2 = hex_to_rgb(target_gradient[color_index + 1])

    # Interpolate between the two target colors in the gradient
    target_color = interpolate_colors(target_color1, target_color2, color_t)

    # Interpolate 20% between the current color and the target color
    current_rgb = hex_to_rgb(current_color)
    final_color = interpolate_colors(current_rgb, target_color, ratio)

    return rgb_to_hex(final_color[0], final_color[1], final_color[2])

# Function to interpolate between two colors
def interpolate_colors(color1, color2, t):
    r = int((1 - t) * color1[0] + t * color2[0])
    g = int((1 - t) * color1[1] + t * color2[1])
    b = int((1 - t) * color1[2] + t * color2[2])
    return (r, g, b)

def pow(base, exp):
    if exp == 0:
        return 1
    elif exp < 0:
        return 1 / pow(base, -exp)  # Handle negative exponents
    
    result = 1.0
    integer_part = int(exp)
    
    # Compute the integer part of the exponentiation
    for _ in range(integer_part):
        result *= base

    # If there is a fractional part, approximate it using a series expansion
    fractional_part = exp - integer_part
    if fractional_part > 0:
        # Approximate the fractional exponent (using a simple method)
        # Use a basic approximation for sqrt (like 0.5 exponent):
        approx_fraction = 1 + fractional_part * (base - 1)
        result *= approx_fraction

    return result

# Helper function to convert an integer to two-digit hexadecimal value
def int_to_hex(n):
    hex_digits = "0123456789ABCDEF"
    return hex_digits[n // 16] + hex_digits[n % 16]

# Convert RGB values to a hexadecimal color code
def rgb_to_hex(r, g, b):
    return '#' + int_to_hex(r) + int_to_hex(g) + int_to_hex(b)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

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