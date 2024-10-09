###
# TIMERS
###

timers = {}

def timer_start(category):
    # Ensure category exists
    if category not in timers:
        timers[category] = {
            "elapsed": 0,  # Store total elapsed time
            "timers": [],  # Store individual timer start times
        }

    # Get the current time in nanoseconds and store it as a unique timer
    start_time_ns = time.now().unix_nano
    timer_id = len(timers[category]["timers"])
    timers[category]["timers"].append(start_time_ns)

    # Return the timer ID for reference in end_time
    return timer_id

def timer_end(category, timer_id):
    if category not in timers:
        fail("Must call start before end for category '{}'".format(category))

    # Ensure the timer_id is valid and exists
    t = timers[category]["timers"]
    if timer_id >= len(t) or t[timer_id] == None:
        fail("Invalid timer_id '{}' for category '{}'".format(timer_id, category))

    # Get the current time as end time and calculate elapsed nanoseconds
    end_time_ns = time.now().unix_nano
    elapsed = end_time_ns - timers[timer_id]

    # Add the elapsed time to the total and mark the timer as stopped
    timers[category]["elapsed"] += elapsed
    timers[category]["timers"][timer_id] = None  # Mark timer as ended

def timer_display():
    result = "PROFILE\n"

    for category in timers:
        elapsed_time_ns = timers[category]["elapsed"]  # Total elapsed time in nanoseconds

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
