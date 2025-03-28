
import time
from pynput import mouse

# Lists to store event timestamps and hover durations
click_times = []
hover_durations = []

# Variables for hover tracking
current_hover_start = None
last_position = None
HOVER_MOVE_THRESHOLD = 5       # Movement in pixels considered as no significant move
HOVER_TIME_THRESHOLD = 0.5     # Minimum time (in seconds) to consider as a hover event

def on_click(x, y, button, pressed):
    """Callback function for mouse click events."""
    if pressed:
        click_time = time.time()
        click_times.append(click_time)
        print(f"Click detected at {(x, y)} at time {click_time:.2f}")

def on_move(x, y):
    """Callback function for mouse move events to track hover duration."""
    global current_hover_start, last_position, hover_durations

    current_time = time.time()

    # Initialize last position and hover start time if this is the first move event.
    if last_position is None:
        last_position = (x, y)
        current_hover_start = current_time
        return

    # Calculate Euclidean distance from the last recorded position.
    dx = x - last_position[0]
    dy = y - last_position[1]
    distance = (dx*2 + dy*2) ** 0.5

    # If movement is significant, conclude any ongoing hover event.
    if distance > HOVER_MOVE_THRESHOLD:
        if current_hover_start is not None:
            hover_duration = current_time - current_hover_start
            if hover_duration >= HOVER_TIME_THRESHOLD:
                hover_durations.append(hover_duration)
                print(f"Hover ended: Duration {hover_duration:.2f} seconds")
        # Reset hover tracking with the new position and time
        current_hover_start = current_time
        last_position = (x, y)
    # If movement is minimal, continue tracking the current hover event.

def on_scroll(x, y, dx, dy):
    """Scroll events are not considered for click/hover analysis in this implementation."""
    pass

# Set up the mouse listener for click, move, and scroll events.
listener = mouse.Listener(
    on_click=on_click,
    on_move=on_move,
    on_scroll=on_scroll
)

print("Starting mouse event listener. Move the mouse and click for 10 seconds...")
listener.start()

# Let the listener run for a specified duration (e.g., 10 seconds)
time.sleep(10)
listener.stop()

# Define the session duration (in seconds). Here it's set to 10 seconds.
session_duration = 10.0

# Calculate Click Rate: total clicks divided by session duration.
click_rate = len(click_times) / session_duration

# Calculate average hover duration if there are any hover events recorded.
average_hover_duration = (sum(hover_durations) / len(hover_durations)) if hover_durations else 0

print("\n--- Summary ---")
print(f"Total Clicks: {len(click_times)}")
print(f"Click Rate: {click_rate:.2f} clicks per second")
print(f"Total Hover Events: {len(hover_durations)}")
print(f"Average Hover Duration: {average_hover_duration:.2f} seconds")
