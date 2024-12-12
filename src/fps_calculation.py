# fps_calculation.py
import time

# Global variables to track FPS
prev_time = time.time()
frame_count = 0
fps = 30

# Function to calculate FPS with optimization
def fps_boost(optimized=True):
    global prev_time, fps, frame_count

    if optimized:
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - prev_time)
            prev_time = current_time
    else:
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

    return fps
