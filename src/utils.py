# utils.py
def calculate_averages(total_frames, avg_horizontal_lines, avg_vertical_lines):
    avg_horizontal_lines /= total_frames
    avg_vertical_lines /= total_frames
    return avg_horizontal_lines, avg_vertical_lines
