# video_processing.py
import cv2
from src.fps_calculation import fps_boost
from src.overlay import add_custom_text, add_icon
from src.line_detection import detect_lines

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    return cap

def process_frame(frame, total_frames, avg_horizontal_lines, avg_vertical_lines, max_horizontal_lines, max_vertical_lines, font_path, font_size, icon, use_optimized_fps):
    frame, horizontal_lines, vertical_lines = detect_lines(frame)

    # Update stats
    avg_horizontal_lines += horizontal_lines
    avg_vertical_lines += vertical_lines
    max_horizontal_lines = max(max_horizontal_lines, horizontal_lines)
    max_vertical_lines = max(max_vertical_lines, vertical_lines)

    # FPS calculation
    fps = fps_boost(use_optimized_fps)

    # Add FPS and stats overlay after lines
    text = f'FPS: {fps:.2f} | Horizontal: {horizontal_lines} | Vertical: {vertical_lines}'
    frame = add_custom_text(frame, text, (10, 30), font_path, font_size, color=(255, 255, 255))

    # Add icon when FPS boost is on
    if use_optimized_fps:
        frame = add_icon(frame, icon, (10, 70))  # Position icon at (10, 70)

    return frame, avg_horizontal_lines, avg_vertical_lines, max_horizontal_lines, max_vertical_lines, fps
