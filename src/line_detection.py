# src/line_detection.py
import cv2
import numpy as np

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    horizontal_lines = 0
    vertical_lines = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float('inf')

            if abs(slope) < 0.1:  # Horizontal
                horizontal_lines += 1
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
            elif abs(slope) > 0.4:  # Vertical
                vertical_lines += 1
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
    else:
        print("No lines detected.")
    
    return frame, horizontal_lines, vertical_lines
