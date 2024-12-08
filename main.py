import cv2
import numpy as np
import time

# Initialize video capture with the video file
cap = cv2.VideoCapture(r'G:\Desktop\Script\lane_detection_CV\lanes_clip.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Function to filter shades of gray, yellow, and white
def filter_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([180, 50, 200])
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_combined = cv2.bitwise_or(mask_gray, mask_yellow)
    mask_combined = cv2.bitwise_or(mask_combined, mask_white)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask_combined)
    return filtered_frame

# Function to calculate FPS using the old method
def calculate_fps_old(start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    return 1 / elapsed_time if elapsed_time > 0 else 0, current_time

# Function to calculate FPS every 30 frames
def calculate_fps_new(frame_count, start_time):
    if frame_count % 30 == 0 and frame_count > 0:
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = 30 / elapsed_time if elapsed_time > 0 else 0
        return fps, current_time
    return None, start_time

# Function to choose FPS calculation method
def fps_boost(method, frame_count, start_time):
    if method == "old":
        return calculate_fps_old(start_time)
    elif method == "new":
        return calculate_fps_new(frame_count, start_time)
    else:
        raise ValueError("Invalid FPS calculation method. Use 'old' or 'new'.")

# Set FPS calculation method: 'old' or 'new'
fps_method = "old"  # Change to "old" to use the old method

# Initialize FPS variables
frame_count = 0
fps_list = []
start_time = time.time()
last_calculated_fps = 0

# Initialize counters for average statistics
total_horizontal_lines = 0
total_vertical_lines = 0
max_horizontal_lines = 0
max_vertical_lines = 0

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    frame_count += 1

    # FPS Calculation
    fps, start_time = fps_boost(fps_method, frame_count, start_time)
    if fps is not None:
        last_calculated_fps = fps
        fps_list.append(fps)

    # Filter colors for line detection
    filtered_frame = filter_colors(frame)
    gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    horizontal_lines = 0
    vertical_lines = 0

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

    # Update line statistics
    total_horizontal_lines += horizontal_lines
    total_vertical_lines += vertical_lines
    max_horizontal_lines = max(max_horizontal_lines, horizontal_lines)
    max_vertical_lines = max(max_vertical_lines, vertical_lines)

    # Display FPS and line statistics
    text = f'FPS: {last_calculated_fps:.2f} | Horizontal: {horizontal_lines} | Vertical: {vertical_lines}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Filtered Line Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate average statistics
average_fps = np.mean(fps_list) if fps_list else 0
average_horizontal_lines = total_horizontal_lines / frame_count
average_vertical_lines = total_vertical_lines / frame_count

# Print summary
print(f"Average FPS: {average_fps:.2f}")
print(f"Average Horizontal Lines: {average_horizontal_lines:.2f}")
print(f"Average Vertical Lines: {average_vertical_lines:.2f}")
print(f"Max Horizontal Lines: {max_horizontal_lines}")
print(f"Max Vertical Lines: {max_vertical_lines}")

cap.release()
cv2.destroyAllWindows()
