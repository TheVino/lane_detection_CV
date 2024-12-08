import cv2
import numpy as np
import time

# Initialize video capture with the video file
cap = cv2.VideoCapture(r'G:\Desktop\Script\lane_detection_CV\lanes_clip.mp4')

# Check if the video file is opened correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Function to filter shades of gray, yellow, and white
def filter_colors(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for gray, yellow, and white
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([180, 50, 200])

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    
    # Create masks for each color range
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine all masks
    mask_combined = cv2.bitwise_or(mask_gray, mask_yellow)
    mask_combined = cv2.bitwise_or(mask_combined, mask_white)

    # Apply the mask to the original frame to isolate colors
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask_combined)
    
    return filtered_frame, mask_combined

# Initialize variables to track FPS and line counts
frame_count = 0
fps_list = []
horizontal_lines_list = []
vertical_lines_list = []

max_horizontal_lines = 0
max_vertical_lines = 0

start_time = time.time()  # Track time for FPS calculation

while True:
    # Read a frame from the video file
    ret, frame = cap.read()
    
    # If no frame is returned, exit the loop (end of video)
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # Increment frame count
    frame_count += 1

    # Filter the colors (shades of gray, yellow, and white) for line detection
    filtered_frame, mask_combined = filter_colors(frame)

    # Convert the filtered frame to grayscale for simpler processing
    gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise in the image, which improves edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines in the image using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Initialize counters for horizontal and vertical lines
    horizontal_lines = 0
    vertical_lines = 0

    # Draw the detected lines and classify them as horizontal or vertical
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate the slope of the line (avoid division by zero)
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float('inf')  # Vertical line

            # Classify the line as horizontal or vertical based on the slope
            if abs(slope) < 0.1:  # Horizontal line (small slope)
                horizontal_lines += 1
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for horizontal lines
            elif abs(slope) > 0.4:  # Vertical line (large slope)
                vertical_lines += 1
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for vertical lines

    # Update max horizontal and vertical lines
    max_horizontal_lines = max(max_horizontal_lines, horizontal_lines)
    max_vertical_lines = max(max_vertical_lines, vertical_lines)

    # Add line counts to lists
    horizontal_lines_list.append(horizontal_lines)
    vertical_lines_list.append(vertical_lines)

    # Calculate FPS for the current frame
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        fps_list.append(fps)
    else:
        fps_list.append(0)

    # Add text to the frame showing the number of detected lines and FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'FPS: {fps:.2f} | Horizontal: {horizontal_lines} | Vertical: {vertical_lines}'
    cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)  # Yellow text

    # Display the original frame with the detected lines and line count overlaid
    cv2.imshow('Filtered Line Detection', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate average FPS, horizontal lines, and vertical lines
average_fps = np.mean(fps_list) if fps_list else 0
average_horizontal_lines = np.mean(horizontal_lines_list) if horizontal_lines_list else 0
average_vertical_lines = np.mean(vertical_lines_list) if vertical_lines_list else 0

# Print the statistics at the end of the video execution
print(f"Average FPS: {average_fps:.2f}")
print(f"Average Horizontal Lines: {average_horizontal_lines:.2f}")
print(f"Average Vertical Lines: {average_vertical_lines:.2f}")
print(f"Max Horizontal Lines: {max_horizontal_lines}")
print(f"Max Vertical Lines: {max_vertical_lines}")

# Release the capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
