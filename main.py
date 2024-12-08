import cv2
import numpy as np

# Initialize video capture with the video file
cap = cv2.VideoCapture(r'G:\Desktop\Script\lane_detection_CV\lanes_clip.mp4')

# Function to filter shades of gray, yellow, and white
def filter_colors(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for gray, yellow, and white
    # Gray: Low saturation, high value, within a certain range of hues
    lower_gray = np.array([0, 0, 100])    # Low saturation, light gray
    upper_gray = np.array([180, 50, 200])  # High value, mid gray

    # Yellow: Hue range for yellow, moderate saturation and high value
    # Adjusted the hue range and saturation range for a better detection of yellow
    lower_yellow = np.array([15, 100, 100])  # Broader yellow range
    upper_yellow = np.array([40, 255, 255])

    # White: High value, low saturation
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

# Initialize list to store vertical line positions
vertical_lines_positions = []

while True:
    # Read a frame from the video file
    ret, frame = cap.read()
    
    # If no frame is returned, exit the loop (end of video)
    if not ret:
        break

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
            elif abs(slope) > 0.6:  # Vertical line (large slope)
                vertical_lines += 1
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for vertical lines

                # Add vertical line position to the list
                vertical_lines_positions.append((x1, x2))

    # Add text to the frame showing the number of detected lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'Horizontal: {horizontal_lines} | Vertical: {vertical_lines}'
    cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)  # Yellow text

    # Display the original frame with the detected lines and line count overlaid
    cv2.imshow('Filtered Line Detection', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
