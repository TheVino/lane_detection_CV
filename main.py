# main.py
import os
import cv2
from src.video_processing import initialize_video_capture, process_frame
from src.utils import calculate_averages
from src.overlay import add_custom_text, add_icon

# Custom font setup
font_path = "assets/fonts/Ubuntu-Regular.ttf"  # Replace with the correct path to your .ttf font file
font_size = 24

# Load the FPS boost icon
icon_path = "assets/fps_boost.png"  # Replace with the correct path to your icon file
icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

# Check if the font exists
if not os.path.exists(font_path):
    print(f"Error: Font file not found at {font_path}.")
    exit()

# Main function
def main():
    # Initialize video capture
    cap = initialize_video_capture(r'G:\Desktop\Script\lane_detection_CV\lanes_clip.mp4')

    # Toggle FPS mode variable
    use_optimized_fps = True

    # Statistics tracking
    avg_vertical_lines = 0
    avg_horizontal_lines = 0
    max_vertical_lines = 0
    max_horizontal_lines = 0
    total_frames = 0
    fps = 30  # Set an initial reasonable FPS value

    # Processing loop
    while True:
        ret, frame = cap.read()
        
        # Exit loop if no frame is returned
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # Count total frames
        total_frames += 1

        # Process the frame and update stats
        frame, avg_horizontal_lines, avg_vertical_lines, max_horizontal_lines, max_vertical_lines, fps = process_frame(
            frame, total_frames, avg_horizontal_lines, avg_vertical_lines, max_horizontal_lines, max_vertical_lines,
            font_path, font_size, icon, use_optimized_fps
        )

        # Display the frame
        cv2.imshow('Lane Detection', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('f'):  # Toggle FPS mode
            use_optimized_fps = not use_optimized_fps

    # Calculate averages and print stats
    avg_horizontal_lines, avg_vertical_lines = calculate_averages(total_frames, avg_horizontal_lines, avg_vertical_lines)
    print(f"Average FPS: {fps:.2f}")
    print(f"Average Horizontal Lines: {avg_horizontal_lines:.2f}")
    print(f"Average Vertical Lines: {avg_vertical_lines:.2f}")
    print(f"Max Horizontal Lines: {max_horizontal_lines}")
    print(f"Max Vertical Lines: {max_vertical_lines}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
