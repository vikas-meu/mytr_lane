import cv2
import numpy as np

# Function for warping perspective
def warp_perspective(frame):
    height, width = frame.shape[:2]

    # Define source points for the region of interest
    src = np.float32([
        [width * 0.45, height * 0.6],  # Top-left
        [width * 0.55, height * 0.6],  # Top-right
        [width * 0.9, height],        # Bottom-right
        [width * 0.1, height]         # Bottom-left
    ])

    # Define destination points for the warped frame
    dst = np.float32([
        [0, 0],                    # Top-left
        [width, 0],                # Top-right
        [width, height],           # Bottom-right
        [0, height]                # Bottom-left
    ])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Perform the perspective warp
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    return warped

# Function for filtering white color using fixed HSV values
def filter_colors_fixed(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fixed HSV values for white detection
    white_lower = np.array([0, 0, 200], np.uint8)  # Adjust here if needed
    white_upper = np.array([180, 50, 255], np.uint8)  # Adjust here if needed

    # Create the mask and filter the frame
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    filtered = cv2.bitwise_and(frame, frame, mask=white_mask)
    return filtered

# Main function
def main():
    video_path = "video.mp4"  # Path to your video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video")
        return

    while True:
        ret, frame = cap.read()

        # Loop video when it ends
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize frame for better performance
        frame = cv2.resize(frame, (640, 360))

        # Step 1: Apply the fixed HSV filter to isolate the white track
        filtered_frame = filter_colors_fixed(frame)

        # Step 2: Apply perspective warp to the filtered frame
        warped_frame = warp_perspective(filtered_frame)

        # Display the results
        cv2.imshow("Warped Frame (Bird's Eye View)", warped_frame)
        cv2.imshow("Filtered Frame", filtered_frame)
        cv2.imshow("Original Frame", frame)

        # Exit loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
