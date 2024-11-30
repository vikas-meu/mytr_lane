import cv2
import numpy as np

# Function for trackbar callback (does nothing)
def nothing(x):
    pass

# Function for warping perspective based on fixed four points
def warp_perspective_dynamic(frame, points):
    height, width = frame.shape[:2]

    # Define source points (fixed four points for the region of interest)
    src = np.float32(points)

    # Define destination points (we want the warped frame to be the same size as the original)
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
    white_lower = np.array([60, 0, 200], np.uint8)  # Adjust here if needed
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

    # Define four fixed points for the region of interest (adjust based on your track)
    src_points = [
        [150, 200],  # Top-left
        [500, 200],  # Top-right
        [550, 350],  # Bottom-right
        [100, 350]   # Bottom-left
    ]

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

        # Step 2: Apply perspective warp with the fixed points
        warped_frame = warp_perspective_dynamic(filtered_frame, src_points)

        # Step 3: Show the original frame and the warped frame in an overlapped window
        combined_frame = np.hstack((frame, warped_frame))  # Stack original and warped frames horizontally

        # Display the combined frames
        cv2.imshow("Original and Warped Frame", combined_frame)

        # Exit loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
