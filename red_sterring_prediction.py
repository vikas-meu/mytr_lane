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
    return filtered, white_mask

# Function to find the center of the white region in the frame
def find_center_of_lane(mask):
    # Find contours in the mask (white region)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Find the largest contour (lane)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the moments of the largest contour to find the centroid
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
    return None

# Function to calculate the steering angle based on the X position of the lane center
def calculate_steering_angle(center_x, frame_width):
    # Assuming the center of the frame is the ideal lane position
    ideal_center = frame_width // 2
    error = center_x - ideal_center  # Difference from the center
    max_angle = 30  # Max steering angle
    steering_angle = (error / ideal_center) * max_angle
    return steering_angle

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
        filtered_frame, white_mask = filter_colors_fixed(frame)

        # Step 2: Apply perspective warp with the fixed points
        warped_frame = warp_perspective_dynamic(filtered_frame, src_points)

        # Step 3: Find the center of the lane in the warped frame (based on the white region)
        center = find_center_of_lane(white_mask)
        
        if center:
            # Step 4: Draw a red dot at the center of the detected lane
            cx, cy = center
            cv2.circle(warped_frame, (cx, cy), 10, (0, 0, 255), -1)  # Red dot

            # Step 5: Calculate the steering angle based on the X position of the center
            steering_angle = calculate_steering_angle(cx, warped_frame.shape[1])

            # Display the calculated steering angle
            cv2.putText(warped_frame, f"Steering Angle: {steering_angle:.2f} degrees", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Step 6: Show the original and warped frames in an overlapped window
        combined_frame = np.hstack((frame, warped_frame))  # Stack original and warped frames horizontally
        cv2.imshow("Original and Warped Frame", combined_frame)

        # Exit loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
