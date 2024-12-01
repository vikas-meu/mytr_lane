import cv2
import numpy as np
import pyfirmata

# Initialize the Arduino board and pins for DC motors
board = pyfirmata.Arduino('COM12')  # Adjust the port as needed
in1 = board.get_pin('d:2:o')
in2 = board.get_pin('d:3:o')
in3 = board.get_pin('d:4:o')
in4 = board.get_pin('d:5:o')
en1 = board.get_pin('d:6:o')
en2 = board.get_pin('d:7:o')

# Initialize some variables for controlling speed and sensitivity
speed_left = 0.5  # Speed for left motor (adjustable)
speed_right = 0.5  # Speed for right motor (adjustable)
sensitivity = 0.01  # Sensitivity for steering (adjustable)

# Set the HSV range for white detection
lower_white = np.array([50, 0, 200])
upper_white = np.array([180, 25, 255])

# Define the initial points for perspective warp (focusing on the lower part of the frame)
# We'll limit the warp to the bottom half of the frame for lane tracking
frame_width = 640
frame_height = 480

# Adjust source points to focus on the lower half of the frame
src_points = np.float32([[150, 400], [490, 400], [0, frame_height], [frame_width, frame_height]])  # Adjusted points
dst_points = np.float32([[150, 0], [490, 0], [150, frame_height], [490, frame_height]])

# Function to apply perspective transform
def warp_perspective(img, src_pts, dst_pts):
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    return warped_img

# Function to detect lane in the warped image and compute steering
def detect_lane_and_steering(warped_img):
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return cx, cy
    return None, None

# Function to control the motors based on the steering
def control_motors(steering_error):
    global speed_left, speed_right

    if steering_error < -sensitivity:
        # Turn left
        in1.write(1)
        in2.write(0)
        in3.write(0)
        in4.write(1)
        speed_left = 0.3  # Slow down left motor
        speed_right = 0.7  # Speed up right motor
        print("Moving Left")
    elif steering_error > sensitivity:
        # Turn right
        in1.write(0)
        in2.write(1)
        in3.write(1)
        in4.write(0)
        speed_left = 0.7  # Speed up left motor
        speed_right = 0.3  # Slow down right motor
        print("Moving Right")
    else:
        # Go straight
        in1.write(1)
        in2.write(0)
        in3.write(1)
        in4.write(0)
        speed_left = 0.5  # Normal speed for both motors
        speed_right = 0.5  # Normal speed for both motors
        print("Moving Straight")

    # Set motor speeds
    en1.write(speed_left)
    en2.write(speed_right)

# Function to update the src points via trackbars
def update_src_points(x):
    global src_points
    src_points = np.float32([[cv2.getTrackbarPos('Point 1 X', 'Controls'), cv2.getTrackbarPos('Point 1 Y', 'Controls')],
                             [cv2.getTrackbarPos('Point 2 X', 'Controls'), cv2.getTrackbarPos('Point 2 Y', 'Controls')],
                             [cv2.getTrackbarPos('Point 3 X', 'Controls'), cv2.getTrackbarPos('Point 3 Y', 'Controls')],
                             [cv2.getTrackbarPos('Point 4 X', 'Controls'), cv2.getTrackbarPos('Point 4 Y', 'Controls')]])

# Create window for controls and add trackbars
cv2.namedWindow('Controls')

cv2.createTrackbar('Point 1 X', 'Controls', 150, 640, update_src_points)
cv2.createTrackbar('Point 1 Y', 'Controls', 400, 480, update_src_points)
cv2.createTrackbar('Point 2 X', 'Controls', 490, 640, update_src_points)
cv2.createTrackbar('Point 2 Y', 'Controls', 400, 480, update_src_points)
cv2.createTrackbar('Point 3 X', 'Controls', 0, 640, update_src_points)
cv2.createTrackbar('Point 3 Y', 'Controls', 480, 480, update_src_points)
cv2.createTrackbar('Point 4 X', 'Controls', 640, 640, update_src_points)
cv2.createTrackbar('Point 4 Y', 'Controls', 480, 480, update_src_points)

# Main loop to process the video
cap = cv2.VideoCapture(1)  # Input video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Warp the image to get the bird's eye view (focused on the lower part)
    warped_frame = warp_perspective(frame, src_points, dst_points)

    # Resize the warped image to match the original frame size for better visualization
    warped_frame_resized = cv2.resize(warped_frame, (frame.shape[1], frame.shape[0]))

    # Detect the lane and compute the steering error
    cx, cy = detect_lane_and_steering(warped_frame_resized)

    if cx is not None and cy is not None:
        # Compute the error in the x-axis (center deviation)
        steering_error = cx - warped_frame_resized.shape[1] // 2

        # Control motors based on the steering error
        control_motors(steering_error)

        # Display the center point on the warped image
        cv2.circle(warped_frame_resized, (cx, cy), 10, (0, 0, 255), -1)  # Red dot to track the center

    # Show the warped image with detected lane and motor direction
        cv2.putText(warped_frame_resized, f"Direction: {'Left' if steering_error < -sensitivity else 'Right' if steering_error > sensitivity else 'Straight'}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # HSV Thresholding Window
    hsv = cv2.cvtColor(warped_frame_resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    hsv_view = cv2.bitwise_and(warped_frame_resized, warped_frame_resized, mask=mask)
    
    # Show both the original frame, warped frame, and HSV thresholded frame side by side
    combined_frame = np.hstack((frame, warped_frame_resized, hsv_view))
    cv2.imshow("Original, Warped & HSV View", combined_frame)

    # Slow down the video processing for visualization
    cv2.waitKey(50)  # Adjust the value to control the speed (larger value = slower)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
