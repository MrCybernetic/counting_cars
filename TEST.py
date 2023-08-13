import cv2
import numpy as np
import pyautogui

zone_of_interest = (25, 190, 890, 500)
img = pyautogui.screenshot(region=zone_of_interest)
frame = np.array(img)
zone_of_interest_SN = [(560, 100), (871, 350), (630, 350), (475, 100)]

# Define the distorted area points
distorted_points = np.array(zone_of_interest_SN, dtype=np.float32)

# Define the desired destination points (rectangular)
destination_points = np.array([[200, 0], [200, 400], [0, 400], [0, 0]], dtype=np.float32)

# Calculate the perspective transformation matrix
perspective_matrix = cv2.getPerspectiveTransform(distorted_points, destination_points)

while True:
    img = pyautogui.screenshot(region=zone_of_interest)
    frame = np.array(img)
    corrected_image = cv2.warpPerspective(frame, perspective_matrix, (200, 400))

    # Now you can perform any image processing on the corrected area
    # For example, you can apply filters, object detection, etc.

    # Display the corrected image
    cv2.imshow("Corrected Image", corrected_image)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q') or cv2.getWindowProperty("Corrected Image", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the window and close
cv2.destroyAllWindows()
