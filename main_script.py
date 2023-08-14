import cv2
import numpy as np
import pyautogui
import time
import cv2.typing
from zones_of_interest import Zone

# Youtube Live Stream area, top, left, width, height
# Livestream : https://www.youtube.com/watch?v=z4vQEMiD3VI
# youtube window in the left of your primary screen
zone_of_interest = (25, 190, 890, 500)

# OpenCV window
cv2.namedWindow("Real-Time Screen Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time Screen Capture", zone_of_interest[2], zone_of_interest[3])

# Zone Sud -> Nord
SN_zone_1 = Zone(zone_of_interest=[(590, 150), (650, 150), (871, 350), (780, 350)])
queue_SN_1 = []
SN_zone_2 = Zone(zone_of_interest=[(515, 150), (550, 150), (720, 350), (640, 350)])
queue_SN_2 = []

# Zone Nord -> Sud
NS_zone_1 = Zone(zone_of_interest=[(475, 150), (515, 150), (640, 350), (560, 350)])
queue_NS_1 = []
NS_zone_2 = Zone(zone_of_interest=[(425, 150), (450, 150), (520, 350), (460, 350)])
queue_NS_2 = []


def main():
    global SN_zone_1, queue_SN_1, SN_zone_2, queue_SN_2, NS_zone_1, queue_NS_1, NS_zone_2, queue_NS_2

    img = pyautogui.screenshot(region=zone_of_interest)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Zone Sud -> Nord 1
    SN_warped_image_1, binary_frame_SN_1 = handle_cars(SN_zone_1, queue_SN_1, frame, 100, 200, 200/42, -15, -35)
    # debugger(SN_zone_1, SN_warped_image_1, binary_frame_SN_1, SN_zone_1.actual_cars, frame)

    # Zone Sud -> Nord 2
    SN_warped_image_2, binary_frame_S2 = handle_cars(SN_zone_2, queue_SN_2, frame, 100, 200, 200/42, 10, -30)
    # debugger(SN_zone_2, SN_warped_image_2, binary_frame_S2, SN_zone_1.actual_cars, frame)

    # Zone Nord -> Sud 1
    NS_warped_image_1, binary_frame_NS_1 = handle_cars(NS_zone_1, queue_NS_1, frame, 100, 200, 200/42, -10, -35)
    # debugger(NS_zone_1, NS_warped_image_1, binary_frame_NS_1, NS_zone_1.actual_cars, frame)

    # Zone Nord -> Sud 2
    NS_warped_image_2, binary_frame_NS_2 = handle_cars(NS_zone_2, queue_NS_2, frame, 100, 200, 200/42, -10, -30)
    # debugger(NS_zone_2, NS_warped_image_2, binary_frame_NS_2, NS_zone_1.actual_cars, frame)

    cv2.putText(frame, "Sud -> Nord: " + str(len(SN_zone_1.cars_counted)+len(SN_zone_2.cars_counted)),
                (730, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
    cv2.putText(frame, "Nord -> Sud: " + str(len(NS_zone_1.cars_counted)+len(NS_zone_2.cars_counted)),
                (510, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
    cv2.imshow("Real-Time Screen Capture", frame)


def filter(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    filtered_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gaussian blur to reduce high frequency noise
    filtered_frame = cv2.GaussianBlur(filtered_frame, (3, 3), 0)
    return filtered_frame


def get_moving_pixels(number_of_frame: int, actual_frame, frames_array: list, tresh_value: int, max_value: int,
                      dilate_iterations: int, erode_iterations: int) -> cv2.typing.MatLike:
    frames_array.append(actual_frame)
    if len(frames_array) > number_of_frame:
        frames_array.pop(0)
        frame_deltas = []
        for i in range(len(frames_array) - 1):
            frame_deltas.append(cv2.absdiff(frames_array[i], frames_array[-1]))
        binary_frames = []
        for frame_delta in frame_deltas:
            binary_frames.append(cv2.threshold(frame_delta, tresh_value, max_value, cv2.THRESH_BINARY)[1])
        # keep pixels that have changed in all frames
        binary_frame_result = binary_frames[0]
        # using bitwise_and to keep only the pixels that have changed in all frames
        for i in range(len(binary_frames) - 1):
            binary_frame_result = cv2.bitwise_or(binary_frame_result, binary_frames[i])
        # dilate the thresholded image to fill in holes
        binary_frame_result = cv2.dilate(binary_frame_result, None, iterations=dilate_iterations)
        # erode the thresholded image to remove noise
        binary_frame_result = cv2.erode(binary_frame_result, None, iterations=erode_iterations)
        return binary_frame_result, frames_array
    else:
        return None, frames_array


def debugger(zone, warped_image, binary_frame, actual_cars, frame):
    cv2.polylines(frame, np.int32([zone.zone_of_interest]), True, (50, 50, 0), 2)
    white_canvas = np.zeros((200, 100), np.uint8)
    for car in actual_cars:
        colored_area = car.get_colored_area(frame, zone.perspective_matrix, zone_of_interest)
        frame = cv2.addWeighted(frame, 1, colored_area, 0.5, 0)
        cv2.circle(white_canvas, (int(car.x), int(car.y)), 20, (255, 0, 0), 1)
    ok = cv2.addWeighted(warped_image, 1, white_canvas, 1, 0)
    if (binary_frame is not None):
        ok = cv2.addWeighted(ok, 1, binary_frame, 0.2, 0)
    cv2.imshow("Warped_SN", warped_image)
    cv2.resizeWindow("Warped_SN", 200, 250)
    cv2.imshow("debug", ok)
    cv2.resizeWindow("debug", warped_image.shape[1]*2, warped_image.shape[0]*2)


def handle_cars(zone, queue, frame, warped_image_width, warped_image_height, ratio_pixel_per_meters, offs_x, offs_y):
    filtered_frame = filter(frame)
    warped_image = zone.get_warped_image(filtered_frame, warped_image_width, warped_image_height)
    binary_frame, queue = get_moving_pixels(3, warped_image, queue, 40, 255, 5, 3)
    if (binary_frame is not None):
        contours, _ = cv2.findContours(binary_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            zone.update(contour)
    for index, car in enumerate(zone.actual_cars):
        if time.time() - car.last_seen > 0.1:
            zone.actual_cars.pop(index)
            break
        car.draw_circle(frame, zone.perspective_matrix, offs_x, offs_y)
        # car.draw_speed(frame, ratio_pixel_per_meters, index)
    return warped_image, binary_frame


if __name__ == '__main__':
    while True:
        main()
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or cv2.getWindowProperty("Real-Time Screen Capture", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the window and close
    cv2.destroyAllWindows()
