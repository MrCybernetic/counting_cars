import cv2
import numpy as np
import pyautogui
import time
import cv2.typing
from zones_of_interest import Zone
from math import dist

# Youtube Live Stream area, top, left, width, height
zone_of_interest = (25, 190, 890, 500)

# OpenCV window
cv2.namedWindow("Real-Time Screen Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time Screen Capture", zone_of_interest[2], zone_of_interest[3])

# Zone Sud -> Nord
SN_zone_1 = Zone(zone_of_interest=[(590, 150), (650, 150), (871, 350), (780, 350)])
queue_SN = []

# Zone Nord -> Sud
# NS_zone = Zone(zone_of_interest=[(550, 200), (630, 350), (475, 350), (450, 200)], destination_points=[[120, 0], [120, 150], [0, 150], [0, 0]])


def main():
    global SN_zone_1, NS_zone, queue_SN

    img = pyautogui.screenshot(region=zone_of_interest)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    filtered_frame = filter(frame)
    white_canvas = np.zeros((200, 100), np.uint8)

    # Zone Sud -> Nord
    SN_warped_image = SN_zone_1.get_warped_image(filtered_frame, 100, 200)
    binary_frame_SN, queue_SN = get_moving_pixels(3, SN_warped_image, queue_SN, 40, 255, 5, 3)
    # cv2.imshow("Warped_SN", SN_warped_image)
    # cv2.resizeWindow("Warped_SN", 200, 250)
    if (binary_frame_SN is not None):
        contours, _ = cv2.findContours(binary_frame_SN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            SN_zone_1.update(contour)
    for index, car in enumerate(SN_zone_1.actual_cars):
        if time.time() - car.last_seen > 0.2:
            SN_zone_1.actual_cars.pop(index)
        colored_area = car.get_colored_area(frame, SN_zone_1.perspective_matrix, zone_of_interest)
        frame = cv2.addWeighted(frame, 1, colored_area, 0.5, 0)
        cv2.circle(white_canvas, (int(car.x), int(car.y)), 20, (255, 0, 0), 1)
        # display the speed over the car
        ratio_pixel_per_meters = 200/42
        speed = (dist(car.coordinates_first_seen, (car.x, car.y)) / (time.time() - car.first_seen))*(1/ratio_pixel_per_meters)*3.6
        cv2.putText(frame, str(int(speed)) + " km/h", (50, 320+20*index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, car.color, 2)
    ok = cv2.addWeighted(SN_warped_image, 1, white_canvas, 1, 0)
    if (binary_frame_SN is not None):
        ok = cv2.addWeighted(ok, 1, binary_frame_SN, 0.2, 0)
    cv2.line(ok, (0, int(SN_zone_1.min_y)), (200, int(SN_zone_1.min_y)), (255, 255, 255), 1)
    cv2.line(ok, (0, int(SN_zone_1.max_y)), (200, int(SN_zone_1.max_y)), (255, 255, 255), 1)
    cv2.imshow("debug", ok)
    cv2.resizeWindow("debug", SN_warped_image.shape[1]*2, SN_warped_image.shape[0]*2)
    cv2.polylines(frame, np.int32([SN_zone_1.zone_of_interest]), True, (50, 50, 0), 2)

    # # Zone Nord -> Sud
    # corrected_image_NS = cv2.warpPerspective(filtered_frame, perspective_matrix_NS, (120, 150))
    # binary_frame_NS, queue_NS = get_moving_pixels(2, corrected_image_NS, queue_NS, 30, 255, 6, 2)
    # cv2.imshow("Warped_NS", corrected_image_NS)
    # cv2.resizeWindow("Warped_NS", 120, 150)
    # if (binary_frame_NS is not None):
    #     unwraped_image_NS = cv2.warpPerspective(binary_frame_NS, perspective_matrix_NS, (900, 500), flags=cv2.WARP_INVERSE_MAP)
    #     contours, _ = cv2.findContours(unwraped_image_NS.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     for contour in contours:
    #         if cv2.contourArea(contour) < 500:
    #             continue
    #         x, y, w, h = cv2.boundingRect(contour)
    #         actual_cars_NS, last_id_NS, cars_counted_NS = update_cars(x, y, w, h, 30, zone_of_interest_NS, last_id_NS, actual_cars_NS,
    #                                                                   cars_counted_NS)
    #         for car in actual_cars_NS:
    #             cv2.rectangle(frame, (car[0], car[1]), (car[0] + car[2], car[1] + car[3]), get_color_from_id(car[4]), 2)
    #         for index, car in enumerate(actual_cars_NS):
    #             if time.time() - car[5] > 0.2:
    #                 actual_cars_NS.pop(index)
    # cv2.polylines(frame, np.int32([zone_of_interest_NS]), True, (0, 50, 50), 2)

    cv2.putText(frame, "Sud -> Nord: " + str(len(SN_zone_1.cars_counted)), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
    # cv2.putText(frame, "Nord -> Sud: " + str(len(cars_counted_NS)), (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
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


if __name__ == '__main__':
    while True:
        main()
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or cv2.getWindowProperty("Real-Time Screen Capture", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the window and close
    cv2.destroyAllWindows()
