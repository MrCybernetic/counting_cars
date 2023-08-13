import cv2
import numpy as np
import pyautogui
from math import dist
import time
import colorsys
import cv2.typing

zone_of_interest = (25, 190, 890, 500)
prev_frame = None

# Zone Sud -> Nord
zone_of_interest_SN = [(700, 200), (871, 350), (630, 350), (550, 200)]
actual_cars_SN = []
last_id_SN = 0
distorted_points_SN = np.array(zone_of_interest_SN, dtype=np.float32)
destination_points_SN = np.array([[200, 0], [200, 150], [0, 150], [0, 0]], dtype=np.float32)
perspective_matrix_SN = cv2.getPerspectiveTransform(distorted_points_SN, destination_points_SN)
queue_SN = []
cars_counted_SN = set()

# Zone Nord -> Sud
zone_of_interest_NS = [(550, 200), (630, 350), (475, 350), (450, 200)]
actual_cars_NS = []
last_id_NS = 0
distorted_points_NS = np.array(zone_of_interest_NS, dtype=np.float32)
destination_points_NS = np.array([[120, 0], [120, 150], [0, 150], [0, 0]], dtype=np.float32)
perspective_matrix_NS = cv2.getPerspectiveTransform(distorted_points_NS, destination_points_NS)
queue_NS = []
cars_counted_NS = set()

cv2.namedWindow("Real-Time Screen Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time Screen Capture", 900, 500)


def main():
    global actual_cars_NS, actual_cars_SN, last_id_SN, last_id_NS, queue_SN, queue_NS, cars_counted_SN, cars_counted_NS

    img = pyautogui.screenshot(region=zone_of_interest)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    filtered_frame = filter(frame)

    # Zone Sud -> Nord
    corrected_image_SN = cv2.warpPerspective(filtered_frame, perspective_matrix_SN, (200, 150))
    binary_frame_SN, queue_SN = get_moving_pixels(2, corrected_image_SN, queue_SN, 30, 255, 6, 2)
    cv2.imshow("Warped_SN", corrected_image_SN)
    cv2.resizeWindow("Warped_SN", 200, 150)
    if (binary_frame_SN is not None):        
        unwraped_image_SN = cv2.warpPerspective(binary_frame_SN, perspective_matrix_SN, (900, 500), flags=cv2.WARP_INVERSE_MAP)
        contours, _ = cv2.findContours(unwraped_image_SN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            actual_cars_SN, last_id_SN, cars_counted_SN = update_cars(x, y, w, h, 30, zone_of_interest_SN, last_id_SN, actual_cars_SN,
                                                                      cars_counted_SN)
            for car in actual_cars_SN:
                cv2.rectangle(frame, (car[0], car[1]), (car[0] + car[2], car[1] + car[3]), get_color_from_id(car[4]), 2)
    for index, car in enumerate(actual_cars_SN):
        if time.time() - car[5] > 0.2:
            actual_cars_SN.pop(index)
    cv2.polylines(frame, np.int32([zone_of_interest_SN]), True, (50, 50, 0), 2)

    # Zone Nord -> Sud
    corrected_image_NS = cv2.warpPerspective(filtered_frame, perspective_matrix_NS, (120, 150))
    binary_frame_NS, queue_NS = get_moving_pixels(2, corrected_image_NS, queue_NS, 30, 255, 6, 2)
    cv2.imshow("Warped_NS", corrected_image_NS)
    cv2.resizeWindow("Warped_NS", 120, 150)
    if (binary_frame_NS is not None):
        unwraped_image_NS = cv2.warpPerspective(binary_frame_NS, perspective_matrix_NS, (900, 500), flags=cv2.WARP_INVERSE_MAP)
        contours, _ = cv2.findContours(unwraped_image_NS.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            actual_cars_NS, last_id_NS, cars_counted_NS = update_cars(x, y, w, h, 30, zone_of_interest_NS, last_id_NS, actual_cars_NS,
                                                                      cars_counted_NS)
            for car in actual_cars_NS:
                cv2.rectangle(frame, (car[0], car[1]), (car[0] + car[2], car[1] + car[3]), get_color_from_id(car[4]), 2)
            for index, car in enumerate(actual_cars_NS):
                if time.time() - car[5] > 0.2:
                    actual_cars_NS.pop(index)
    cv2.polylines(frame, np.int32([zone_of_interest_NS]), True, (0, 50, 50), 2)
    
    cv2.putText(frame, "Sud -> Nord: " + str(len(cars_counted_SN)), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
    cv2.putText(frame, "Nord -> Sud: " + str(len(cars_counted_NS)), (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
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
            binary_frame_result = cv2.bitwise_and(binary_frame_result, binary_frames[i])
        # dilate the thresholded image to fill in holes
        binary_frame_result = cv2.dilate(binary_frame_result, None, iterations=dilate_iterations)
        # erode the thresholded image to remove noise
        binary_frame_result = cv2.erode(binary_frame_result, None, iterations=erode_iterations)
        return binary_frame_result, frames_array
    else:
        return None, frames_array


def update_cars(x, y, w, h, distance, zone, id, actual_cars, cars_counted: set):
    updated_cars = actual_cars
    # check if the center of the car rectangle is in the zone
    # zone is a list of tuple (x1, y1, x2, y2, x3, y3, x4, y4)
    if not is_inside_quadrilateral(x + w / 2, y + h / 2, zone):
        return updated_cars, id, cars_counted

    for index, car in enumerate(updated_cars):
        if time.time() - car[6] > 0.5:
            cars_counted.add(car[4])
        if dist((x, y), (car[0], car[1])) < distance:
            updated_cars[index] = (x, y, w, h, car[4], time.time(), car[6])
            return updated_cars, id, cars_counted
    id += 1
    updated_cars.append((x, y, w, h, id, time.time(), time.time()))

    return updated_cars, id, cars_counted


def get_color_from_id(id):
    # id is the id of the car
    # color is chosen in the hue color space
    # the color is chosen by the id of the car
    # the color is chosen in the range of 0 to 255 but the id is in the range of 0 to 1000
    hue = int(id * 255 / 10)
    hue = hue % 255
    rgb_value = tuple(int(a * 255) for a in colorsys.hsv_to_rgb(hue / 255, 1, 1))
    return rgb_value


def area(x1, y1, x2, y2, x3, y3):
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0


def is_inside_quadrilateral(x, y, points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    total_area = area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3)

    sub_area1 = area(x, y, x1, y1, x2, y2)
    sub_area2 = area(x, y, x2, y2, x3, y3)
    sub_area3 = area(x, y, x3, y3, x4, y4)
    sub_area4 = area(x, y, x1, y1, x4, y4)

    return total_area == sub_area1 + sub_area2 + sub_area3 + sub_area4


if __name__ == '__main__':
    while True:
        main()
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or cv2.getWindowProperty("Real-Time Screen Capture", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the window and close
    cv2.destroyAllWindows()
