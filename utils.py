import cv2
import numpy as np


def filter(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    filtered_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered_frame = cv2.GaussianBlur(filtered_frame, (3, 3), 0)
    return filtered_frame


def get_moving_pixels(number_of_frame: int, actual_frame, frames_array: list, tresh_value: int, max_value: int,
                      dilate_iterations: int, erode_iterations: int) -> cv2.typing.MatLike:
    frames_array.append(actual_frame)
    if len(frames_array) > number_of_frame:
        frames_array.pop(0)
        binary_frames = [
            cv2.threshold(cv2.absdiff(frames_array[i], frames_array[-1]), tresh_value, max_value, cv2.THRESH_BINARY)[1]
            for i in range(len(frames_array) - 1)
        ]
        binary_frame_result = cv2.bitwise_or(*binary_frames)
        binary_frame_result = cv2.dilate(binary_frame_result, None, iterations=dilate_iterations)
        binary_frame_result = cv2.erode(binary_frame_result, None, iterations=erode_iterations)
        return binary_frame_result, frames_array
    else:
        return None, frames_array


def debugger(zone, warped_image, binary_frame, actual_cars, frame):
    cv2.polylines(frame, np.int32([zone.zone_of_interest]), True, (50, 50, 0), 2)
    white_canvas = np.zeros((200, 100), np.uint8)
    for car in actual_cars:
        colored_area = car.get_colored_area(frame, zone.perspective_matrix)
        frame = cv2.addWeighted(frame, 1, colored_area, 0.5, 0)
        cv2.circle(white_canvas, (int(car.x), int(car.y)), 20, (255, 0, 0), 1)
    ok = cv2.addWeighted(warped_image, 1, white_canvas, 1, 0)
    if (binary_frame is not None):
        ok = cv2.addWeighted(ok, 1, binary_frame, 0.2, 0)
    cv2.imshow("Warped_SN", warped_image)
    cv2.resizeWindow("Warped_SN", 200, 250)
    cv2.imshow("debug", ok)
    cv2.resizeWindow("debug", warped_image.shape[1]*2, warped_image.shape[0]*2)
