import cv2
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zones_of_interest import Zone


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


def debugger(zone: "Zone", warped_image: np.array, binary_frame: cv2.typing.MatLike, actual_cars: list, frame: cv2.typing.MatLike) -> None:
    black_image = np.zeros_like(warped_image)
    for car in actual_cars:
        colored_area = car.get_colored_area(frame, zone.perspective_matrix)
        frame[colored_area != 0] = colored_area[colored_area != 0]
    # in warped image, display the min y and max y lines
    cv2.line(black_image, (0, int(zone.min_y)), (black_image.shape[1], int(zone.min_y)), 255, 2)
    cv2.line(black_image, (0, int(zone.max_y)), (black_image.shape[1], int(zone.max_y)), 255, 2)
    # using perpective inverse, display the black_image on the original image
    black_image = cv2.warpPerspective(black_image, zone.perspective_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    black_image_gbr = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(frame, 1, black_image_gbr, 1, 0, frame)

    cv2.polylines(frame, np.int32([zone.zone_of_interest]), True, (150, 0, 0), 1)
    cv2.putText(frame, str(zone.num), (int(zone.zone_of_interest[0][0]), int(zone.zone_of_interest[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
