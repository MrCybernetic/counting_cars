import time
from math import dist
import cv2.typing
import dataclasses
import colorsys
import numpy as np


def update_cars(x: int, y: int, w: int, h: int, max_distance: int, id: int, actual_cars: list, cars_counted: set,
                contour_area: cv2.typing.MatLike, y_min: int, y_max: int):
    updated_cars = actual_cars

    if (y + h / 2 < y_min) or (y + h / 2 > y_max):
        return updated_cars, id, cars_counted

    for index, car in enumerate(updated_cars):
        if time.time() - car.first_seen > 0.5:
            cars_counted.add(car.id)
        if dist((x + w / 2, y + h / 2), (car.x, car.y)) < max_distance:
            updated_cars[index] = Car(x + w / 2, y + h / 2, w, h, contour_area, car.id, time.time(), car.first_seen, car.coordinates_first_seen)
            return updated_cars, id, cars_counted
    id += 1
    updated_cars.append(Car(x + w / 2, y + h / 2, w, h, contour_area, id, time.time(), time.time(), (x + w / 2, y + h / 2)))

    return updated_cars, id, cars_counted


@dataclasses.dataclass
class Car:
    x: int
    y: int
    w: int
    h: int
    contour_area: cv2.typing.MatLike
    id: int
    last_seen: float
    first_seen: float
    coordinates_first_seen: tuple[int, int]

    def __post_init__(self):
        self.color = get_color_from_id(self.id)

    def get_colored_area(self, frame: np.ndarray, perspective_matrix: np.ndarray) -> np.ndarray:
        colored_area_canvas = np.zeros_like(frame)
        colored_area = cv2.fillPoly(colored_area_canvas, [self.contour_area], self.color)
        colored_area = cv2.warpPerspective(colored_area, perspective_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)
        return colored_area

    def draw_circle(self, frame: np.ndarray, perspective_matrix: np.ndarray, offs_x: int, offs_y: int) -> np.ndarray:
        x, y = self.x, self.y
        x, y = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), np.linalg.pinv(perspective_matrix))[0][0]
        cv2.circle(frame, (int(x) + offs_x, int(y) + offs_y), 8, self.color, -1)
        return frame

    def draw_speed(self, frame: np.ndarray, ratio_pixel_per_meters: float, index: int) -> None:
        elapsed_time = time.time() - self.first_seen
        if elapsed_time != 0:
            speed = (dist(self.coordinates_first_seen, (self.x, self.y)) / elapsed_time) * (1 / ratio_pixel_per_meters) * 3.6
            cv2.putText(frame, str(int(speed)) + " km/h", (50, 320 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)


def get_color_from_id(id: int) -> tuple[int, int, int]:
    hue = int(id * 255 / 10)
    hue = hue % 255
    rgb_value = tuple(int(a * 255) for a in colorsys.hsv_to_rgb(hue / 255, 1, 1))
    return rgb_value
