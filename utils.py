import time
from math import dist
import cv2.typing
import dataclasses
import colorsys
import numpy as np


def update_cars(x, y, w, h, distance, id, actual_cars, cars_counted: set, contour_area: cv2.typing.MatLike, y_min, y_max):
    updated_cars = actual_cars
    if (y+h/2 < y_min) or (y+h/2 > y_max):
        return updated_cars, id, cars_counted

    for index, car in enumerate(updated_cars):
        if time.time() - car.first_seen > 0.5:
            cars_counted.add(car.id)
        if dist((x+w/2, y+h/2), (car.x, car.y)) < distance:
            updated_cars[index] = Car(x+w/2, y+h/2, w, h, contour_area, car.id, time.time(), car.first_seen, car.coordinates_first_seen)
            return updated_cars, id, cars_counted
    id += 1
    updated_cars.append(Car(x+w/2, y+h/2, w, h, contour_area, id, time.time(), time.time(), (x+w/2, y+h/2)))

    return updated_cars, id, cars_counted


@dataclasses.dataclass
class Car:
    x: int
    y: int
    w: int
    h: int
    contour_area: int
    id: int
    last_seen: float
    first_seen: float
    coordinates_first_seen: tuple[int, int]

    def __post_init__(self):
        self.color = get_color_from_id(self.id)

    def get_colored_area(self, frame, perpective_matrix, zone_of_interest):
        colored_area_canvas = np.zeros_like(frame)
        colored_area = cv2.fillPoly(colored_area_canvas, [self.contour_area], self.color)
        colored_area = cv2.warpPerspective(colored_area, perpective_matrix, (zone_of_interest[2], zone_of_interest[3]), flags=cv2.WARP_INVERSE_MAP)
        return colored_area


def get_color_from_id(id):
    hue = int(id * 255 / 10)
    hue = hue % 255
    rgb_value = tuple(int(a * 255) for a in colorsys.hsv_to_rgb(hue / 255, 1, 1))
    return rgb_value
