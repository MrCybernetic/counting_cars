import dataclasses
import numpy as np
import cv2
import cv2.typing
from cars import update_cars
from utils import filter, get_moving_pixels
import time


@dataclasses.dataclass
class Zone:
    zone_of_interest: list[tuple[int, int]]
    warped_image_width: int
    warped_image_height: int
    ratio_pixel_per_meters: float
    offs_x_for_circle: int
    offs_y_for_circle: int
    num: int
    actual_cars: list = dataclasses.field(default_factory=list)
    cars_counted: set = dataclasses.field(default_factory=set)
    last_id: int = 0
    min_y: int = 0
    max_y: int = 99999

    def get_warped_image(self, frame: cv2.typing.MatLike, width: int, height: int) -> np.array:
        self.distorted_points = np.array(self.zone_of_interest, dtype=np.float32)
        self.destination_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        self.perspective_matrix = cv2.getPerspectiveTransform(self.distorted_points, self.destination_points)
        self.warped_image = cv2.warpPerspective(frame, self.perspective_matrix, (width, height))
        return self.warped_image

    def update(self, contour: cv2.typing.MatLike) -> None:
        x, y, w, h = cv2.boundingRect(contour)
        min_zone_y = min(self.zone_of_interest, key=lambda x: x[1])[1]
        max_zone_y = max(self.zone_of_interest, key=lambda x: x[1])[1]
        real_coord = cv2.perspectiveTransform(np.array([[[0, min_zone_y]], [[0, max_zone_y]]], dtype=np.float32),
                                              self.perspective_matrix)
        min_zone_y = real_coord[0][0][1]
        max_zone_y = real_coord[1][0][1]
        self.min_y = 0.25*(max_zone_y-min_zone_y)
        self.max_y = 0.75*(max_zone_y-min_zone_y)
        self.actual_cars, self.last_id, self.cars_counted = update_cars(
            x, y, w, h, 20, self.last_id, self.actual_cars, self.cars_counted, contour, self.min_y, self.max_y
        )


def handle_cars(zone: Zone, queue: list, frame_src, frame_dst):
    filtered_frame = filter(frame_src)
    warped_image = zone.get_warped_image(filtered_frame, zone.warped_image_width, zone.warped_image_height)
    binary_frame, queue = get_moving_pixels(3, warped_image, queue, 40, 255, 6, 3)
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
        car.draw_circle(frame_dst, zone.perspective_matrix, zone.offs_x_for_circle, zone.offs_y_for_circle)
        # car.draw_speed(frame, ratio_pixel_per_meters, index)
    return warped_image, binary_frame
