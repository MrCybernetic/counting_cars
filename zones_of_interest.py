import dataclasses
import numpy as np
import cv2
import cv2.typing
from utils import update_cars


@dataclasses.dataclass
class Zone:
    zone_of_interest: list[tuple[int, int]]
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
        # equal to min of interest zone y
        min_zone_y = min(self.zone_of_interest, key=lambda x: x[1])[1]
        max_zone_y = max(self.zone_of_interest, key=lambda x: x[1])[1]
        self.min_y = 0.25*(max_zone_y-min_zone_y)
        self.max_y = 0.75*(max_zone_y-min_zone_y)
        self.actual_cars, self.last_id, self.cars_counted = update_cars(x, y, w, h, 20, self.last_id, self.actual_cars,
                                                                        self.cars_counted, contour, self.min_y, self.max_y)
