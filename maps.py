import cv2
import numpy as np


def display_cars_on_map(cars: list, perspective_matrix: np.ndarray, frame: np.ndarray):
    for car in cars:
        car_position = np.array([car.x, car.y])
        car_position = cv2.perspectiveTransform(car_position.reshape(1, 1, 2), perspective_matrix)[0][0]
        cv2.circle(frame, (int(car_position[0]), int(car_position[1])), 5, car.color, -1)
