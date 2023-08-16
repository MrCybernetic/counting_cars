import cv2
import numpy as np

img_path = "map.png"
txt_path = "coordinates.txt"
qudrltrl_path = "quadrilaterals.txt"


def display_click_and_quadrilaterals(img_path, txt_path):
    img = cv2.imread(img_path)
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', on_click)
    with open(txt_path, "w") as f:
        f.write("")
    while True:
        show_image()
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def show_image():
    img = cv2.imread(img_path)
    points = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            x, y = line.split(",")
            x, y = int(x), int(y)
            points.append((int(x), int(y)))
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        if len(points) >= 4 and len(points) % 4 == 0:
            for i in range(0, len(points), 4):
                cv2.polylines(img, np.int32([points[i:i+4]]), True, (0, 255, 0), 1)

    cv2.imshow('image', img)


def on_click(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        with open(txt_path, "a") as f:
            f.write(f"{x},{y}\n")
        nb_points = open(txt_path, "r").readlines()
        if len(nb_points) >= 4 and len(nb_points) % 4 == 0:
            with open(qudrltrl_path, "w") as f:
                for i in range(0, len(nb_points), 4):
                    f.write(f"[({nb_points[i].strip()}), ({nb_points[i+1].strip()}), ({nb_points[i+2].strip()}), ({nb_points[i+3].strip()})]\n")


def draw_quadrilaterals(img_path, qudrltrl_path):
    img = cv2.imread(img_path)
    with open(qudrltrl_path, "r") as f:
        for index, line in enumerate(f.readlines()):
            points = line.strip()[2:-2].split("), (")
            points = [tuple(map(int, point.split(","))) for point in points]
            cv2.polylines(img, np.int32([points]), True, (0, 255, 0), 1)
            cv2.putText(img, str(index), points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    display_click_and_quadrilaterals(img_path, txt_path)
    # draw_quadrilaterals(img_path, qudrltrl_path)
