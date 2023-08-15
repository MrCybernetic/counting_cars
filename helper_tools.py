import cv2

img_path = "2023-08-15 19_48_26-Real-Time Screen Capture.png"
txt_path = "coordinates.txt"


def main():
    img = cv2.imread(img_path)
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', on_click)
    # reset file
    with open("coordinates.txt", "w") as f:
        f.write("")

    while True:
        show_image()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def show_image():
    img = cv2.imread(img_path)
    with open(txt_path, "r") as f:
        for line in f.readlines():
            x, y = line.split(",")
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow('image', img)


def on_click(event, x, y, p1, p2):
    if event == cv2.EVENT_LBUTTONDOWN:
        with open("coordinates.txt", "a") as f:
            f.write(f"{x},{y}\n")


if __name__ == "__main__":
    main()
