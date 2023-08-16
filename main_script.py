import cv2
import numpy as np
import requests
import yt_dlp
from m3u8 import loads

from maps import display_cars_on_map
from stream_mgmt import VideoStreamWidget
from utils import debugger
from zones_of_interest import Zone, handle_cars


def main(zone_configs: list, video_stream_widget: VideoStreamWidget, map_path: str, perpective_map: np.ndarray, debug: bool = False):
    while True:
        frame = video_stream_widget.get_frame()
        if frame is None:
            continue
        debug_frame = np.zeros_like(frame)
        frame_dst = frame.copy()
        frame_map = cv2.imread(map_path)
        for index, (zone, queue) in enumerate(zone_configs):
            warped_image, binary_frame = handle_cars(zone, queue, frame, frame_dst)
            display_cars_on_map(zone.actual_cars, perpective_map[index], frame_map)
            if debug:
                debugger(zone, warped_image, binary_frame, zone.actual_cars, debug_frame)
        cv2.imshow("Map", frame_map)
        # Display zone counts
        cv2.putText(frame_dst, "Periph Sud -> Nord: " + str(sum(len(zone.cars_counted) for zone, _ in zone_configs[0:2])),
                    (700, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
        cv2.putText(frame_dst, "Periph Nord -> Sud: " + str(sum(len(zone.cars_counted) for zone, _ in zone_configs[2:4])),
                    (480, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
        cv2.putText(frame_dst, "Bretelle Nord -> Sud: " + str(sum(len(zone.cars_counted) for zone, _ in zone_configs[4:6])),
                    (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
        cv2.putText(frame_dst, "Bretelle Sud -> Nord: " + str(sum(len(zone.cars_counted) for zone, _ in zone_configs[6:8])),
                    (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
        if debug:
            cv2.imshow("debug", debug_frame)
            cv2.resizeWindow("debug", debug_frame.shape[1], debug_frame.shape[0])
        cv2.imshow("Real-Time Screen Capture", frame_dst)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or cv2.getWindowProperty("Real-Time Screen Capture", cv2.WND_PROP_VISIBLE) < 1:
            video_stream_widget.capture.release()
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    livestream_url = "https://www.youtube.com/watch?v=z4vQEMiD3VI"
    video_metadata = yt_dlp.YoutubeDL().extract_info(livestream_url, download=False)
    manifest_url = video_metadata["manifest_url"]
    response = requests.get(manifest_url)
    manifest_content = response.text
    manifest = loads(manifest_content)
    playlists = manifest.data['playlists']

    RESOLUTION_854x480 = '854x480'
    playlist_url = None
    width = height = None

    for playlist in playlists:
        if playlist['stream_info']['resolution'] == RESOLUTION_854x480:
            playlist_url = playlist['uri']
            width, height = map(int, playlist['stream_info']['resolution'].split('x'))
            break
    if playlist_url is None:
        raise Exception('No playlist found')

    cv2.namedWindow("Real-Time Screen Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Screen Capture", width, height)

    zones = [
        [Zone([(575, 150), (620, 150), (840, 350), (750, 350)], 100, 200, 200/42, -15, -35, 0), []],
        [Zone([(500, 150), (540, 150), (695, 350), (620, 350)], 100, 200, 200/42, 10, -30, 1), []],
        [Zone([(475, 150), (505, 150), (620, 350), (560, 350)], 100, 200, 200/42, -10, -35, 2), []],
        [Zone([(425, 150), (450, 150), (520, 350), (460, 350)], 100, 200, 200/42, -10, -30, 3), []],
        [Zone([(245, 80), (280, 80), (115, 300), (50, 300)], 100, 200, 200/42, 0, -30, 4), []],
        [Zone([(300, 80), (365, 80), (295, 310), (150, 310)], 100, 200, 200/42, -10, -30, 5), []],
        [Zone([(615, 70), (645, 70), (845, 205), (795, 205)], 100, 200, 200/42, 10, -30, 6), []],
        [Zone([(670, 75), (730, 75), (850, 160), (820, 160)], 100, 300, 200/42, 10, -30, 7), []]
    ]
    src_points_map = []
    for zone in zones:
        zones_src = []
        zones_src.append((0, 0))
        zones_src.append((zone[0].warped_image_width, 0))
        zones_src.append((zone[0].warped_image_width, zone[0].warped_image_height))
        zones_src.append((0, zone[0].warped_image_height))
        src_points_map.append(zones_src)
    destinations_points_map = [
        [(409, 196), (419, 204), (319, 300), (311, 291)],
        [(398, 188), (407, 195), (309, 291), (300, 283)],
        [(377, 170), (386, 180), (286, 275), (278, 266)],
        [(366, 161), (379, 171), (279, 265), (267, 255)],
        [(362, 88), (375, 97), (312, 160), (305, 153)],
        [(377, 96), (388, 107), (318, 165), (314, 159)],
        [(452, 200), (462, 212), (401, 267), (395, 259)],
        [(462, 213), (475, 224), (411, 276), (402, 266)]
    ]
    perpectives_map = []
    for index, dest in enumerate(destinations_points_map):
        perpectives_map.append(cv2.getPerspectiveTransform(np.array(src_points_map[index], dtype=np.float32),
                                                           np.array(dest, dtype=np.float32)))
    video_stream_widget = VideoStreamWidget(playlist_url)
    frame = video_stream_widget.get_frame()
    while frame is None:
        frame = video_stream_widget.get_frame()

    main(zones, video_stream_widget, "map.png", perpectives_map, debug=True)
