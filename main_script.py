import cv2
from vidgear.gears import CamGear
from m3u8 import loads
import requests
from utils import debugger
from stream_mgmt import VideoStreamWidget
from zones_of_interest import Zone, handle_cars


def main(zone_configs: list, video_stream_widget: VideoStreamWidget, debug: bool = False):
    while True:
        frame = video_stream_widget.get_frame()
        if frame is None:
            continue

        frame_dst = frame.copy()
        for zone, queue in zone_configs:
            warped_image, binary_frame = handle_cars(zone, queue, frame, frame_dst)
            if debug:
                debugger(zone, warped_image, binary_frame, zone.actual_cars, frame_dst)

        # Display zone counts
        cv2.putText(frame_dst, "Sud -> Nord: " + str(sum(len(zone.cars_counted) for zone, _ in zone_configs[:2])),
                    (700, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)
        cv2.putText(frame_dst, "Nord -> Sud: " + str(sum(len(zone.cars_counted) for zone, _ in zone_configs[2:])),
                    (480, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 21, 200), 2)

        cv2.imshow("Real-Time Screen Capture", frame_dst)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or cv2.getWindowProperty("Real-Time Screen Capture", cv2.WND_PROP_VISIBLE) < 1:
            video_stream_widget.capture.release()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    livestream_url = "https://www.youtube.com/watch?v=z4vQEMiD3VI"
    stream = CamGear(source=livestream_url, stream_mode=True, logging=True).start()
    video_metadata = stream.ytv_metadata
    stream.stop()
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
        [Zone([(575, 150), (620, 150), (840, 350), (750, 350)], 100, 200, 200/42, -15, -35), []],
        [Zone([(500, 150), (530, 150), (680, 350), (620, 350)], 100, 200, 200/42, 10, -30), []],
        [Zone([(475, 150), (505, 150), (620, 350), (560, 350)], 100, 200, 200/42, -10, -35), []],
        [Zone([(425, 150), (450, 150), (520, 350), (460, 350)], 100, 200, 200/42, -10, -30), []]
    ]

    video_stream_widget = VideoStreamWidget(playlist_url)
    frame = video_stream_widget.get_frame()
    while frame is None:
        frame = video_stream_widget.get_frame()

    main(zones, video_stream_widget)
