# Traffic Monitoring System

![image](https://github.com/MrCybernetic/counting_cars/assets/3818606/49f78e2d-0017-4dac-98a2-88ba2cef9aea) 

## Overview

The Traffic Monitoring System is a Python-based project that uses computer vision techniques to monitor traffic and detect vehicles in real-time. The system is designed to work with live video streams and analyze vehicle movements within specific zones of interest.

## Features

- Real-time vehicle detection and tracking within designated zones.
- Vehicle speed estimation based on pixel-to-meter ratio.
- Integration with live video streams from YouTube ([Periph Nord Lyon Live](https://www.youtube.com/watch?v=z4vQEMiD3VI "live"))
- Modular design for easy expansion and customization.
- Graphical user interface for displaying processed video and zone statistics.

## Requirements

- Python 3.x
- OpenCV (cv2)
- m3u8
- Requests

## Getting Started

1. Clone the repository:

```bash
   git clone https://github.com/MrCybernetic/counting_cars.git
   cd counting_cars
```
2. Install the required dependencies:

```bash
   pip install -r requirements.txt
```
3. Run the main script:
```bash
   python main_script.py
   ```

## Usage
Setup the zones in the main script:
```python
    # see Zone class for more details
    zones = [
        [Zone([(575, 150), (620, 150), (840, 350), (750, 350)], 100, 200, 200/42, -15, -35), []],
        [Zone([(500, 150), (530, 150), (680, 350), (620, 350)], 100, 200, 200/42, 10, -30), []],
        [Zone([(475, 150), (505, 150), (620, 350), (560, 350)], 100, 200, 200/42, -10, -35), []],
        [Zone([(425, 150), (450, 150), (520, 350), (460, 350)], 100, 200, 200/42, -10, -30), []]
    ]
    
```

## Project Structure
- **main_script.py:** Main entry point of the system.
- **zones_of_interest.py:** Defines the Zone class and its methods.
- **cars.py:** Implements vehicle tracking and speed estimation.
- **utils.py:** Contains utility functions used throughout the project.
- **stream_mgmt.py:** Handles video streaming and frame retrieval.

## Contributing
Contributions are welcome! If you find any issues or have improvements to suggest, please create a pull request.

## Acknowledgements
- [OpenCV](https://opencv.org/)
- [m3u8](https://pypi.org/project/m3u8/)
- [Requests](https://pypi.org/project/requests/)
- [Periph Nord Lyon Live](https://www.youtube.com/watch?v=z4vQEMiD3VI)
- [yt_dlp](https://github.com/yt-dlp/yt-dlp)

## License
MIT License : [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) 