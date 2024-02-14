# mbot_apriltag_project

## How to use 
### Installation
1. Install Apriltag Library
Directly install from the official [github repo](https://github.com/AprilRobotics/apriltag):
    ```bash
    git clone https://github.com/AprilRobotics/apriltag.git
    ```

2. Following the install [instruction](https://github.com/AprilRobotics/apriltag) there:
    ```bash
    cd apriltag
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    sudo cmake --build build --target install
    ```

### Clone this repo
```bash
git clone https://github.com/MBot-Project-Development/mbot_apriltag.git
```

### Run scripts
- video_streamer.py 
    - Only shows the video stream to test your camera
    - Run `python3 video_streamer.py` then visit `http://your_mbot_ip:5001/video`
- save_image.py 
    - Show the video stream to save image to `/images` for camera calibration
    - Run `python3 save_image.py` then visit `http://your_mbot_ip:5001`
- camera_calibration.py 
    - Use the images from `/images` output result as `cam_calibration_data.npz`. The result will be used directly by apriltag_streamer.py you don't have to modify anything.
    - Run `python3 camera_calibration.py`
- apriltag_streamer.py
    - Run `python3 apriltag_streamer.py` then visit `http://your_mbot_ip:5001/video`
    - It runs apriltag detection, when tag is detected, pose estimation will be printed on the screen.
     ![](example.png)

### Troubleshooting
If encounter error during runtime:"ImportError: libapriltag.so.3: cannot open shared object file: No such file or directory"

1. Verify the Installation Location

    This file is typically installed in a directory like /usr/local/lib or /usr/lib. 
    ```bash
    ls /usr/local/lib | grep libapriltag
    ```
    ```bash
    ls /usr/lib | grep libapriltag
    ```
    - If there is output showing "libapriltag.so.3", we move to the next step

2. Update the Library Cache

    If the library is installed in a standard location but still not found, the system's library cache may need to be updated. Run ldconfig to refresh the cache:

    ```bash
    sudo ldconfig
    ```

### Author and maintainer
- The original author of this project is Shaw Sun.
- The current maintainer of this project is Shaw Sun. Please direct all questions regarding support, contributions, and issues to the maintainer. The maintainer is responsible for overseeing the project's development, reviewing and merging contributions, and ensuring the project's ongoing stability and relevance.
