# mbot_apriltag_project

Install apriltag:

Directly install from the official github repo: https://github.com/AprilRobotics/apriltag

```bash
git clone https://github.com/AprilRobotics/apriltag.git
```

Then:
```bash
cd apriltag
cmake -B build -DCMAKE_BUILD_TYPE=Release
sudo cmake --build build --target install
```

If encounter error "ImportError: libapriltag.so.3: cannot open shared object file: No such file or directory"

1. Verify the Installation Location

    This file is typically installed in a directory like /usr/local/lib or /usr/lib. 
    ```bash
    ls /usr/local/lib | grep libapriltag
    ```
    ```bash
    ls /usr/lib | grep libapriltag
    ```

2. Update the Library Cache

    If the library is installed in a standard location but still not found, the system's library cache may need to be updated. Run ldconfig to refresh the cache:

    ```bash
    sudo ldconfig
    ```
