# Object Tracking System

The Object Tracking System is a program designed for real-time object tracking using various tracking methods and neural networks.

## Features
1) Real-time object tracking from a camera or video file.
2) Video recording and screenshot capturing.
3) Multiple tracking methods: SSD Tracker, KCF, MIL, MOSSE.
4) Save results to a custom directory.
5) Compare tracking methods by measuring system ticks.

## Requirements
To run the program, you need Python and the required dependencies. Ensure you have:
- Python 3.8 or newer
- OpenCV
- NumPy
- Tkinter
- PIL (Pillow)

## Installation
1. Clone the repository:
   git clone https://github.com/VadikRakitin/ObjectTrackingSystem.git
   cd ObjectTrackingSystem
2. Install the required dependencies:
   pip install -r requirements.txt
3. Run the program:
   python main.py

## Repository Files
  main.py — the main program file.
  MobileNetSSD_deploy.prototxt — SSD model configuration file.
  MobileNetSSD_deploy.caffemodel — SSD model weights.
  haarcascade_car.xml - Haar cascade classifier file for detecting cars.
  logo.jpg — program logo.
  requirements.txt — list of dependencies.

## Usage
1) Select a video source: Choose the video source (camera or video file) by pressing the corresponding button to switch between sources.
2) Start tracking: Press START to begin tracking objects in the video. The program will start capturing frames and measure processing time using system ticks.
3) Stop tracking: Press STOP to stop tracking and pause the video. The program will display the number of system ticks, calculated using cv2.getTickCount(), which measures the video processing time.
4) Record video: Use the Start/Stop Recording button to begin or end video recording. Recorded videos are saved to the selected folder with an appropriate filename.
5) Take screenshots: Use the Screenshot button to save the current video frame as an image in the selected folder.
6) Switch tracking methods: Use the dropdown menu to select one of four available tracking methods: SSD Tracker, KCF, MIL, MOSSE.
7) Change save folder: Use the Change Save Folder button to specify a folder for saving videos and screenshots.
8) Change video source: Use the Change Source button to select another camera or video source (e.g., cameras with index 0, 1, 2, etc.).


