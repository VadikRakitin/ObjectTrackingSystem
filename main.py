import tkinter as tk
from tkinter import ttk, simpledialog, filedialog
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from datetime import datetime

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Tracking System")
        self.root.geometry("900x500")

        # Store the index of the selected camera
        self.selected_camera_index = 0  # Default camera is 0
        self.video_file_path = None  # To store the path of the selected video file

        self.tracking_method = tk.StringVar(value="SSD Tracker")

        # Variables for counting frames and time
        self.frame_count = 0  # Frame counter
        self.start_time = None  # Video start time

        # Main frame for structuring the interface
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Frame for buttons (left)
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Configure weight distribution among rows in button_frame
        button_frame.rowconfigure(0, weight=0)  # Logo
        button_frame.rowconfigure(1, weight=1)  # Start/Stop
        button_frame.rowconfigure(2, weight=1)  # Start/Stop Recording
        button_frame.rowconfigure(3, weight=1)  # Screenshot
        button_frame.rowconfigure(4, weight=1)  # Change Save Folder
        button_frame.rowconfigure(5, weight=1)  # Change Source
        button_frame.rowconfigure(6, weight=0)  # Exit

        # Logo
        logo_image = Image.open("logo.jpg")
        logo_image = logo_image.resize((150, 100), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = ttk.Label(button_frame, image=logo_photo)
        logo_label.image = logo_photo  # Keep a reference to prevent garbage collection

        # Center the logo
        logo_label.grid(row=0, column=0, pady=(20, 10), padx=(60, 0), sticky="nsew")

        # Center the column
        button_frame.columnconfigure(0, weight=1)

        # Start and Stop buttons in one row
        start_stop_frame = ttk.Frame(button_frame)
        start_stop_frame.grid(row=1, column=0, pady=5, sticky="ew")

        self.start_button = ttk.Button(start_stop_frame, text="START", command=self.start_video)
        self.start_button.pack(side="left", padx=5, ipadx=10, ipady=5, fill="x")

        self.stop_button = ttk.Button(start_stop_frame, text="STOP", command=self.stop_video)
        self.stop_button.pack(side="right", padx=5, ipadx=10, ipady=5, fill="x")

        # Button for recording with larger padding
        self.record_button = ttk.Button(button_frame, text="Start/Stop Recording", command=self.toggle_recording)
        self.record_button.grid(row=2, column=0, pady=20, ipadx=10, ipady=10, sticky="ew")

        # Screenshot button with larger padding
        self.screenshot_button = ttk.Button(button_frame, text="Screenshot", command=self.take_screenshot)
        self.screenshot_button.grid(row=3, column=0, pady=15, ipadx=10, ipady=10, sticky="ew")

        # Button for changing the save folder
        self.change_folder_button = ttk.Button(button_frame, text="Change Save Folder", command=self.change_save_folder)
        self.change_folder_button.grid(row=4, column=0, pady=15, ipadx=10, ipady=10, sticky="ew")

        # Create a frame to hold both the "Load Video File" button and the tracking method selector
        video_and_tracking_frame = ttk.Frame(button_frame)
        video_and_tracking_frame.grid(row=5, column=0, pady=15, sticky="ew")

        # Button for loading a video file
        self.load_video_button = ttk.Button(video_and_tracking_frame, text="Load Video File", command=self.load_video_file)
        self.load_video_button.pack(side="left", padx=5, ipadx=10, ipady=5, fill="x")

        # Tracking method selector (Combo box)
        self.tracking_method_menu = ttk.Combobox(video_and_tracking_frame, textvariable=self.tracking_method,
                                                  values=["SSD Tracker", "KCF Tracker", "MIL Tracker", "MOSSE Tracker"], state="readonly")
        self.tracking_method_menu.pack(side="right", padx=5, ipadx=10, ipady=5, fill="x")

        # Source and Exit buttons in one row
        change_exit_frame = ttk.Frame(button_frame)
        change_exit_frame.grid(row=6, column=0, pady=5, sticky="ew")

        self.change_source_button = ttk.Button(change_exit_frame, text="Change Source", command=self.change_source)
        self.change_source_button.pack(side="left", padx=5, ipadx=10, ipady=5, fill="x")

        self.quit_button = ttk.Button(change_exit_frame, text="EXIT", command=self.root.quit)
        self.quit_button.pack(side="right", padx=5, ipadx=10, ipady=5, fill="x")

        # Frame for displaying video (right) and set scaling
        self.video_frame = ttk.Label(main_frame, relief="sunken")
        self.video_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure weight distribution among frames
        main_frame.columnconfigure(0, weight=1)  # Left area (for buttons)
        main_frame.columnconfigure(1, weight=6)  # Right area (for video)
        main_frame.rowconfigure(0, weight=1)  # Scale down for the right area

        # Initialize variables for video
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.save_path = os.getcwd()

        # Load SSD model
        self.net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                        "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.bbox_colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

        # Initialize tracking objects
        self.tracker = None
        self.initialized_tracker = False  # Track if a tracker is initialized

    def start_video(self):
        if not self.is_running:
            if self.video_file_path:
                self.cap = cv2.VideoCapture(self.video_file_path)
            else:
                self.cap = cv2.VideoCapture(self.selected_camera_index)
            self.is_running = True
            self.frame_count = 0  # Reset frame count when starting
            self.start_time = cv2.getTickCount()  # Reset start time
            self.update_video()

    def stop_video(self):
        if self.is_running:
            self.is_running = False
            self.cap.release()
            if self.is_recording:
                self.out.release()
                self.is_recording = False
            self.initialized_tracker = False  # Reset tracker state

            # Calculate elapsed time and number of frames
            elapsed_ticks = (cv2.getTickCount() - self.start_time)  # number of ticks
            print(f"Tracking method: {self.tracking_method.get()}")
            print(f"Tracking duration (ticks): {elapsed_ticks}")

    def format_time(self):
        now = datetime.now()
        return f"{now.hour}h{now.minute}m{now.second}s"

    def take_screenshot(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                current_date = datetime.now().strftime("%Y-%m-%d")
                time_suffix = self.format_time()  # Get formatted time
                screenshot_file = os.path.join(self.save_path, f'screenshot_{current_date}_{time_suffix}.png')
                cv2.imwrite(screenshot_file, frame)
                print(f"Screenshot saved at: {screenshot_file}")

    def toggle_recording(self):
        if self.is_running:
            if not self.is_recording:
                current_date = datetime.now().strftime("%Y-%m-%d")
                time_suffix = self.format_time()
                video_file = os.path.join(self.save_path, f'record_{current_date}_{time_suffix}.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.out = cv2.VideoWriter(video_file, fourcc, 20.0, (640, 480))
                self.is_recording = True
                print(f"Recording started: {video_file}")
            else:
                self.out.release()
                self.is_recording = False
                print("Recording stopped.")

    def change_source(self):
        source = simpledialog.askinteger("Change Source", "Enter camera index (0, 1, 2...):")
        if source is not None:
            self.selected_camera_index = source  # Update the selected camera index
            print(f"Selected camera index: {self.selected_camera_index}")

    def change_save_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_path = folder
            print(f"Save path changed to: {self.save_path}")

    def load_video_file(self):
        video_file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if video_file:
            self.video_file_path = video_file
            print(f"Loaded video file: {self.video_file_path}")

    def detect_with_haar(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = self.haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Detection of cars
        cars = self.car_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_objects = []  # List to save all found objects

        # If a face is found
        for (x, y, w, h) in faces:
            detected_objects.append(("face", (x, y, w, h)))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Drawing a rectangle for the face

        # If cars are found
        for (x, y, w, h) in cars:
            detected_objects.append(("car", (x, y, w, h)))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Drawing a rectangle for the car

        return detected_objects  # Return the list of found objects

    def initialize_tracker(self, frame, bbox=None):
        if self.tracking_method.get() == "SSD Tracker":
            # Object detection via SSD
            detected_objects = self.detect_with_ssd(frame)
            if detected_objects:
                # Selecting found object
                obj_type, (x, y, w, h) = detected_objects[0]
                bbox = (x, y, w, h)
                # Passing the object to the CSRT method
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, bbox)
                self.initialized_tracker = True
                print(f"SSD Tracker is initialized with a bounding box: {bbox}")
        else:
            if bbox is None:
                bbox = self.detect_with_haar(frame)
                if bbox is None:
                    print("Failed to find object for tracking!")
                    return

            tracker_mapping = {
                "KCF Tracker": cv2.TrackerKCF_create,
                "MIL Tracker": cv2.TrackerMIL_create,
                "MOSSE Tracker": cv2.TrackerMOSSE_create,
            }
            tracker_func = tracker_mapping.get(self.tracking_method.get())
            if tracker_func:
                self.tracker = tracker_func()
                self.tracker.init(frame, bbox)
                self.initialized_tracker = True
                print(f"{self.tracking_method.get()} initialized with bounding box: {bbox}")

    def update_video(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        frame = cv2.resize(frame, (640, 480))
        self.frame_count += 1

        tracking_method_name = self.tracking_method.get()

        if tracking_method_name == "SSD Tracker":
            detected_objects = self.detect_with_ssd(frame)
            if detected_objects:
                for obj_type, (x, y, w, h) in detected_objects:
                    if self.initialized_tracker:
                        success, bbox = self.tracker.update(frame)
                        if success and len(bbox) == 4:
                            x, y, w, h = map(int, bbox)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Tracking failure", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                        2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            if self.initialized_tracker:
                success, bbox = self.tracker.update(frame)
                if success and len(bbox) == 4:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Tracking failure", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                bbox = self.detect_with_haar(frame)
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the type of tracking method
        cv2.putText(frame, f"Method: {tracking_method_name}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        self.video_frame.configure(image=photo)
        self.video_frame.image = photo

        # Record video if recording is active
        if self.is_recording:
            self.out.write(frame)

        self.root.after(20, self.update_video)

    def detect_with_ssd(self, frame):
        # Pre-process the frame for SSD model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        # Loop over detections and draw bounding boxes for objects with confidence > 0.2
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, x2, y2) = box.astype("int")
                color = self.bbox_colors[idx]
                label = f"{self.classes[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()