import cv2
import os
import time
import requests
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np  # Added numpy import

#urls for model and configuration
face_detection_model_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
face_detection_config_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

#local file paths
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/facial_rec')
model_dir = os.path.join(base_dir, 'models')
config_dir = os.path.join(base_dir, 'config_files')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

face_detection_model_file = os.path.join(model_dir, 'deploy.prototxt')
face_detection_config_file = os.path.join(config_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

#function to download model files if they don't exist
def download_file(url, file_name):
    if not os.path.exists(file_name):
        print(f"downloading {file_name}...")
        response = requests.get(url)
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} downloaded successfully.")
    else:
        print(f"{file_name} already exists, skipping download.")

#download model and config files
download_file(face_detection_model_url, face_detection_model_file)
download_file(face_detection_config_url, face_detection_config_file)

#load the dnn model for face detection
net = cv2.dnn.readNetFromCaffe(face_detection_model_file, face_detection_config_file)

#initialize camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("can't open camera")
    exit()

#set camera resolution
desired_width = 1920
desired_height = 1080
camera.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

#tkinter window setup
root = tk.Tk()
root.title("Face Detection")
root.geometry(f"{desired_width}x{desired_height+50}")

#label to display the video
label = tk.Label(root)
label.pack()

captured_frame = None

#function to capture a frame and store it
def capture_frame():
    global captured_frame
    ret, frame = camera.read()
    if ret:
        captured_frame = frame
        print("frame captured for face detection.")
        root.destroy()
        show_detected_faces()

#button to capture frame
capture_button = tk.Button(root, text="Detect Faces", command=capture_frame)
capture_button.pack(side=tk.BOTTOM)

#function to perform face detection and show the result in a new window
def show_detected_faces():
    #convert the captured frame to a blob for SSD model
    h, w = captured_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(captured_frame, 1.0, (300, 300), [104, 117, 123], False, False)
    
    #set the input for the network
    net.setInput(blob)
    detections = net.forward()

    #loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        #filter out weak detections
        if confidence > 0.5:
            #get the coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #draw the bounding box and confidence on the frame
            text = f"{confidence * 100:.2f}%"
            cv2.rectangle(captured_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(captured_frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    #create a new tkinter window to display the image with detected faces
    result_window = tk.Tk()
    result_window.title("Detected Faces")

    #convert the captured frame with detections to an Image and display it
    frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    result_label = tk.Label(result_window, image=imgtk)
    result_label.imgtk = imgtk
    result_label.pack()

    result_window.mainloop()

#function to update the video feed in the tkinter window
def update_frame():
    ret, frame = camera.read()
    if ret:
        #convert frame to RGB (tkinter uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        #update the label with the new frame
        label.imgtk = imgtk
        label.configure(image=imgtk)

    #call this function again after a short delay
    label.after(10, update_frame)

#start updating the video feed
update_frame()

#start the tkinter event loop
root.mainloop()

#release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
