!pip install opencv-python
!pip install webrtc_streamer
import streamlit as st
import numpy as np
from webrtc_streamer import webrtc_streamer

# Load the pre-trained model
model = np.load("yolov3.weights")

# Create a webrtc streamer object
streamer = webrtc_streamer()

# Create a text box to display the class labels
st.text("Class labels:")

# Create a list to store the class labels
class_labels = [
    "person",
    "car",
    "dog",
    "cat",
    "bike",
    "bus",
    "truck",
    "motorcycle",
    "airplane",
    "train",
]

# Start the webrtc streamer
streamer.start()

# Loop over the frames in the video
while True:
    # Get the next frame from the webrtc streamer
    frame = streamer.get_frame()

    # Convert the frame to a NumPy array
    frame_np = np.array(frame)

    # Detect objects in the frame
    detections = model.detectMultiScale(frame_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the detected objects
    for detection in detections:
        # Get the bounding box coordinates
        x, y, w, h = detection[0:4]

        # Get the confidence score
        confidence = detection[4]

        # If the confidence score is greater than a threshold, draw the bounding box and label
        if confidence > 0.5:
            # Draw the bounding box
            cv2.rectangle(frame_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Get the class label
            class_id = detection[5]

            # Get the class name
            class_name = class_labels[class_id]

            # Draw the class label
            cv2.putText(frame_np, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    st.image(frame_np)

    # Check if the user wants to quit
    if st.button("Quit"):
        break

# Stop the webrtc streamer
streamer.stop()
