# Install required packages
# pip install roboflow supervision opencv-python

from roboflow import Roboflow
import supervision as sv
import cv2

# Initialize Roboflow with your API key
rf = Roboflow(api_key="y2BO50RFoRosd6vUrc8r")
project = rf.workspace().project("digitraffic")
model = project.version(1).model

# Open the video file
video_path = "test.mp4"  # Specify your video file path here
cap = cv2.VideoCapture(video_path)

# Check if video is opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Define the video writer to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving video (can change based on your preference)
out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Define desired resize dimensions (change these as needed)
resize_width = 640  # Width of the resized frame
resize_height = 480  # Height of the resized frame

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Predicting for the current frame
    result = model.predict(frame, confidence=40, overlap=30).json()
    
    # Create detections manually from the result
    detections = []
    for prediction in result["predictions"]:
        x, y = prediction["x"], prediction["y"]
        width, height = prediction["width"], prediction["height"]
        
        # Calculate x2, y2 (bottom-right corner of the bounding box)
        x2 = x + width
        y2 = y + height
        
        confidence = prediction["confidence"]
        class_id = prediction["class_id"]
        
        # Create Detection object (bounding box coordinates, confidence, class_id)
        detection = sv.Detection(x1=x, y1=y, x2=x2, y2=y2, confidence=confidence, class_id=class_id)
        detections.append(detection)

    # Annotate the frame with bounding boxes and labels
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Resize the frame to the desired size
    resized_frame = cv2.resize(annotated_frame, (resize_width, resize_height))

    # Display the annotated and resized frame
    cv2.imshow("Annotated Video", resized_frame)

    # Write the annotated frame to the output video file
    out.write(annotated_frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
