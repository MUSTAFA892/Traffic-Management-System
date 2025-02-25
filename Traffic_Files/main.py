from ultralytics import YOLO
import cv2

# Load the custom-trained YOLO model
model = YOLO('trained.pt')  # Provide the path to your .pt model

# Open the video
cap = cv2.VideoCapture('test.mp4')  # Provide the path to your video file
name = model.names
print(name)
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Output video setup (optional)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(20)
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Set the confidence threshold
conf_threshold = 0.3  # 80% confidence

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    results = model(frame)

    # Filter the detections based on the confidence threshold
    detections = results[0].boxes
    for detection in detections:
        conf = detection.conf[0].item()  # Get the confidence score
        if conf >= conf_threshold:
            # Get coordinates and label for detected object
            x1, y1, x2, y2 = map(int, detection.xyxy[0].cpu().numpy())  # Bounding box
            cls_id = int(detection.cls[0].item())  # Class ID
            label = model.names[cls_id]  # Access the label from the model's 'names' attribute
            # Draw bounding box and label
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow('Detection', frame)

    # Write the frame with bounding boxes to the output video
    output.write(frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()
