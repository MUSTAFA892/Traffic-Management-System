import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import time

# Load Faster R-CNN with ResNet-50 backbone and replace the classifier head
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Initialize the model with your custom number of classes
num_classes = 2  # Background + car

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes)
model.load_state_dict(torch.load("fasterrcnn_resnet50_epoch_4.pth"))
model.to(device)
model.eval()

# Prepare image tensor from an OpenCV frame
def prepare_image_opencv(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to PIL image
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
    return image_tensor.to(device)

# Function to get class name from class ID
COCO_CLASSES = {1: 'car'}

def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")

# Draw bounding boxes on the frame
def draw_boxes_on_frame(frame, prediction, threshold=0.5):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)
            
            # Draw bounding box on the frame
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({score:.2f})", (int(x_min), int(y_min)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Open video file
cap = cv2.VideoCapture("test.mp4")

# Define a time for FPS control (e.g., 30 FPS target)
fps_target = 30
frame_time = 1 / fps_target
prev_time = time.time()

frame_count = 0

# Loop through each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Control FPS by adding a delay
    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff < frame_time:
        continue  # Skip this frame if we are ahead of time
    prev_time = current_time

    # Resize the frame for faster processing (Optional)
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to reduce processing time

    # Prepare the frame for the model
    image_tensor = prepare_image_opencv(frame_resized)

    # Perform inference on the frame
    with torch.no_grad():
        prediction = model(image_tensor)

    # Draw bounding boxes on the frame
    draw_boxes_on_frame(frame_resized, prediction)

    # Display frame using OpenCV
    cv2.imshow("Video with Predictions", frame_resized)

    # Exit the video when the user presses the 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value of 'ESC'
        break

    frame_count += 1
    print(f"Processed frame {frame_count}")

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
