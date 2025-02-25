

# Traffic Management System

This project aims to leverage deep learning techniques for traffic management by detecting vehicles and analyzing traffic patterns. The system is designed to work with the COCO-style dataset, pre-trained YOLO models, and various testing scripts for video analysis.

## Project Structure

### 1. **Datasets**

#### `coco_json/`
This folder contains the COCO-style annotations for the traffic dataset. It includes:

- **train.json**: The training set annotations in COCO JSON format.
- **valid.json**: The validation set annotations in COCO JSON format.

These files contain detailed information about the traffic scenes, including object classes (e.g., cars, trucks, buses) and bounding box coordinates.

### 2. **Testing Videos**

#### `Traffic_files/`
This directory contains video files that will be used for testing vehicle detection. The video files simulate real-world traffic scenarios, with vehicles moving through various environments. You can run the testing scripts to analyze these videos for vehicle detection.

### 3. **YOLO Pretrained Models**

#### `yolo_weights/`
This folder contains the pre-trained YOLO model weights that are used for object detection. These weights have been trained on large datasets and can be used directly to detect vehicles in testing videos. The models are available in various versions (e.g., YOLOv3, YOLOv4, etc.).

### 4. **Notebooks**

#### `notebooks/`
This folder contains Jupyter Notebooks that can be used for training, testing, and visualizing the results. The notebooks are organized into the following categories:

- **Training Notebook**: For training YOLO on the traffic dataset. It loads the COCO JSON annotations, prepares the dataset, and trains the model.
- **Testing Notebook**: For testing the trained YOLO models on the testing video files. The notebook allows you to evaluate the model performance by displaying bounding boxes over detected vehicles.
- **Visualization Notebook**: For visualizing model predictions on images and videos. It displays results of object detection using the YOLO model.

### 5. **Testing Scripts**

#### `scripts/`
This directory contains various Python scripts that automate the testing process on video files. The scripts utilize pre-trained YOLO models to detect vehicles in the videos and save results in a convenient format (such as annotated frames or detection logs).

- **test_video.py**: A script to run vehicle detection on videos stored in the `Traffic_files/` folder. It outputs annotated video files and logs the detection results.
- **evaluate_model.py**: This script is used to evaluate the performance of a YOLO model on the validation dataset or a custom test set. It computes metrics such as precision, recall, and mAP (mean Average Precision).

---

## Setup Instructions

### 1. **Clone the Repository**
First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/traffic-management.git
cd traffic-management
```

### 2. **Install Dependencies**
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. **Download YOLO Pre-trained Weights**
Download the YOLO pre-trained model weights and place them in the `yolo_weights/` folder.

You can use the official YOLO weights:
- YOLOv3: [Download Link]
- YOLOv4: [Download Link]

### 4. **Dataset Preparation**
Ensure that the `coco_json/` folder contains the `train.json` and `valid.json` files, which can be used for training and validation.

### 5. **Training the Model**
To train the YOLO model on your dataset, run the training notebook located in the `notebooks/` folder. Alternatively, you can use the following command in your terminal to initiate the training process using the provided script:

```bash
python scripts/train_model.py
```

### 6. **Testing the Model**
Once the model is trained or if you want to use pre-trained weights, you can test the model on the traffic video files. Run the following command:

```bash
python scripts/test_video.py --video_path <path_to_video_file> --weights <path_to_yolo_weights>
```

You can also use the `notebooks/testing_notebook.ipynb` to manually test the model on videos.

### 7. **Evaluation**
To evaluate your model's performance on the validation dataset or another test set, run:

```bash
python scripts/evaluate_model.py --weights <path_to_yolo_weights> --dataset <path_to_validation_dataset>
```

This will output performance metrics like precision, recall, and mAP.

---

## Usage

- **Training**: Train a custom YOLO model on the traffic dataset using the training scripts and notebooks. You can fine-tune the model for specific types of vehicles and traffic conditions.
- **Testing**: Use the pre-trained YOLO models to detect vehicles in traffic videos. The `test_video.py` script provides an easy way to apply the model to video files.
- **Visualization**: View the results of the vehicle detection in the Jupyter notebooks or by inspecting the annotated video files generated during testing.

---

## Dependencies

- Python 3.x
- TensorFlow, PyTorch (depending on YOLO implementation)
- OpenCV for video processing
- Other dependencies are listed in `requirements.txt`

---

## Notes

- Make sure you have a GPU enabled if you want to speed up the training and testing process, as YOLO models are computationally intensive.
- The dataset and models are designed to be modular, allowing you to swap different YOLO versions and test on various video files.
  
