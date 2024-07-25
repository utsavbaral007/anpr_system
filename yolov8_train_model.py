from ultralytics import YOLO

# Path to your dataset and data.yaml file
data_yaml = "../anpr_datset/data.yaml"
train_image_folder = "../anpr_datset/train/images"
train_label_folder = "../anpr_datset/train/labels"
val_image_folder = "../anpr_datset/valid/images"
val_label_folder = "../anpr_datset/valid/labels"
test_image_folder = "../anpr_datset/test/images"
test_label_folder = "../anpr_datset/test/labels"

# load a model
model = YOLO("yolov8s.pt")

# Train the model
results = model.train(data="anpr_dataset/data.yaml", epochs=50, imgsz=640, batch=32) 

# Print test results
print("Test Results:")
print(results)


