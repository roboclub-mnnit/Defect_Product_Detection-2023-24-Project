import os

os.chdir("D:\\VSCode\\TFODCourse")
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

CUSTOM_MODEL_NAME = "my_ssd_mobnet1"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
LABEL_MAP_NAME = "label_map.pbtxt"

paths = {
    "WORKSPACE_PATH": os.path.join("Tensorflow", "workspace"),
    "SCRIPTS_PATH": os.path.join("Tensorflow", "scripts"),
    "APIMODEL_PATH": os.path.join("Tensorflow", "models"),
    "ANNOTATION_PATH": os.path.join("Tensorflow", "workspace", "annotations"),
    "IMAGE_PATH": os.path.join("Tensorflow", "workspace", "images"),
    "MODEL_PATH": os.path.join("Tensorflow", "workspace", "models"),
    "PRETRAINED_MODEL_PATH": os.path.join(
        "Tensorflow", "workspace", "pre-trained-models"
    ),
    "CHECKPOINT_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME
    ),
    "OUTPUT_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "export"
    ),
    "TFJS_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "tfjsexport"
    ),
    "TFLITE_PATH": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "tfliteexport"
    ),
    "PROTOC_PATH": os.path.join("Tensorflow", "protoc"),
}

files = {
    "PIPELINE_CONFIG": os.path.join(
        "Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "pipeline.config"
    ),
    "TF_RECORD_SCRIPT": os.path.join(paths["SCRIPTS_PATH"], TF_RECORD_SCRIPT_NAME),
    "LABELMAP": os.path.join(paths["ANNOTATION_PATH"], LABEL_MAP_NAME),
}


labels = [
    {"name": "missing_hole", "id": 1},
    {"name": "mouse_bite", "id": 2},
    {"name": "open_circuit", "id": 3},
    {"name": "short", "id": 4},
    {"name": "spur", "id": 5},
    {"name": "spurious_copper", "id": 6},
]

with open(files["LABELMAP"], "w") as f:
    for label in labels:
        f.write("item { \n")
        f.write("\tname:'{}'\n".format(label["name"]))
        f.write("\tid:{}\n".format(label["id"]))
        f.write("}\n")

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files["PIPELINE_CONFIG"])
detection_model = model_builder.build(model_config=configs["model"], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths["CHECKPOINT_PATH"], "ckpt-3")).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter
import matplotlib

matplotlib.use("TkAgg")
# matplotlib inline

category_index = label_map_util.create_category_index_from_labelmap(files["LABELMAP"])

IMAGE_PATH = os.path.join(paths["IMAGE_PATH"], "test", "11_missing_hole_02.jpg")

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load your image
import argparse

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Path to the input image file")

# Parse the command-line arguments
args = parser.parse_args()

# Access the image path from the arguments
input_image_path = args.input

# Now you can use 'input_image_path' in your detection code

img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

# Perform object detection on the image
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop("num_detections"))
detections = {
    key: value[0, :num_detections].numpy() for key, value in detections.items()
}
detections["num_detections"] = num_detections
detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

# Create an image copy with annotated boxes and labels
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections["detection_boxes"],
    detections["detection_classes"] + 1,  # Shift class IDs to avoid class 0
    detections["detection_scores"],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,
    min_score_thresh=0.2,
    agnostic_mode=False,
)

# Extract detected class labels
detected_classes = detections["detection_classes"]

unique_labels = set()

# Loop through the detected classes and store unique labels
for detected_class in detected_classes:
    if detected_class > 0:
        label = category_index[detected_class]["name"]
        unique_labels.add(label)

# Display the image with detected labels on the side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display the image with annotations on the left side
ax1.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
ax1.set_title("Detected Objects")
ax1.axis("off")

# Create an image to display the unique detected labels on the right side
labels_image = np.zeros((image_np_with_detections.shape[0], 200, 3), dtype=np.uint8)
cv2.rectangle(
    labels_image, (0, 0), (200, image_np_with_detections.shape[0]), (255, 255, 255), -1
)

# Display the unique detected labels on the right side
# Display the unique detected labels centered both vertically and horizontally with a title
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1 # Increase the font size
font_color = (0, 0, 0)  # Black color
line_type = cv2.LINE_AA

# Calculate the total height
total_height = sum(
    cv2.getTextSize(label, font, font_scale, 2)[0][1] for label in unique_labels
)

# Create a white image
resized_image = 255 * np.ones(
    (total_height + 50, labels_image.shape[1], 3), dtype=np.uint8
)

# Calculate the starting y-position to center the labels vertically
y_pos = 40

# Title text
# title = "Detected Labels"
# title_size = cv2.getTextSize(title, font, font_scale, 2)
# title_pos = (
#     (resized_image.shape[1] - title_size[0][0]) // 2,
#     y_pos - 20,
# )  # Adjust spacing above the labels

for label in unique_labels:
    text_size = cv2.getTextSize(label, font, font_scale, 2)

    # Calculate the starting x-position to center the labels horizontally
    x_pos = (resized_image.shape[1] - text_size[0][0]) // 2

    # Draw the text
    cv2.putText(
        resized_image, label, (x_pos, y_pos), font, font_scale, font_color, 2, line_type
    )

    # Increment y_pos for the next label
    y_pos += text_size[0][1] + 10  # Adjust spacing between labels

# Draw the title text
# cv2.putText(resized_image, title, title_pos, font, font_scale, font_color, 2, line_type)

# Display the resized white image with the title and the unique detected labels centered
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Labels")
plt.axis("off")
plt.show()
