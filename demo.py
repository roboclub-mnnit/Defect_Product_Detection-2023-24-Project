import os
import fpdf
os.chdir('F:\\DPD\\Project')
import tensorflow as tf
import pandas as pd
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
import tkinter as tk
from tkinter import font
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import subprocess
from fpdf import FPDF

import tensorflow as tf
from object_detection.utils import (
    config_util,
    label_map_util,
    visualization_utils as viz_utils,
)
from object_detection.builders import model_builder
os.chdir('F:\\DPD\\Project')
import cv2
import numpy as np
from matplotlib import pyplot as plt


# create a window
window = tk.Tk()
# set the title of the window
window.title("Defect Product Detection")
# Set the icon bitmap for the window
window.iconbitmap("icon.ico")



# load the background image
bg_image = Image.open(r"background.png")
alpha = 0.30
# create a transparent version of the image
bg_image.putalpha(int(255 * alpha))
bg_image = bg_image.resize((1290, 720), Image.LANCZOS)
background_image = ImageTk.PhotoImage(bg_image)
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
# window.configure(bg="linear-gradient(to bottom, #87CEEB, #F0F0F0)")

# set the size of the window
window.geometry("1290x720")

image_count = 0

def handle_button_click(file_paths, button_num):
    global image_count
    image_count = 0
    detect_defects(file_paths, button_num)
    
def select_files(button_num):
    file_paths = filedialog.askopenfilenames(
        title=f"Select Image Files for Button {button_num}",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp *.ppm")],
        initialdir=os.path.expanduser("~"),
        multiple=True,
    )
    if file_paths:
        handle_button_click(file_paths, button_num)


label = tk.Label(
    window, text="DEFECT PRODUCT DETECTION", pady=10, padx=20, borderwidth=2
)
label.config(
    foreground="navy", background="white", font=("Roboto", 45), pady=10
)
label.pack()

# Add a label to the window
label = tk.Label(
    window,
    foreground="#0047AB",
    background="white",
    text="Click down the buttons to select the file",
)
label.config(font=("Helvetica", 24))
label.place(x=435, y=220)

# Create a label to display detected labels
detected_labels_label = tk.Label(
    window, foreground="white", background="black", text=""
)
detected_labels_label.config(font=("ariel", 20))
detected_labels_label.place(x=950, y=20)

# Create a custom font with an increased size
custom_font = ("Arial", 14)

# Add a button to select and detect defects in an image file
button1 = ttk.Button(master=window, text="Select Batch1 images", command=lambda: select_files(1), style="Custom.TButton")
button1.place(x=200, y=320, width=220, height=50)  # Adjust width and height for the left side

button2 = ttk.Button(master=window, text="Select Batch2 images", command=lambda: select_files(2), style="Custom.TButton")
button2.place(x=900, y=320, width=220, height=50)  # Adjust width and height for the right side

# Configure the style with the custom font
style = ttk.Style()
style.configure("Custom.TButton", font=custom_font ,foreground="#3F00FF", label_background="#7DF9FF" )


message_label1 = tk.Label(
    window, foreground="#6F8FAF", background="white", text="", font=("Helvetica", 12)
)
message_label1.place(x=200, y=380)
message_label2 = tk.Label(
    window, foreground="#6F8FAF", background="white", text="", font=("Helvetica", 12)
)
message_label2.place(x=900, y=380)
message_label3 = tk.Label(
    window, foreground="#6F8FAF", background="white", text="", font=("Helvetica", 12)
)
message_label3.place(x=500, y=600)

total_labels_count1 = {}
total_labels_count2 = {}

detect_defects_counter = 1

def detect_defects(file_paths,button_num):
    global image_count
    global detect_defects_counter  
    if file_paths:

        CUSTOM_MODEL_NAME = "my_ssd_mobnet_tuned"
        # ... (rest of your detect_defects function remains unchanged)
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
                "Tensorflow",
                "workspace",
                "models",
                CUSTOM_MODEL_NAME,
                "pipeline.config",
            ),
            "TF_RECORD_SCRIPT": os.path.join(
                paths["SCRIPTS_PATH"], TF_RECORD_SCRIPT_NAME
            ),
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
        detection_model = model_builder.build(
            model_config=configs["model"], is_training=False
        )

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths["CHECKPOINT_PATH"], "ckpt-11")).expect_partial()

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

        category_index = label_map_util.create_category_index_from_labelmap(
            files["LABELMAP"]
        )

        # IMAGE_PATH = file_path_label

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
        for IMAGE_PATH in file_paths:
            img = cv2.imread(IMAGE_PATH)
            image_np = np.array(img)

            # Perform object detection on the image
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32
            )
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop("num_detections"))
            detections = {
                key: value[0, :num_detections].numpy()
                for key, value in detections.items()
            }
            detections["num_detections"] = num_detections
            detections["detection_classes"] = detections["detection_classes"].astype(
                np.int64
            )

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

                    # Increment the count for the label in the outer dictionary
                    if(button_num == 1):
                        total_labels_count1[label] = total_labels_count1.get(label, 0) + 1
                    if(button_num == 2):
                        total_labels_count2[label] = total_labels_count2.get(label, 0) + 1


            # Display the image with detected labels on the side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Display the image with annotations on the left side
            ax1.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
            ax1.set_title("Detected Objects")
            ax1.axis("off")

            # Create an image to display the unique detected labels on the right side
            labels_image = np.zeros(
                (image_np_with_detections.shape[0], 200, 3), dtype=np.uint8
            )
            cv2.rectangle(
                labels_image,
                (0, 0),
                (200, image_np_with_detections.shape[0]),
                (255, 255, 255),
                -1,
            )

            # Display the unique detected labels on the right side
            # Display the unique detected labels centered both vertically and horizontally with a title
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7  # Increase the font size
            font_color = (0, 0, 0)  # Black color
            line_type = cv2.LINE_AA

            # Calculate the total height
            total_height = sum(
                cv2.getTextSize(label, font, font_scale, 2)[0][1]
                for label in unique_labels
            )

            # Create a white image
            resized_image = 255 * np.ones(
                (total_height + 100, labels_image.shape[1], 3), dtype=np.uint8
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
                    resized_image,
                    label,
                    (x_pos, y_pos),
                    font,
                    font_scale,
                    font_color,
                    2,
                    line_type,
                )

                # Increment y_pos for the next label
                y_pos += text_size[0][1] + 10  # Adjust spacing between labels

            # Draw the title text
            # cv2.putText(resized_image, title, title_pos, font, font_scale, font_color, 2, line_type)

            # Display the resized white image with the title and the unique detected labels centered
            
            plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            plt.title("Detected Labels")
            plt.axis("off")
            # plt.show()
            # Run the main event loop

            # Save the figure
            if (button_num == 1):
                save_path = os.path.join("F:\\DPD\\Project\\gui\\Button_1", f"detected_image{detect_defects_counter}.jpg")
                plt.savefig(save_path)
                message_label1.config(text="Batch 1 images processed successfully!")
                
                
            if(button_num == 2):
                save_path = os.path.join("F:\\DPD\\Project\\gui\\Button_2", f"detected_image{detect_defects_counter}.jpg")
                plt.savefig(save_path)
                message_label2.config(text="Batch 2 images processed successfully!")
                # window.after(5000, message_label3, message_label2)
                

            detect_defects_counter += 1

            # Close the plot to avoid displaying it
            # message_label3.config(text="Your report file is Ready !! Close the application")
            plt.close()
            image_count += 1
            
            # detected_labels_counts[f"Image {image_count}"] = len(unique_labels)

            # detected_labels_label.config(
            # text="\n".join([f"Detected Labels for Image {image_count}: {count}" for image, count in detected_labels_counts.items()])
            # )
            # detected_labels_label.place(x=320, y=320)


    

while True:
    try:
        window.update()
    except tk.TclError:
        break  # Break the loop if the window is closed
    
print("Total Count of Each Label1:")
for label, count in total_labels_count1.items():
    print(f"{label}: {count}")
print("Total Count of Each Label2:")
for label, count in total_labels_count2.items():
    print(f"{label}: {count}")

df1 = pd.DataFrame(list(total_labels_count1.items()), columns=['Label', 'Count'])
df2 = pd.DataFrame(list(total_labels_count2.items()), columns=['Label', 'Count'])

# Write DataFrames to Excel
excel_file1 = 'F:\\DPD\\Project\\gui\\Labels_Result_Button1.xlsx'
df1.to_excel(excel_file1, index=False)
print(f"Excel file for Button 1 created: {excel_file1}")

excel_file2 = 'F:\\DPD\\Project\\gui\\Labels_Result_Button2.xlsx'
df2.to_excel(excel_file2, index=False)
print(f"Excel file for Button 2 created: {excel_file2}")

# script_path = 'D:\\VSCode\\TFODCourse\\GUI\\pdf.py'

# # Run the other script using subprocess
# subprocess.run(['python', script_path]

# Read data from Excel files
df1 = pd.read_excel("F:\\DPD\\Project\\gui\\Labels_Result_Button1.xlsx")
df2 = pd.read_excel("F:\\DPD\\Project\\gui\\Labels_Result_Button2.xlsx")

# Path to folders containing images for Dataset 1 and Dataset 2
dataset1_folder = "F:\\DPD\\Project\\gui\\Button_1"
dataset2_folder = "F:\\DPD\\Project\\gui\\Button_2"

# Create a PDF document
pdf = FPDF()
pdf.add_page()

# Add title
pdf.set_font("Arial", style="B", size=16)
pdf.cell(200, 10, txt="Defect Product Report", ln=True, align="C")

# Add subtitle
pdf.set_font("Arial", style="I", size=12)
pdf.cell(200, 10, txt="Batch 1 and Batch 2", ln=True, align="C")
pdf.ln(10)  # Add a line break

# Plot histograms side by side and add to PDF
plt.figure(figsize=(12, 6))

# Plot histogram for Dataset 1
plt.subplot(1, 2, 1)
bars1 = plt.bar(df1["Label"], df1["Count"])
plt.title("Batch 1 - Label Histogram")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Annotate bars with label counts
for bar in bars1:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.05,
        round(yval, 2),
        ha="center",
        va="bottom",
    )

# Plot histogram for Dataset 2
plt.subplot(1, 2, 2)
bars2 = plt.bar(df2["Label"], df2["Count"])
plt.title("Batch 2 - Label Histogram")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Annotate bars with label counts
for bar in bars2:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.05,
        round(yval, 2),
        ha="center",
        va="bottom",
    )

# Save the combined histogram plot as an image
plt.savefig("temp_combined_hist.png")
plt.close()

# Add combined histogram image to PDF
pdf.image("temp_combined_hist.png", x=10, y=None, w=190)
pdf.ln(10)  # Add a line break

# Add label counts to PDF
# pdf.set_font("Arial", style='B', size=14)
# pdf.cell(200, 10, txt="Label Counts", ln=True)
# pdf.ln(10)

# # # Add label counts as a table
# pdf.set_font("Arial", size=12)
# col_widths = [pdf.get_string_width("Label") + 20, pdf.get_string_width("Count") + 20]
# pdf.set_fill_color(200, 220, 255)  # Set background color for the table header
# pdf.cell(col_widths[0], 10, "Label", border=1, fill=True)
# pdf.cell(col_widths[1], 10, "Count", border=1, fill=True)
# pdf.ln()

# # # Add data to the table for Dataset 1
# for _, row in df1.iterrows():
#     pdf.cell(col_widths[0], 10, str(row['Label']), border=1)
#     pdf.cell(col_widths[1], 10, str(row['Count']), border=1)
#     pdf.ln()

# # # Add data to the table for Dataset 2
# for _, row in df2.iterrows():
#     pdf.cell(col_widths[0], 10, str(row['Label']), border=1)
#     pdf.cell(col_widths[1], 10, str(row['Count']), border=1)
#     pdf.ln()

# Add images to the PDF
pdf.ln(10)  # Add some space between table and images


# Add images for Dataset 1
pdf.set_font("Arial", style="B", size=14)
pdf.cell(200, 10, txt="Dataset 1 - Sample Images", ln=True)
pdf.ln(2)

# Concatenate images horizontally for Dataset 1
for i in range(0, len(os.listdir(dataset1_folder)), 3):
    images_folder1 = os.listdir(dataset1_folder)[i:i+3]
    concatenated_image_path1 = f'temp_images_dataset_1_{i}.png'
    images1 = [cv2.imread(os.path.join(dataset1_folder, img)) for img in images_folder1]
    concatenated_image1 = cv2.hconcat(images1)
    cv2.imwrite(concatenated_image_path1, concatenated_image1)
    pdf.image(concatenated_image_path1, x=10, y=None, w=200)
    pdf.ln(6)

# Add images for Dataset 2
pdf.set_font("Arial", style="B", size=14)
pdf.cell(200, 10, txt="Dataset 2 - Sample Images", ln=True)
pdf.ln(2)

# Concatenate images horizontally for Dataset 2
for i in range(0, len(os.listdir(dataset2_folder)), 3):
    images_folder2 = os.listdir(dataset2_folder)[i:i+3]
    concatenated_image_path2 = f'temp_images_dataset_2_{i}.png'
    images2 = [cv2.imread(os.path.join(dataset2_folder, img)) for img in images_folder2]
    concatenated_image2 = cv2.hconcat(images2)
    cv2.imwrite(concatenated_image_path2, concatenated_image2)
    pdf.image(concatenated_image_path2, x=10, y=None, w=200)
    pdf.ln(6)

# Save the PDF
pdf.output("label_histograms_and_counts_with_images_horizontal.pdf")

# Remove temporary files
for i in range(0, len(os.listdir(dataset1_folder)), 3):
    os.remove(f'temp_images_dataset_1_{i}.png')
for i in range(0, len(os.listdir(dataset2_folder)), 3):
    os.remove(f'temp_images_dataset_2_{i}.png')