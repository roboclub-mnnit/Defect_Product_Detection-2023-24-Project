import os

os.chdir("D:\\VSCode\\TFODCourse")
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import cv2

# Read data from Excel files
df1 = pd.read_excel("D:\\VSCode\\TFODCourse\\GUI\\Labels_Result_Button1.xlsx")
df2 = pd.read_excel("D:\\VSCode\\TFODCourse\\GUI\\Labels_Result_Button2.xlsx")

# Path to folders containing images for Dataset 1 and Dataset 2
dataset1_folder = "D:\\VSCode\\TFODCourse\\GUI\\Images_Result\\Button_1"
dataset2_folder = "D:\\VSCode\\TFODCourse\\GUI\\Images_Result\\Button_2"

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