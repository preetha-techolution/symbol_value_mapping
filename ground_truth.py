import os
import pandas as pd
import io
import json
import re
from PIL import Image
from google.cloud import vision
import tempfile

# Import the cloud_vision_inference function from your google_ocr.py file
from google_ocr import cloud_vision_inference

# Function to crop the image based on vertices
def crop_image(image_path, vertices):
    try:
        image = Image.open(image_path)
        # Convert vertices into the required format for cropping
        x_min = min([v['x'] for v in vertices])
        y_min = min([v['y'] for v in vertices])
        x_max = max([v['x'] for v in vertices])
        y_max = max([v['y'] for v in vertices])
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        return cropped_image
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

# Function to run OCR using the Google Cloud Vision inference function
def run_ocr(cropped_image):
    # Create a temporary file to save the cropped image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        cropped_image.save(temp_file, format='PNG')
        temp_image_path = temp_file.name  # Get the temporary file path

    # Call the cloud_vision_inference function with the image path
    ocr_results = cloud_vision_inference(temp_image_path)

    # Remove the temporary file after using it
    os.remove(temp_image_path)

    # Extract the first text result (if available)
    if ocr_results and ocr_results[0].get("text"):
        return ocr_results[0]["text"].strip()
    return ""

# Preprocessing function to remove unwanted text
def remove_unwanted_text(type, result):
    result = re.sub(r'\[lt\]', '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()
    if type == 'lot_no':
        result = result.lower()
        result = result.replace('lot', '')
        result = result.replace('no', '')
        result = result.replace(':', '')
        result = result.replace('.', '')
        result = result.replace('number', '')
        result = result.replace('catalogue', '')
        result = result.replace('catalog', '')
        result = result.replace('batch', '')
        result = result.replace('code', '')
        result = result.replace(':', " ")
        result = result.replace('(', '')
        result = result.replace(')', '')
        result = result.replace('[lt]', '')
        result = result.replace('[', '')
        result = result.replace(']', '')
        result = result.replace('la', '')
        result = result.replace('lo', '')
        result = result.replace('le', '')
        result = result.replace('t', '')
        result = result.replace('(10)', '')
        result = result.replace('$', '')
        result = result.replace('/','')
        result = result.upper()

    elif type == 'ref_no':
        result = result.lower()
        result = result.replace('ref', '')
        result = result.replace('reference', '')
        result = result.replace('batch', '')
        result = result.replace('code', '')
        result = result.replace('erc', '')
        result = result.replace('each', '')
        result = result.replace(':', '')
        result = result.replace('.', '')
        result = result.replace('/','')
        result = result.replace('no', '')
        result = result.replace('number', '')
        result = result.replace('ref', '')
        result = result.replace('rep', '')
        result = result.replace('(10)', '')
        result = result.replace('$', '')
        result = result.replace('(', '')
        result = result.replace(')', '')
        result = result.replace('catalogue', '')
        result = result.replace('catalog', '')
        result = result.replace(':', " ")
        result = result.replace('ret', '')
        result = result.replace('ree', '')
        result = result.replace('leb', '')
        result = result.replace('[', '')
        result = result.replace(']', '')
        result = result.replace('rf', '')
        result = result.replace('lea','')
        result = result.replace('order','')
        result = result.replace('order/','')
        result = result.upper()

    elif type == 'use_by':
        #result = re.sub(r'(?i)useby', '', result).strip()
        result = result.lower()
        result = result.replace('lot', '')
        result = result.replace('ref', '')
        result = result.replace(':', '')
        result = result.replace('number', '')
        result = result.replace('exp', '')
        result = result.replace('use-by', '')
        result = result.replace('$', '')
        result = result.replace('(', '')
        result = result.replace(')', '')
        result = result.replace('useby','')
        result = result.replace('useby:','')
        result = result.replace('usebydate','')
        result = result.replace('.', '')
        result = result.replace('date', '')
        result = result.replace('exp.date','')
        result = result.replace('expdate','')
        result = result.upper()

    return result

# Read the input CSV file
df = pd.read_csv('dataSetCollection_angle_corrected_data_resources.csv')

# Initialize a list to store the extracted OCR results
ocr_results = []

# Define the folder path where the images are stored
image_folder_path = 'angle_corrected_images'  # Update with your folder path

# List all image files in the specified folder
image_filenames = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# Iterate over each image file in the folder
for image_filename in image_filenames:
    # Strip any whitespace from the image filename
    image_filename = image_filename.strip()

    # Search for the corresponding row in the CSV based on the image filename in the 'name' column
    matching_row = df[df['name'] == image_filename]  # Search in the 'name' column

    # If a matching row is found, process the annotations
    if not matching_row.empty:
        row = matching_row.iloc[0]  # Get the first matching row
        image_annotations = json.loads(row['imageAnnotations'])

        # For each annotation, check if it's for 'ref_no', 'lot_no', or 'use_by'
        ocr_data = {"image_id": row['_id'], "image_name": image_filename}  # Add image_name to ocr_data
        for annotation in image_annotations:
            for option in annotation['selectedOptions']:
                if option['value'] in ['ref_no', 'lot_no', 'use_by']:
                    vertices = annotation.get('vertices')
                    if vertices:  # Ensure vertices are available
                        # Crop the image based on the bounding box
                        cropped_img = crop_image(os.path.join(image_folder_path, image_filename), vertices)
                        if cropped_img:  # Check if cropped image is not None
                            # Run OCR on the cropped image
                            ocr_text = run_ocr(cropped_img)
                            # Preprocess the OCR text before saving
                            preprocessed_text = remove_unwanted_text(option['value'], ocr_text)
                            # Save the preprocessed OCR result under the respective class
                            ocr_data[option['value']] = preprocessed_text
        
        # Append the OCR data for this image to the results
        print(ocr_data)
        ocr_results.append(ocr_data)
    else:
        print(f"No matching entry found in CSV for image: {image_filename}")

# Convert OCR results to DataFrame and save as CSV
ocr_df = pd.DataFrame(ocr_results)
ocr_df.to_csv('ocr_results.csv', index=False)

print("OCR process completed and results saved.")
