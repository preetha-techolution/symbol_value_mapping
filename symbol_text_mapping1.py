import math
import cv2
import numpy as np
import os
from symbol_inf import obb_results  # Import YOLO bounding boxes from another file
from google_ocr import ocr_results
from google_ocr import cloud_vision_inference

# Create a directory to save cropped images if it doesn't exist
output_dir = 'cropped_images'
os.makedirs(output_dir, exist_ok=True)

# Function to check if a point is inside a bounding box
def is_point_inside_bbox(x_center, y_center, ocr_bbox):
    x_min = min(ocr_bbox['vertices'][0]['x'], ocr_bbox['vertices'][3]['x'])
    x_max = max(ocr_bbox['vertices'][1]['x'], ocr_bbox['vertices'][2]['x'])
    y_min = min(ocr_bbox['vertices'][0]['y'], ocr_bbox['vertices'][1]['y'])
    y_max = max(ocr_bbox['vertices'][2]['y'], ocr_bbox['vertices'][3]['y'])
    
    return x_min <= x_center <= x_max and y_min <= y_center <= y_max

# Function to extend the OBB bounding box to the right within the OCR bounding box
def extend_bbox_to_right(obb, ocr_bbox):
    obb_x_center = obb['obb']['x_center']
    obb_width = obb['obb']['width']
    
    obb_left = obb_x_center - obb_width / 2  # Calculate the left x-coordinate of the OBB
    obb_right = obb_x_center + obb_width / 2  # Calculate the right x-coordinate of the OBB

    # Determine the new right edge based on the OCR bounding box
    x_max_ocr = max(ocr_bbox['vertices'][1]['x'], ocr_bbox['vertices'][2]['x'])

    # Start the new bounding box from the right x-vertex of the existing OBB
    new_left = obb_right  # New left x-coordinate is the current right edge of the OBB

    # The new right edge is determined by the maximum x-coordinate of the OCR bounding box
    obb_right = x_max_ocr  # Extend the right x-coordinate to match OCR bbox

    # Keep the height the same; calculate new height from the original OBB
    obb_height = obb['obb']['height']  # Original height
    new_height = obb_height

    # Update the OBB width and center based on the new right edge
    new_width = obb_right - new_left  # Calculate the new width
    obb['obb']['width'] = new_width
    obb['obb']['height'] = new_height  # Keep the height same as original
    obb['obb']['x_center'] = new_left + new_width / 2  # Recalculate the center

    return obb

# Function to crop the image based on the OBB
def crop_image(image, obb):
    # Get bounding box coordinates
    x_center = obb['obb']['x_center']
    y_center = obb['obb']['y_center']
    width = obb['obb']['width']
    height = obb['obb']['height']

    # Calculate the cropping box coordinates
    x_min = int(x_center - width / 2)
    x_max = int(x_center + width / 2)
    y_min = int(y_center - height / 2)
    y_max = int(y_center + height / 2)

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

# Load the image where the OBBs are located
# Replace 'image_path.jpg' with the actual path to your image
image_path = 'test.png'
image = cv2.imread(image_path)

# Lists to store updated OBB detections
obb_inside_ocr = []
ref_processed = False
lot_processed = False
# Iterate over YOLO OBB results
for obb in obb_results:
    obb_center_x = obb['obb']['x_center']
    obb_center_y = obb['obb']['y_center']
    
    # Check if the center of the OBB lies within any OCR bounding box
    for ocr in ocr_results:
        if is_point_inside_bbox(obb_center_x, obb_center_y, ocr['bbox']):
            # If the class is 'ref' or 'lot', extend the bbox to the right
            if obb['class'] in ['ref', 'lot']:
                if obb['class'] == 'ref' and not ref_processed:
                    ref_processed = True
                    obb = extend_bbox_to_right(obb, ocr['bbox'])
                    
                    # Check if the obb_right is equal to the new_left after extension
                    obb_right = obb_center_x + (obb['obb']['width'] / 2)
                    new_left = obb_right  # new_left is the right edge of the original OBB
                    
                    if math.isclose(obb_right, new_left, abs_tol=1e-3):
                        print(f"Valid extended OBB for class {obb['class']}: {obb}")
                        obb_inside_ocr.append(obb)  # Add only if the condition is met

                        # Crop the image based on the updated OBB
                        cropped_image = crop_image(image, obb)
                        
                        
                        # Save the cropped image
                        save_path = os.path.join(output_dir, f"{obb['class']}_obb.jpg")
                        cv2.imwrite(save_path, cropped_image)

                        ocr_result = cloud_vision_inference(save_path)
                        print(f"OCR result for {obb['class']}: {ocr_result}")

                if obb['class'] == 'lot' and not lot_processed:
                    lot_processed = True
                    obb = extend_bbox_to_right(obb, ocr['bbox'])
                    
                    # Check if the obb_right is equal to the new_left after extension
                    obb_right = obb_center_x + (obb['obb']['width'] / 2)
                    new_left = obb_right  # new_left is the right edge of the original OBB
                    
                    if math.isclose(obb_right, new_left, abs_tol=1e-3):
                        print(f"Valid extended OBB for class {obb['class']}: {obb}")
                        obb_inside_ocr.append(obb)  # Add only if the condition is met

                        # Crop the image based on the updated OBB
                        cropped_image = crop_image(image, obb)
                        
                        
                        # Save the cropped image
                        save_path = os.path.join(output_dir, f"{obb['class']}_obb.jpg")
                        cv2.imwrite(save_path, cropped_image)

                        ocr_result = cloud_vision_inference(save_path)
                        print(f"OCR result for {obb['class']}: {ocr_result}")


            break  # Break after processing the first matching OCR bbox



# Print the results
#print("Updated OBB inside OCR bounding boxes:")
#print(obb_inside_ocr)

