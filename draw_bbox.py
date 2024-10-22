import cv2
import numpy as np
import os

def draw_bounding_box(image, bbox_data, color, label):
    # Calculate the vertices of the bounding box
    x_center = bbox_data['obb']['x_center']
    y_center = bbox_data['obb']['y_center']
    width = bbox_data['obb']['width']
    height = bbox_data['obb']['height']
    rotation = bbox_data['obb']['rotation']

    # Get the box points based on the OBB
    box_points = cv2.boxPoints(((x_center, y_center), (width, height), np.degrees(rotation)))
    box_points = np.int32(box_points)  # Convert to integer points

    # Draw the bounding box
    cv2.polylines(image, [box_points], isClosed=True, color=color, thickness=2)

    # Put the label text
    cv2.putText(image, label, (box_points[0][0], box_points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_ocr_bounding_box(image, ocr_bbox, color, label):
    # Draw the OCR bounding box (axis-aligned rectangle)
    vertices = ocr_bbox['vertices']
    top_left = (vertices[0]['x'], vertices[0]['y'])
    bottom_right = (vertices[2]['x'], vertices[2]['y'])

    # Draw the rectangle
    cv2.rectangle(image, top_left, bottom_right, color, 2)

    # Put the OCR label text
    cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def plot_bboxes_on_image(image_path, bbox_list):
    # Read the input image
    image = cv2.imread(image_path)
    colors = {
        'ref': (0, 255, 0),  # Green for ref class
        'lot': (255, 0, 0),  # Red for lot class
        'expiry': (0, 0, 255)  # Blue for expiry class
    }

    # Iterate over the bounding boxes and draw them
    for bbox_data in bbox_list:
        if 'class' in bbox_data:  # It's a YOLO bounding box with OBB
            class_label = bbox_data['class']
            confidence = bbox_data['confidence']
            color = colors.get(class_label, (0, 255, 255))  # Default to yellow if class not found

            # Draw YOLO OBB bounding box
            draw_bounding_box(image, bbox_data, color, f"{class_label.upper()} ({confidence:.2f})")

        if 'text' in bbox_data:  # It's an OCR bounding box
            ocr_label = bbox_data.get('text', 'OCR')
            draw_ocr_bounding_box(image, bbox_data['bbox'], (0, 255, 255), ocr_label)  # Yellow for OCR box

    # Save the annotated image to a file
    output_path = "annotated_image_with_ocr.png"
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

# Define the path to your image
image_path = "test_sample/572cfab2691f11efb47542010a800003.png"  # Replace with your actual image path

# Updated list of bounding boxes with OBB and OCR data
bbox_list = [
    # First object (lot)
    {'class': 'expiry', 'confidence': 0.8913092613220215, 'obb': {'x_center': 928.5609130859375, 'y_center': 969.4102783203125, 'width': 39.262908935546875, 'height': 30.963415145874023, 'rotation': 1.5720528364181519}},
    
    # Second object (OCR text)
    {'text': '338949592027-04-24', 'bbox': {'vertices': [{'x': 1240, 'y': 909}, {'x': 1383, 'y': 909}, {'x': 1383, 'y': 981}, {'x': 1240, 'y': 981}]}, 'confidence': 0.9845772981643677, 'alphanumeric': True}
]

# Call the function to draw bounding boxes on the image
plot_bboxes_on_image(image_path, bbox_list)
