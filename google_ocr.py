import io
import os
import re
from google.cloud import vision

# Set the path to your Google Cloud service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "firebase_keys/vision_key.json"

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

def load_image(path):
    """
    Load an image from the specified path.

    Parameters:
        path (str): Path to the image file.

    Returns:
        bytes: The image content in bytes.
    """
    with io.open(path, 'rb') as image_file:
        return image_file.read()

def is_strictly_alphanumeric(text):
    """
    Check if the text contains both alphabetic and numeric characters,
    ignoring special characters like *, -, etc.

    Parameters:
        text (str): The text to check.

    Returns:
        bool: True if the text contains both letters and digits, False otherwise.
    """
    # Remove special characters from the string
    cleaned_text = text.replace('-', '').replace('*', '')
    #print(cleaned_text)
    
    # Check if it contains both alphabetic and numeric characters
    has_alpha = any(char.isalpha() for char in cleaned_text)
    has_digit = any(char.isdigit() for char in cleaned_text)
    
    return has_digit or (has_digit and has_alpha)

def perform_text_detection(client, image_content):
    """
    Perform document text detection on the given image content using Google Cloud Vision API.

    Parameters:
        client (vision.ImageAnnotatorClient): The Vision API client.
        image_content (bytes): The image content in bytes.

    Returns:
        list: List of detected text, bounding boxes, confidence scores, and alphanumeric status.
    """
    image = vision.Image(content=image_content)
    response = client.document_text_detection(image=image)
    result_list = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                bbox = paragraph.bounding_box
                text = ''
                confidence = 1

                for word in paragraph.words:
                    for symbol in word.symbols:
                        text += symbol.text
                        if symbol.confidence < confidence:
                            confidence = symbol.confidence

                # Add the strict alphanumeric status check
                alphanumeric_status = is_strictly_alphanumeric(text)

                result_list.append({
                    "text": text,
                    "bbox": {
                        "vertices": [
                            {"x": vertex.x, "y": vertex.y} for vertex in bbox.vertices
                        ]
                    },
                    "confidence": confidence,
                    "alphanumeric": alphanumeric_status
                })

    return result_list

def cloud_vision_inference(image_path):
    """
    Perform document text detection on an image using Google Cloud Vision API.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        list: List of detected text, bounding boxes, confidence scores, and alphanumeric status.
    """
    # Load the image content
    image_content = load_image(image_path)
    # Perform text detection
    text_results = perform_text_detection(client, image_content)
    
    # Filter results where alphanumeric status is True
    alphanumeric_results = [result for result in text_results if result['alphanumeric']]
    
    # Return the filtered results
    return alphanumeric_results

# Example usage
image_path = r"cropped_images\lot_obb.jpg"  # Update with the actual path to your image
ocr_results = cloud_vision_inference(image_path)
print(ocr_results)


# Print only the results where alphanumeric status is True
#for result in ocr_results:
    #print(result)
