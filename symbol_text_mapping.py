import math
#from symbol_inf import obb_results  # Import YOLO bounding boxes from another file
#from google_ocr import ocr_results    # Import OCR results from another file
import re
from symbol_inf import get_obb_results
from google_ocr import cloud_vision_inference

def is_alphanumeric(text):
    """
    Check if a given text is strictly alphanumeric (contains both letters and digits) or contains only digits.

    Parameters:
        text (str): The text to check.

    Returns:
        bool: True if the text is strictly alphanumeric or contains only digits, False otherwise.
    """
    has_letter = any(c.isalpha() for c in text)  # Check for letters
    has_digit = any(c.isdigit() for c in text)    # Check for digits

    # Return True if the text has both letters and digits or only digits
    return (has_letter and has_digit) or (not has_letter and has_digit)

def is_valid_date(text):
    """
    Check if a given text matches the date pattern 'yyyy-mm-dd'.

    Parameters:
        text (str): The text to check.

    Returns:
        bool: True if the text matches the date pattern, False otherwise.
    """
    # Regular expression for the date pattern 'yyyy-mm-dd'
    date_pattern = r'\b\d{4}-\d{2}-\d{2}\b'
    
    # Check if the text matches the pattern
    return bool(re.match(date_pattern, text))

def calculate_center_yolo(bbox):
    """
    Calculate the center of a bounding box in YOLO format.

    Parameters:
        bbox (dict): Bounding box with 'obb' details.

    Returns:
        tuple: Center (x, y) of the bounding box.
    """
    x_center = bbox['obb']['x_center']
    y_center = bbox['obb']['y_center']
    return x_center, y_center

def calculate_center_ocr(bbox):
    """
    Calculate the center of a bounding box in OCR format.

    Parameters:
        bbox (dict): Bounding box with vertices.

    Returns:
        tuple: Center (x, y) of the bounding box.
    """
    # Extract vertices
    vertices = bbox['vertices']  
    
    # Get all x and y coordinates from vertices
    x_coords = [vertex['x'] for vertex in vertices]
    y_coords = [vertex['y'] for vertex in vertices]
    
    # Calculate the center
    center_x = (min(x_coords) + max(x_coords)) / 2
    center_y = (min(y_coords) + max(y_coords)) / 2
    return center_x, center_y

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
        point1 (tuple): (x, y) coordinates of the first point.
        point2 (tuple): (x, y) coordinates of the second point.

    Returns:
        float: The Euclidean distance.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def extract_number(text, keyword):
    """
    Extract the first number after the keyword (e.g., 'LOT', 'REF') in the text.

    Parameters:
        text (str): The input string to process.
        keyword (str): The keyword to identify the number following it.

    Returns:
        str: The extracted number or the original text if no number is found.
    """
    # Split the text by keyword to search for the number after it
    parts = text.split(keyword)
    
    # If the keyword is present, find the first number after the keyword
    if len(parts) > 1:
        # Extract the part after the keyword and find the number
        candidate_text = parts[1].strip()  # Get the text after the keyword
        result = ''.join([char for char in candidate_text if char.isdigit()])
        
        # If a number is found, return it
        if result:
            return result
    
    # Return the original text if no number is found
    return text


def remove_unwanted_text(type, result):
    result = re.sub(r'\[lt\]', '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()
    if type == 'lot_no':
        #result = extract_number(result, 'LOT')
        result = result.lower()
        #result = result.replace('lot', '')
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
        result = result.replace('[lt]', '')
        result = result.replace('[', '')
        result = result.replace(']', '')
        result = result.replace('la', '')
        result = result.replace('lo', '')
        result = result.replace('t', '')
        result = result.replace('(10)', '')
        result = result.upper()
        
        

    elif type == 'ref_no':
        #result = extract_number(result, 'REF')
        result = result.lower()
        result = result.replace('ref', '')
        result = result.replace('reference', '')
        result = result.replace('batch', '')
        result = result.replace('code', '')
        result = result.replace(':', '')
        result = result.replace('.', '')
        result = result.replace('no', '')
        result = result.replace('number', '')
        result = result.replace('ref', '')
        result = result.replace('rep', '')
        result = result.replace('(10)', '')
        result = result.replace('catalogue', '')
        result = result.replace('catalog', '')
        result = result.replace(':', " ")
        result = result.replace('ret', '')
        result = result.replace('ree', '')
        result = result.replace('leb', '')
        result = result.replace('[', '')
        result = result.replace(']', '')
        result = result.replace('rf', '')
        result = result.upper()
        

    elif type == 'expiry':
        result = re.sub(r'(?i)useby', '', result).strip()
        result = result.lower()
        result = result.replace('lot', '')
        result = result.replace('ref', '')
        result = result.replace(':', '')
        result= result.replace('number', '')
        result = result.replace('exp', '')
        result = result.replace('use-by', '')
        result = result.replace('.', '')
        result =  result.replace('date', '')
        result = result.upper()
        #result = extract_number(result, 'USE_BY')
    #print(result)
    return result

def is_to_the_right(yolo_bbox, ocr_bbox_center):
    """
    Check if the OCR bounding box center is strictly to the right of the YOLO bounding box
    and within the vertical bounds of the YOLO bounding box.
    
    Parameters:
        yolo_bbox (dict): YOLO bounding box in OBB format.
        ocr_bbox_center (tuple): (x, y) coordinates of the OCR bounding box center.

    Returns:
        bool: True if the OCR bounding box is strictly to the right of the YOLO bounding box and
              within its vertical bounds.
    """
    x_center_yolo = yolo_bbox['obb']['x_center']
    y_center_yolo = yolo_bbox['obb']['y_center']
    yolo_height = yolo_bbox['obb']['height']

    # OCR bounding box should be to the right and within the vertical bounds of the YOLO bounding box
    return ocr_bbox_center[0] > x_center_yolo and (y_center_yolo - yolo_height / 2) <= ocr_bbox_center[1] <= (y_center_yolo + yolo_height / 2)

def find_closest_alphanumeric(obb_results, ocr_results):
    """
    Find the closest alphanumeric bounding boxes for all detections of "ref", "lot", and "expiry" classes.

    Parameters:
        obb_results (list): List of YOLO bounding boxes for "ref", "lot", and "expiry".
        ocr_results (list): List of OCR detected bounding boxes.

    Returns:
        dict: Lists of closest alphanumeric pairs for each class ("ref", "lot", "expiry").
    """
    closest_pairs = {}

    # Extract relevant YOLO bounding boxes for "ref", "lot", and "expiry"
    yolo_ref = [det for det in obb_results if det['class'] == 'ref']
    yolo_lot = [det for det in obb_results if det['class'] == 'lot']
    yolo_expiry = [det for det in obb_results if det['class'] == 'expiry']

    # Extract strictly alphanumeric OCR bounding boxes
    ocr_alphanumeric = [det for det in ocr_results if is_alphanumeric(det['text'])]

    # Find closest OCR for each "ref" detection
    for ref in yolo_ref:
        ref_center = calculate_center_yolo(ref)
        closest_ocr = None
        min_distance = float('inf')

        for ocr in ocr_alphanumeric:
            ocr_center = calculate_center_ocr(ocr['bbox'])
            distance = calculate_distance(ref_center, ocr_center)

            if is_to_the_right(ref, ocr_center) and distance < min_distance:
                min_distance = distance
                closest_ocr = ocr

        if closest_ocr:
            closest_ocr['text'] = remove_unwanted_text('ref_no', closest_ocr['text'])
        closest_pairs[ref['class']] = {
                "ref": ref,
                "closest_alphanumeric": closest_ocr
            }

    # Find closest OCR for each "lot" detection
    for lot in yolo_lot:
        lot_center = calculate_center_yolo(lot)
        closest_ocr = None
        min_distance = float('inf')

        for ocr in ocr_alphanumeric:
            ocr_center = calculate_center_ocr(ocr['bbox'])
            distance = calculate_distance(lot_center, ocr_center)

            if is_to_the_right(lot, ocr_center) and distance < min_distance:
                min_distance = distance
                closest_ocr = ocr

        if closest_ocr:
            closest_ocr['text'] = remove_unwanted_text('lot_no', closest_ocr['text'])
        closest_pairs[lot['class']] = {
                "lot": lot,
                "closest_alphanumeric": closest_ocr
            }

    # Find closest OCR for each "expiry" detection
    for expiry in yolo_expiry:
        expiry_center = calculate_center_yolo(expiry)
        closest_ocr = None
        min_distance = float('inf')

        for ocr in ocr_results:
            processed_text = remove_unwanted_text('expiry', ocr['text'])

            if is_valid_date(processed_text):
                ocr_center = calculate_center_ocr(ocr['bbox'])
                distance = calculate_distance(expiry_center, ocr_center)

                if ocr_center[0] >= expiry_center[0] and distance < min_distance:
                    min_distance = distance
                    closest_ocr = ocr

        if closest_ocr:
            closest_ocr['text'] = remove_unwanted_text('expiry', closest_ocr['text'])
        closest_pairs[expiry['class']] = {
                    "expiry": expiry,
                    "closest_alphanumeric": closest_ocr
                }

    return closest_pairs

# Find the closest alphanumeric bounding boxes
# Find the closest alphanumeric bounding boxes
obb_results = get_obb_results("test16.png")
ocr_results = cloud_vision_inference("angle_corrected_images/89019e7a6ac411ef822c42010a800003.jpg")
closest_pairs = find_closest_alphanumeric(obb_results, ocr_results)

# Output the results for both "ref" and "lot"
for class_key in ['ref', 'lot', 'expiry']:
    if class_key in closest_pairs:
        # Get the corresponding YOLO class data
        yolo_data = closest_pairs[class_key]

        # Print all relevant information
        print(f"Class: {class_key.capitalize()}")
        print(f"YOLO Data: {yolo_data[class_key]}")
        
        if yolo_data['closest_alphanumeric']:
            ocr_text = yolo_data['closest_alphanumeric']['text']
            ocr_bbox = yolo_data['closest_alphanumeric']['bbox']
            print(f"Closest Alphanumeric OCR Text: {ocr_text}")
            print(f"Closest OCR Bounding Box: {ocr_bbox}")
        else:
            print("Closest Alphanumeric OCR Text: None")
        
        print("\n")  # Print a newline for better readability



##############################DOUBTS################################################3
#what will be the min lngth of lot
#can lot be alphanumeric(test33.jpg)




#smaller patches
#paddle ocr results
#text detection check with yolo 