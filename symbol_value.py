import math
import cv2
import numpy as np
import os
import re
#from symbol_inf import obb_results  # Import YOLO bounding boxes from another file
#from google_ocr import ocr_results
from symbol_inf import get_obb_results
from google_ocr import cloud_vision_inference
from fetch_ref import get_nearest_ref_number
import pandas as pd

obb_inside_ocr = []
ref_processed = False
lot_processed = False
expiry_processed = False
# Function to check if a point is inside a bounding box
def is_point_inside_bbox(x_center, y_center, ocr_bbox):
    x_min = min(ocr_bbox['vertices'][0]['x'], ocr_bbox['vertices'][3]['x'])
    x_max = max(ocr_bbox['vertices'][1]['x'], ocr_bbox['vertices'][2]['x'])
    y_min = min(ocr_bbox['vertices'][0]['y'], ocr_bbox['vertices'][1]['y'])
    y_max = max(ocr_bbox['vertices'][2]['y'], ocr_bbox['vertices'][3]['y'])
    
    return x_min <= x_center <= x_max and y_min <= y_center <= y_max

def check_yolo_with_ocr(obb_results, ocr_results):
    """
    Check if any YOLO bounding boxes (OBBs) are inside any Google OCR bounding boxes.
    
    Parameters:
        obb_results (list): List of YOLO bounding boxes.
        ocr_results (list): List of OCR detected bounding boxes.
    
    Returns:
        dict: A dictionary with YOLO bounding boxes as keys and a boolean value indicating
              whether they are inside any OCR bounding box.
    """
    results = {}
    found_overlap = False  # Flag to check if there is at least one overlap

    for obb in obb_results:
        obb_center = calculate_center_yolo(obb)  # Get the center of the YOLO bounding box
        obb_x_center, obb_y_center = obb_center
        
        # Convert `obb` to a tuple key (hashable)
        obb_key = (obb['obb']['x_center'], obb['obb']['y_center'], obb['obb']['width'], obb['obb']['height'])
        # Flag to indicate if the OBB is inside any OCR bounding box
        is_inside_any_ocr = False

        for ocr in ocr_results:
            # Check if the YOLO bounding box center is inside the OCR bounding box
            if is_point_inside_bbox(obb_x_center, obb_y_center, ocr['bbox']):
                is_inside_any_ocr = True
                found_overlap = True  # Set overlap flag to True
                break  # Exit the loop once we find that it's inside any OCR box

        # Store the result for the current OBB
        results[obb_key] = is_inside_any_ocr

    if found_overlap:
        return results  # Proceed if at least one overlap is found
    else:
        # Handle case when no overlap is found, such as skipping further operations
        return None
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

def extract_date(text):
    # Define the regex pattern to match yyyy-mm-dd format with two digits for month and day
    pattern = r'(\d{4}-\d{2}-\d{2})'
    
    # Search for the date pattern in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the matched date, otherwise return None
    if match:
        return match.group(1)
    return ""

def remove_text_after_keyword(text, keyword):
    """
    Remove the keyword and any text that follows it.

    Parameters:
        text (str): The original text.
        keyword (str): The keyword to remove from the text.

    Returns:
        str: The text without the keyword and any characters after it.
    """
    keyword_pos = text.lower().find(keyword.lower())
    if keyword_pos != -1:
        return text[:keyword_pos].strip()
    return text

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
        result = result.replace('mber', '')
        result = result.replace('catalogue', '')
        result = result.replace('"DORIOPIORQUE"', '')
        result = result.replace('"', '')
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
        result = result.replace('expdate:','')
        result = result.replace(':','')
        result = result.upper()

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
    Find the closest alphanumeric bounding boxes for "ref" and "lot" classes.

    Parameters:
        obb_results (list): List of YOLO bounding boxes for "ref" and "lot".
        ocr_results (list): List of OCR detected bounding boxes.

    Returns:
        dict: Closest alphanumeric pairs for "ref" and "lot".
    """
    closest_pairs = {}
    ref_closest_ocr = []
    lot_closest_ocr = []
    global ref_processed, lot_processed, expiry_processed
    # Extract relevant YOLO bounding boxes for "ref" and "lot"
    yolo_ref = [det for det in obb_results if det['class'] == 'ref']
    yolo_lot = [det for det in obb_results if det['class'] == 'lot']
    yolo_expiry = [det for det in obb_results if det['class'] == 'expiry']

    # Extract strictly alphanumeric OCR bounding boxes
    ocr_alphanumeric = [det for det in ocr_results if is_alphanumeric(det['text'])]

    #print("yolo ref===================",yolo_ref)
    if not ref_processed:
        ref_processed = True
        print("inside ref")
        for ref in yolo_ref:
            ref_center = calculate_center_yolo(ref)
            closest_ocr = None
            min_distance = float('inf')

            for ocr in ocr_alphanumeric:
                ocr_center = calculate_center_ocr(ocr['bbox'])
                distance = calculate_distance(ref_center, ocr_center)
                #print("dist===================", distance)

                # Check if OCR bounding box is strictly to the right and within vertical bounds of the "ref" symbol
                print(is_to_the_right(ref, ocr_center))
                if is_to_the_right(ref, ocr_center) and distance < min_distance:
                    min_distance = distance
                    closest_ocr = ocr
                    #print("ref================================", closest_ocr)

            # Preprocess the closest OCR text for "ref"
            if closest_ocr:
                closest_ocr['text'] = remove_unwanted_text('ref_no', closest_ocr['text'])
                #closest_ocr['text'] = get_nearest_ref_number(closest_ocr['text'])
                #print("is this date ?========================", closest_ocr['text'])
                if is_valid_date(closest_ocr['text']):
                    print(is_valid_date(closest_ocr['text']))
                    pass
                else:   
                    closest_ocr['text'] = remove_text_after_keyword(closest_ocr['text'], "lot") 
                    ref_closest_ocr.append(closest_ocr)


         # Find the closest OCR with the highest confidence
        if ref_closest_ocr:
            print(ref_closest_ocr)
            for ref_ocr in ref_closest_ocr:
                if ref['class'] in closest_pairs and closest_pairs[ref['class']]['closest_alphanumeric'] is not None:
                    if ref_ocr is not None:
                            temp =  closest_pairs[ref['class']]['closest_alphanumeric'] 
                            #print("closest pairs=====================",closest_pairs)
                            conf =  temp['confidence']
                            #print("confidence================", conf)
                            if temp is None:
                                closest_pairs[ref['class']] = {
                                "ref": ref,
                                "closest_alphanumeric": ref_ocr
                                } 
                else:
                    closest_pairs[ref['class']] = {
                    "ref": ref,
                    "closest_alphanumeric": ref_ocr
                    }


# Repeat similar changes for the "lot" part as well.


    # Handle "lot" class with strict and linear conditions
    if not lot_processed:
        lot_processed = True
        for lot in yolo_lot:
            lot_center = calculate_center_yolo(lot)
            closest_ocr = None
            min_distance = float('inf')

            # First, check with the strict condition
            for ocr in ocr_alphanumeric:
                ocr_center = calculate_center_ocr(ocr['bbox'])
                distance = calculate_distance(lot_center, ocr_center)
                #print("distance", distance)
                #print("lot=======",lot)
                #print("ocr_center==============,",ocr_center)
                # Check if OCR bounding box is strictly to the right and within vertical bounds of the "lot" symbol
                if is_to_the_right(lot, ocr_center) and distance < min_distance:
                    min_distance = distance
                    closest_ocr = ocr
                    #print("closest ocr", closest_ocr)

            # If no closest OCR is found with the strict condition, try with a linear check
                # if closest_ocr is None or len(closest_ocr['text']) < 5:
                #     for ocr in ocr_alphanumeric:
                #         ocr_center = calculate_center_ocr(ocr['bbox'])
                #         distance = calculate_distance(lot_center, ocr_center)

                #         # Check if OCR bounding box is to the right and check the distance
                #         if ocr_center[0] >= lot_center[0] and distance < min_distance:
                #             min_distance = distance
                #             closest_ocr = ocr

            # Preprocess the closest OCR text for "lot"
            if closest_ocr:
                closest_ocr['text'] = remove_unwanted_text('lot_no', closest_ocr['text'])
                #print("why date?",closest_ocr['text'])
                
                date_value = extract_date(closest_ocr['text'])
                #print("extracted date:", date_value)
                #print(is_valid_date(date_value))
                #print("is it bool?",type(is_valid_date(date_value)))
                if is_valid_date(date_value):
                    print("valid date==========")
                    
                else:   
                    #closest_ocr['text'] = remove_text_after_keyword(closest_ocr['text'], "ref") 
                    lot_closest_ocr.append(closest_ocr)
        #print("lot closest ocr ==================", lot_closest_ocr)
        #input("enter")
        if lot_closest_ocr:
            
            for lot_ocr in lot_closest_ocr:
                if lot['class'] in closest_pairs and closest_pairs[lot['class']]['closest_alphanumeric'] is not None:
                    if lot_ocr is not None:
                            temp =  closest_pairs[lot['class']]['closest_alphanumeric'] 
                            print("closest pairs=====================",closest_pairs)
                            conf =  temp['confidence']
                            print("confidence================", conf)
                            if temp is None:
                                closest_pairs[lot['class']] = {
                                "lot": lot,
                                "closest_alphanumeric": lot_ocr
                                } 
                else:
                    closest_pairs[lot['class']] = {
                    "lot": lot,
                    "closest_alphanumeric": lot_ocr
                    }

    
        # Process for "expiry" class
    if not expiry_processed:
        expiry_processed = True
        print("yolo_expiry=========", yolo_expiry)
        for expiry in yolo_expiry:
            print("each expiry", expiry)
            expiry_center = calculate_center_yolo(expiry)
            closest_ocr = None
            min_distance = float('inf')

            for ocr in ocr_results:
                    # Process OCR text for expiry before checking the date pattern
                    processed_text = remove_unwanted_text('expiry', ocr['text'])
                    processed_text = extract_date(ocr['text'])
                    #print("processed date==========", processed_text)
                    
                    if is_valid_date(processed_text):  # Check the processed text for valid date
                        ocr_center = calculate_center_ocr(ocr['bbox'])
                        distance = calculate_distance(expiry_center, ocr_center)
                        if is_to_the_right(expiry, ocr_center) and distance < min_distance:
                            min_distance = distance
                            closest_ocr = ocr
                            print("right ocr", closest_ocr)
                        if closest_ocr is None :
                            for ocr in ocr_alphanumeric:
                                ocr_center = calculate_center_ocr(ocr['bbox'])
                                distance = calculate_distance(expiry_center, ocr_center)
                                # Check if OCR bounding box is to the right and check the distance
                                if ocr_center[0] >= expiry_center[0] and distance < min_distance:
                                    min_distance = distance
                                    closest_ocr = ocr
                                    print("closest ocr ", closest_ocr)

                    if closest_ocr:
                        closest_ocr['text'] = extract_date(closest_ocr['text'])

                    if expiry['class'] in closest_pairs and closest_pairs[expiry['class']]['closest_alphanumeric'] is not None:
                        if closest_ocr is not None:
                            temp =  closest_pairs[expiry['class']]['closest_alphanumeric'] 
                            #print("closest pairs=====================",closest_pairs)
                            conf =  temp['confidence']
                            #print("confidence================", conf)
                            #print("temp===========", temp)
                            if temp is None or conf < closest_ocr['confidence']:
                                closest_pairs[expiry['class']] = {
                                "expiry": expiry,
                                "closest_alphanumeric": closest_ocr
                                } 
                    else:
                        #print("i am in else")
                        closest_pairs[expiry['class']] = {
                        "expiry": expiry,
                        "closest_alphanumeric": closest_ocr
                        }
    print("==================================================================================================================================",closest_pairs)
    return closest_pairs

map_keys = {
    "ref": "ref_no",
    "lot": "lot_no",
    "expiry": "use_by"
}

def process_images_in_directory(folder_path = None):
    global obb_inside_ocr ,ref_processed ,lot_processed ,expiry_processed
    table_list = []
    print("folder path: ", folder_path)
    for dirpath, dirnames, filenames in os.walk(folder_path): 
        print(filenames)
        for file in filenames:
            row_dict = {}
            row_dict["image_name"] = file
            file = os.path.join(folder_path, file)
            image = cv2.imread(file)
            obb_results = get_obb_results(file)
            ocr_results = cloud_vision_inference(file)
            print("obb results========= ", obb_results)
            print("ocr results=========== ", ocr_results)
            
            obb_check_results = check_yolo_with_ocr(obb_results, ocr_results)
            print("obb check results: ", obb_check_results)
            if obb_check_results is None:
                
                print("class key: ", class_key)
                closest_pairs = find_closest_alphanumeric(obb_results, ocr_results)
                #print("=========================find closest alphanumeric",closest_pairs)
                print("closest pair: ", closest_pairs)
                for class_key in ['ref', 'lot', 'expiry']:
                    if class_key in closest_pairs:
                        # Get the corresponding YOLO class data
                        yolo_data = closest_pairs[class_key]

                        # Print all relevant information
                        #print(f"Class: {class_key.capitalize()}")
                        #print(f"YOLO Data: {yolo_data[class_key]}")
                        
                        if yolo_data['closest_alphanumeric']:
                            ocr_text = yolo_data['closest_alphanumeric']['text']
                            ocr_bbox = yolo_data['closest_alphanumeric']['bbox']
                            print(f"Closest Alphanumeric OCR Text for {class_key.capitalize()}: {ocr_text}")
                            row_dict[map_keys[class_key]] = ocr_text
                            #print(f"Closest OCR Bounding Box: {ocr_bbox}")
                        else:
                            row_dict[map_keys[class_key]] = "None"
                            print(f"Closest Alphanumeric OCR Text for {class_key.capitalize()}: None")
                        
                        print("\n")  # Print a newline for better readabilit
            else:

                for obb, is_inside in obb_check_results.items():
                    if is_inside:
                        #print(f"YOLO bounding box {obb} is inside an OCR bounding box.")
                        output_dir = 'cropped_images'
                        os.makedirs(output_dir, exist_ok=True)
                        for obb in obb_results:
                            obb_center_x = obb['obb']['x_center']
                            obb_center_y = obb['obb']['y_center']
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
                                                #print(f"Valid extended OBB for class {obb['class']}: {obb}")
                                                obb_inside_ocr.append(obb)  # Add only if the condition is met

                                                # Crop the image based on the updated OBB
                                                cropped_image = crop_image(image, obb)
                                                
                                                
                                                # Save the cropped image
                                                save_path = os.path.join(output_dir, f"{obb['class']}_obb.jpg")
                                                cv2.imwrite(save_path, cropped_image)

                                                ocr_result = cloud_vision_inference(save_path)
                                                for result in ocr_result:
                                                    result['text'] = remove_unwanted_text('ref_no', result['text'])
                                                    result['text'] = remove_text_after_keyword(result['text'], "lot")
                                                    #result['text'] = get_nearest_ref_number(result['text'])
                                                    #print("====================bbox failure", result)
                                                    row_dict[map_keys[obb['class']]] = result['text']
                                                    print(f"OCR result for {obb['class']}: {result['text']}")

                                        if obb['class'] == 'lot' and not lot_processed:
                                            lot_processed = True
                                            obb = extend_bbox_to_right(obb, ocr['bbox'])
                                            
                                            # Check if the obb_right is equal to the new_left after extension
                                            obb_right = obb_center_x + (obb['obb']['width'] / 2)
                                            new_left = obb_right  # new_left is the right edge of the original OBB
                                            
                                            if math.isclose(obb_right, new_left, abs_tol=1e-3):
                                                #print(f"Valid extended OBB for class {obb['class']}: {obb}")
                                                obb_inside_ocr.append(obb)  # Add only if the condition is met

                                                # Crop the image based on the updated OBB
                                                cropped_image = crop_image(image, obb)
                                                
                                                
                                                # Save the cropped image
                                                save_path = os.path.join(output_dir, f"{obb['class']}_obb.jpg")
                                                cv2.imwrite(save_path, cropped_image)

                                                ocr_result = cloud_vision_inference(save_path)
                                                for result in ocr_result:
                                                    result['text'] = remove_unwanted_text('lot_no', result['text'])
                                                    result['text'] = remove_text_after_keyword(result['text'], "ref")
                                                    #print("====================bbox failure", result)
                                                    row_dict[map_keys[obb['class']]] = result['text']
                                                    print(f"OCR result for {obb['class']}: {result['text']}")

                                                    


                                    #break  # Break after processing the first matching OCR bbox
                                
                    else:
                        #print(f"YOLO bounding box {obb} is not inside any OCR bounding box.")
                        closest_pairs = find_closest_alphanumeric(obb_results, ocr_results)
                        print("Else =================== find closest alphanumeric",closest_pairs)
                        for class_key in ['ref', 'lot', 'expiry']:
                            if class_key in closest_pairs:
                                # Get the corresponding YOLO class data
                                yolo_data = closest_pairs[class_key]

                                # Print all relevant information
                                #print(f"Class: {class_key.capitalize()}")
                                #print(f"YOLO Data: {yolo_data[class_key]}")
                                
                                if yolo_data['closest_alphanumeric']:
                                    ocr_text = yolo_data['closest_alphanumeric']['text']
                                    ocr_bbox = yolo_data['closest_alphanumeric']['bbox']
                                    print(f"Closest Alphanumeric OCR Text for {class_key.capitalize()}: {ocr_text}")
                                    row_dict[map_keys[class_key]] = ocr_text
                                    #print(f"Closest OCR Bounding Box: {ocr_bbox}")
                                else:
                                    print(f"Closest Alphanumeric OCR Text for {class_key.capitalize()}: None")
                                    row_dict[map_keys[class_key]] = "None"
                                
                                print("\n")  # Print a newline for better readability
            obb_inside_ocr = []
            ref_processed = False
            lot_processed = False
            expiry_processed = False
            print(row_dict)
            table_list.append(row_dict)
        result_df = pd.DataFrame(table_list)
        result_df.to_csv('ocr_prediction.csv', index=False)

    
    print("final resultz: ", table_list)
process_images_in_directory("test_sample")

# Load the image where the OBBs are located
# Replace 'image_path.jpg' with the actual path to your image
# image_path = 'test16.png'
# image = cv2.imread(image_path)
# obb_results = get_obb_results(image_path)
# ocr_results = cloud_vision_inference(image_path)

# # Lists to store updated OBB detections
# obb_inside_ocr = []
# ref_processed = False
# lot_processed = False
# expiry_processed = False
# # Iterate over YOLO OBB results
# obb_check_results = check_yolo_with_ocr(obb_results, ocr_results)
# #print(obb_check_results)
# if obb_check_results is None:
#      closest_pairs = find_closest_alphanumeric(obb_results, ocr_results)
#      for class_key in ['ref', 'lot', 'expiry']:
#         if class_key in closest_pairs:
#             # Get the corresponding YOLO class data
#             yolo_data = closest_pairs[class_key]

#             # Print all relevant information
#             #print(f"Class: {class_key.capitalize()}")
#             #print(f"YOLO Data: {yolo_data[class_key]}")
            
#             if yolo_data['closest_alphanumeric']:
#                 ocr_text = yolo_data['closest_alphanumeric']['text']
#                 ocr_bbox = yolo_data['closest_alphanumeric']['bbox']
#                 print(f"Closest Alphanumeric OCR Text for {class_key.capitalize()}: {ocr_text}")
#                 #print(f"Closest OCR Bounding Box: {ocr_bbox}")
#             else:
#                 print("Closest Alphanumeric OCR Text: None")
            
#             print("\n")  # Print a newline for better readabilit
# else:

#     for obb, is_inside in obb_check_results.items():
#         if is_inside:
#             #print(f"YOLO bounding box {obb} is inside an OCR bounding box.")
#             output_dir = 'cropped_images'
#             os.makedirs(output_dir, exist_ok=True)
#             for obb in obb_results:
#                 obb_center_x = obb['obb']['x_center']
#                 obb_center_y = obb['obb']['y_center']
#                 for ocr in ocr_results:
#                     if is_point_inside_bbox(obb_center_x, obb_center_y, ocr['bbox']):
#                             # If the class is 'ref' or 'lot', extend the bbox to the right
#                         if obb['class'] in ['ref', 'lot']:
#                             if obb['class'] == 'ref' and not ref_processed:
#                                 ref_processed = True
#                                 obb = extend_bbox_to_right(obb, ocr['bbox'])
                                
#                                 # Check if the obb_right is equal to the new_left after extension
#                                 obb_right = obb_center_x + (obb['obb']['width'] / 2)
#                                 new_left = obb_right  # new_left is the right edge of the original OBB
                                
#                                 if math.isclose(obb_right, new_left, abs_tol=1e-3):
#                                     #print(f"Valid extended OBB for class {obb['class']}: {obb}")
#                                     obb_inside_ocr.append(obb)  # Add only if the condition is met

#                                     # Crop the image based on the updated OBB
#                                     cropped_image = crop_image(image, obb)
                                    
                                    
#                                     # Save the cropped image
#                                     save_path = os.path.join(output_dir, f"{obb['class']}_obb.jpg")
#                                     cv2.imwrite(save_path, cropped_image)

#                                     ocr_result = cloud_vision_inference(save_path)
#                                     for result in ocr_result:
#                                         result['text'] = remove_unwanted_text('ref_no', result['text'])
#                                         print(f"OCR result for {obb['class']}: {result['text']}")

#                             if obb['class'] == 'lot' and not lot_processed:
#                                 lot_processed = True
#                                 obb = extend_bbox_to_right(obb, ocr['bbox'])
                                
#                                 # Check if the obb_right is equal to the new_left after extension
#                                 obb_right = obb_center_x + (obb['obb']['width'] / 2)
#                                 new_left = obb_right  # new_left is the right edge of the original OBB
                                
#                                 if math.isclose(obb_right, new_left, abs_tol=1e-3):
#                                     #print(f"Valid extended OBB for class {obb['class']}: {obb}")
#                                     obb_inside_ocr.append(obb)  # Add only if the condition is met

#                                     # Crop the image based on the updated OBB
#                                     cropped_image = crop_image(image, obb)
                                    
                                    
#                                     # Save the cropped image
#                                     save_path = os.path.join(output_dir, f"{obb['class']}_obb.jpg")
#                                     cv2.imwrite(save_path, cropped_image)

#                                     ocr_result = cloud_vision_inference(save_path)
#                                     for result in ocr_result:
#                                         result['text'] = remove_unwanted_text('lot_no', result['text'])
#                                         print(f"OCR result for {obb['class']}: {result['text']}")


#                         #break  # Break after processing the first matching OCR bbox
                    
#         else:
#             #print(f"YOLO bounding box {obb} is not inside any OCR bounding box.")
#             closest_pairs = find_closest_alphanumeric(obb_results, ocr_results)
#             for class_key in ['ref', 'lot', 'expiry']:
#                 if class_key in closest_pairs:
#                     # Get the corresponding YOLO class data
#                     yolo_data = closest_pairs[class_key]

#                     # Print all relevant information
#                     #print(f"Class: {class_key.capitalize()}")
#                     #print(f"YOLO Data: {yolo_data[class_key]}")
                    
#                     if yolo_data['closest_alphanumeric']:
#                         ocr_text = yolo_data['closest_alphanumeric']['text']
#                         ocr_bbox = yolo_data['closest_alphanumeric']['bbox']
#                         print(f"Closest Alphanumeric OCR Text for {class_key.capitalize()}: {ocr_text}")
#                         #print(f"Closest OCR Bounding Box: {ocr_bbox}")
#                     else:
#                         print("Closest Alphanumeric OCR Text: None")
                    
#                     print("\n")  # Print a newline for better readability

            
            
            
            
            