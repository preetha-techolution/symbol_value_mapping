from google_ocr import cloud_vision_inference
import re
from rlef_utils import send_to_rlef
import Config
import random
import threading
from datetime import datetime 
# ocr_model = cloud_vision_inference


def extract_data(data):
    texts = []
    bboxes = []
    confidences = []

    for entry in data:
        texts.append(entry['text'])
        
        # Extracting the bbox vertices
        bbox = entry['bbox']['vertices']
        bbox = [(vertex['x'], vertex['y']) for vertex in bbox]
        bboxes.append(bbox)
        
        # Extracting confidence
        confidences.append(entry['confidence'])
    
    return texts, bboxes, confidences






def text_ocr_process_single_image(img_path, type):
    result = cloud_vision_inference(img_path)
    try:
        if not result or not isinstance(result, list):
            # raise ValueError("Invalid result format from cloud_vision_inference")
            print("Invalid result format from cloud_vision_inference")

        # Extract labels, bounding boxes, and confidence scores
        rlef_labels, rlef_bboxes, rlef_confidences = extract_data(result)

        # Send segmentation to RLEF
        threading.Thread(target=send_to_rlef.send_segmentation_to_rlef, args=(
                                Config.RLEF_REQUEST_ID,           # unique_id
                                "backlog",                        # status
                                "csv",                            # csv
                                Config.RLEF_OCR_MODEL_ID,         # model
                                "ocr-image",                      # image_label
                                f"ocr-{Config.RLEF_REQUEST_ID}",  # tag                     
                                round(sum(rlef_confidences)/len(rlef_confidences),2),                             # confidence_score
                                "predicted",                      # prediction
                                "imageAnnotations",               # model_type
                                img_path,                         # filename
                                "image/png",                      # file_type
                                rlef_bboxes,                      # segmentations
                                rlef_confidences,                 # confidence_scores
                                rlef_labels                       # annotation_labels
                            )).start()
    
    except Exception as e:
        print(f"Unable to send to RLEF. Error: {e}")

    text = ""
    ocr_confidence = 0
    for entry in result:
        if type == 'barcode':
            if entry['text'].startswith('('):
                text +=" " + entry['text']
            else:
                continue 
        else:
            text += " " + entry['text']
        ocr_confidence += entry['confidence']
    return text

def digit_exists_in_text(text):
    return any(char.isdigit() for char in text)

def extract_number(ocr_text, number_type):
    """
    Extract the specified number (e.g., LOT number or REF number) from the OCR text.
    The number should start after the specified type (e.g., 'LOT' or 'REF') and begin with a character between A-Z or 0-9.

    Parameters:
    ocr_text (str): The OCR text from which to extract the number.
    number_type (str): The type of number to extract ('LOT' or 'REF').

    Returns:
    str: The extracted number, or an empty string if no valid number is found.
    """
    # Define regex patterns for different number types
    patterns  = {
        'USE_BY': r'(\d{4}-\d{2}-\d{2})',
        'REF': r'\b(?:[A-Za-z]*\d+[A-Za-z]*)(?:-[A-Za-z]*\d+[A-Za-z]*)*\b',
        'LOT': r'\b(?:[A-Za-z]*\d+[A-Za-z]*)(?:-[A-Za-z]*\d+[A-Za-z]*)*\b'
    }

    # Validate number_type and get the corresponding pattern
    if number_type not in patterns:
        raise ValueError(f"Unsupported number type: {number_type}")

    if number_type == 'REF' or number_type == 'LOT':
        final_text = ""
        ocr_text = ocr_text.split(' ')
        ## To be changed later
        if len(ocr_text) > 1:
            for word in ocr_text:
                if digit_exists_in_text(word):
                    word = word.replace(' ', '')
                    final_text += " " + word 

            return final_text 
    
        else:
            ocr_text = ocr_text[0]
            ocr_text = ocr_text.replace(' ', '')
            return ocr_text
        
    elif number_type == 'USE_BY':

        pattern = patterns[number_type]
        
        # Compile the pattern and search for matches
        number_pattern = re.compile(pattern, re.IGNORECASE)
        match = number_pattern.search(ocr_text)

        if match:
            # Return the extracted number
            return match.group(0)
        else:
            # Return an empty string if no valid number is found
            return ''

def extract_and_format_date(date_str):
    ## Convert DDMMYY to YYYY-MM-DD
    formatted_date = f'20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}'
    # formatted_date = f'{date_str[2:4]}-{date_str[:2]}-20{date_str[4:]}'
    return formatted_date




def parse_barcode(barcode):
    # Dictionary to store extracted details
    parsed_data = {}

    # Define patterns for each segment
    patterns = {
        'gtin_number': r'\(01\)(\d{14})',  # GTIN is 14 digits after (01)
        'lot_no': r'\(10\)([A-Za-z0-9]{1,20})',  # Lot number is 1 to 20 alphanumeric characters after (10)
        'manufacture_date': r'\(11\)(\d{6})',  # Manufacture date is 6 digits (YYMMDD) after (11)
        'serial_number': r'\(21\)([A-Za-z0-9]{1,14})',  # Serial number is 1 to 14 alphanumeric characters after (21)
        'best_by_date': r'\(15\)(\d{6})',  # Best by date is 6 digits (YYMMDD) after (15)
        'use_by': r'\(17\)(\d{6})'  # Expiry date is 6 digits (YYMMDD) after (17)
    }

    # Iterate through the patterns and extract matching values
    for key, pattern in patterns.items():
        match = re.search(pattern, barcode)
        if match:
            parsed_data[key] = match.group(1)

    date_types = ['best_by_date', 'use_by', 'manufacture_date']
    for key in parsed_data.keys():
        if key in date_types:
            parsed_data[key] = extract_and_format_date(parsed_data[key])
    return parsed_data


def extract_details_from_barcode(final_result, all_barcodes):
    
    for barcode in all_barcodes:
        extracted_data = parse_barcode(barcode)
        for key in extracted_data.keys():


            if key in final_result:
                final_result[key]['text'] = extracted_data[key]
                final_result[key]['crop_img_path'] = ""
            else:
                final_result[key] = {}
                final_result[key]['text'] = extracted_data[key]
                final_result[key]['crop_img_path'] = ""
            
    return final_result 



    # if 'barcode' in final_result:
    #     final_barcode = final_result['barcode']['text']
    # else:
    #     return final_result
    
    # if final_barcode != '':
    #     if extract_and_format_date(final_barcode) != '':
    #         final_use_by = extract_and_format_date(final_barcode)
    #         if 'use_by' not in final_result:
    #             final_result['use_by'] = {}
    #         final_result['use_by']['text'] = final_use_by        
    #         final_result['use_by']['crop_img_path'] = ""      

    # if '(21)' in final_barcode:
    #     final_serial_number = final_barcode.split('(21)')[-1]
    #     if 'serial_number' not in final_result:
    #         final_result['serial_number'] = {}
    #     final_result['serial_number']['text'] = final_serial_number
    #     final_result['serial_number']['crop_img_path'] = ""
    
    # if '(10)' in final_barcode:
    #     final_lot_no = final_barcode.split('(10)')[-1]
    #     try:
    #         final_lot_no = final_lot_no.split('(')[0]
    #         final_result['lot_no']['text'] = final_lot_no 
    #     except:
    #         print("it don't have any brackets after the lot number")
    # return final_result 
    

            
def remove_unwanted_text(type, result):
    if type == 'lot_no':
        result = result.lower()
        result = result.replace('lot', '')
        result = result.replace('number', '')
        result = result.replace('catalogue', '')
        result = result.replace('catalog', '')
        result = result.replace('batch', '')
        result = result.replace('code', '')
        result = result.replace(':', " ")
        result = result.upper()
        # result = self.filter_raw_text(result, 'LOT')
        result = extract_number(result, 'LOT')

    elif type == 'ref_no':
        result = result.lower()
        result = result.replace('reference', '')
        result = result.replace('batch', '')
        result = result.replace('code', '')
        result = result.replace(':', '')
        result = result.replace('number', '')
        result = result.replace('ref', '')
        result = result.replace('rep', '')
        result = result.replace('catalogue' ,'')
        result = result.replace('catalog', '')
        result = result.replace(':', " ")
        result = result.upper()
        result = extract_number(result, 'REF')
        # result = self.filter_raw_text(result, 'REF')
        
    elif type == 'use_by':
        result = extract_number(result, 'USE_BY')
    
    return result

